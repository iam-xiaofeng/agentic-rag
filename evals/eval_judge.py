"""**头号指标**：一次裁判调用，读 agent 自己引的关键句，打三个分。

    python eval_agentic.py --local ab_sol.jsonl --per-type 30     # 先跑出逐题 JSONL
    python eval_judge.py ab_sol.jsonl                             # 再打分
    python eval_judge.py ab_sol.jsonl --baseline ab_ds.jsonl      # 两个配置**同题配对**比较

━━━ 为什么换掉 `context_recall_fact`（子串匹配 gold 证据句）━━━

1. **过严**：gold 句只是数据集作者挑的某一句，同篇文章里另一句完全可能同样支撑答案 —— 判 0 分。
2. **冗余同权**：实测 inference 类平均 **3.36 条**证据里只有 **1.47 条**真正含答案，其余是氛围铺垫；
   漏掉照样答对，指标却照扣分。（comparison / temporal 多是"A 和 B 是不是都…"的**合取**问题，
   两条缺一不可，不算冗余 —— 所以**该由裁判逐题判哪些是必需的**，而不是一律要求全召回。）
3. **没有成本项**：搜 10 次 × 32 片段必然赢过搜 1 次 × 32，哪怕后 9 次全是重复。

━━━ 三个分 ━━━

    correct     答案是否与标准答案实质一致
    sufficient  **agent 自己引的那几句**够不够推出标准答案（= 你要的"召回"，按需判、不要求全覆盖）
    faithful    答案里的每个论断是否都能追回到它引的句子（忠诚度 / 反幻觉）

━━━ 一个守卫（确定性，`eval_agentic.py` 已算好）━━━

    cited_grounded   那些引用里，真能在 agent **实际检索到**的上下文中找到的比例

**没有它，这三个分会在最关键的地方骗人**：引用是 agent 的**自述**。
「没检索到」和「检索到了但没引出来」在裁判眼里长得一模一样 —— 会写引用的模型分高，
检索一样好但引用潦草的模型分低。这正好砸在"换模型带来的提升是真是假"这个问题上。
所以本脚本在 `cited_grounded` 偏低时**拒绝给出结论**。

━━━ 还有一条裁判治不好的 ━━━

**猜测下限是数据集的性质，不是指标的性质。** 本数据集 comparison 96% 是 Yes/No 且 60% 答 "Yes"、
temporal 89% 是 Yes/No —— **一个恒答 "Yes" 的空壳就有 0.60 / 0.49 的 correct**。
所以 `correct` 的绝对值永远要减掉地板才有意义（地板跑 `eval_answerability.py`，
它是**跑一次的常数**，不是每轮都测的指标）。

裁判固定为 `RAG_JUDGE_MODEL`（退到 `RAG_MODEL`）—— 换裁判 = 换尺子，新旧分数就不可比了（实验13）。
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import json
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rag.llm import build_judge

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

_PROMPT = """You are grading one answer produced by a retrieval-augmented QA system.

QUESTION:
{question}

GOLD ANSWER:
{gold}

GOLD EVIDENCE collected by the dataset author (NOT all of these are necessarily required \
to answer — decide for yourself which ones the question actually needs):
{facts}

SYSTEM ANSWER:
{answer}

SENTENCES THE SYSTEM SAYS IT RELIED ON:
{cited}

Grade three things, each a number from 0.0 to 1.0:

"correct" — does SYSTEM ANSWER match GOLD ANSWER in substance? Wording, extra explanation \
and formatting do not matter. 1.0 = same answer; 0.0 = different, missing, or a refusal. \
Use values in between only for partially-right answers.

"sufficient" — could a reader who saw ONLY the sentences the system relied on derive the \
GOLD ANSWER? Grade SUFFICIENCY, not coverage: do NOT require every gold evidence item to \
appear. If the question is conjunctive ("did both A and B ...", "which of X and Y ..."), \
every conjunct must be supported. Otherwise one sentence may well be enough. \
1.0 = the cited sentences establish the gold answer on their own; 0.5 = they establish \
part of it and the rest is a plausible leap; 0.0 = they do not support it at all, or \
there are no cited sentences.

"faithful" — is every factual claim in SYSTEM ANSWER traceable to the cited sentences? \
1.0 = fully traceable; 0.0 = the answer asserts specifics the cited sentences do not \
contain (hallucination). Ignore hedging, restating the question, and general framing.

Judge the three independently: an answer can be correct but unsupported (the model knew it \
without retrieving), or well-supported but wrong.

Reply with ONLY a JSON object, no prose and no code fence:
{{"correct": 0.0, "sufficient": 0.0, "faithful": 0.0, "why": "<one short sentence>"}}"""

_JSON = re.compile(r"\{.*\}", re.S)


def _gold(benchmark: str = "multihoprag") -> dict[str, dict]:
    """{问句: {answer, evidence_list:[{fact}]}}。两个评测集拍成同一形状，裁判 prompt 不用改。"""
    if benchmark == "musique":
        from rag.corpus_musique import load_questions
        ex, _ = load_questions()
        return {e.inputs["question"]: {"answer": e.outputs["reference"],
                                       "evidence_list": [{"fact": f} for f in
                                                         e.outputs["reference_contexts"]]}
                for e in ex}
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {r["query"]: r for r in raw}


def _load(path: str) -> list[dict]:
    return [json.loads(l) for l in
            pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _parse(txt: str) -> dict | None:
    """裁判的 JSON。解析不出来返回 None —— **不当 0 分**，单独统计（本项目栽过多次的那类坑）。"""
    m = _JSON.search(txt or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ("correct", "sufficient", "faithful"):
        v = d.get(k)
        if not isinstance(v, (int, float)):
            return None
        out[k] = min(1.0, max(0.0, float(v)))
    out["why"] = str(d.get("why", ""))[:200]
    return out


def _ci(d, boots: int = 20000) -> tuple[float, float, float]:
    a = np.asarray(list(d), float)
    if len(a) < 3:
        return (float(a.mean()) if len(a) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, len(a), size=(boots, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _score_dump(path: str, gold: dict, judge, concurrency: int, out: str | None) -> list[dict]:
    rows = [r for r in _load(path) if "errored" not in r and r.get("type") != "null"]
    jobs = []
    for r in rows:
        g = gold.get(r["question"])
        if not g or "answer" not in r:
            continue
        facts = [e["fact"] for e in (g.get("evidence_list") or []) if e.get("fact")]
        cited = r.get("cited") or []
        jobs.append((r, _PROMPT.format(
            question=r["question"], gold=g.get("answer", ""),
            facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)) or "(none)",
            answer=(r["answer"] or "(empty)").strip()[:4000],
            cited="\n".join(f"- {s}" for s in cited) or "(the system cited nothing)")))
    if not jobs:
        raise SystemExit(f"{path} 里没有可判的记录 —— 需要 answer/cited 字段，"
                         f"旧版 dump 没存，得用新版 eval_agentic.py --local 重跑。")

    def ask(job):
        for wait in (0, 5, 20, 60):
            if wait:
                time.sleep(wait)
            try:
                r = _parse(judge.invoke(job[1]).content)
                if r:
                    return r
            except Exception:      # noqa: BLE001 —— 网关抽风 / 解析失败，都重试
                pass
        return None

    print(f"{path}: 裁判 {len(jobs)} 次调用（每题 1 次）…", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        got = list(pool.map(ask, jobs))
    recs = []
    for (r, _), j in zip(jobs, got):
        # 空答案的确定性兜底：裁判会给 faithful=1.0（"没有论断，忠诚度平凡为真"），逻辑上说得通
        # 但在均值里就变成**打空的题反而加分**。三项一律归 0 —— 和本项目一贯的原则一致：
        # 失败必须可见，而且不能长得像成功。
        if j and not (r.get("answer") or "").strip():
            j = {**j, "correct": 0.0, "sufficient": 0.0, "faithful": 0.0,
                 "why": "[确定性兜底] agent 答案为空"}
        recs.append(r | {"judge": j})
    if out:
        pathlib.Path(out).write_text("\n".join(
            json.dumps({k: v for k, v in r.items() if k != "contexts"}, ensure_ascii=False)
            for r in recs), encoding="utf-8")
    return recs


def _table(recs: list[dict], title: str) -> None:
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")   # noqa: E731
    order = ["comparison", "inference", "temporal", "2hop", "3hop", "4hop"]
    types = sorted({r["type"] for r in recs}, key=lambda t: (order.index(t) if t in order else 99, t))
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'type':12s}{'n':>4}{'correct':>10}{'sufficient':>12}{'faithful':>10}"
          f"{'引用属实':>10}{'n_search':>10}")
    for t in [*types, "全部"]:
        d = [r for r in recs if (t == "全部" or r["type"] == t) and r["judge"]]
        if not d:
            continue
        print(f"{t:12s}{len(d):>4}"
              + "".join(f"{mean([r['judge'][k] for r in d]):>{w}.3f}"
                        for k, w in (("correct", 10), ("sufficient", 12), ("faithful", 10)))
              + f"{mean([r['cited_grounded'] for r in d if r.get('cited_grounded') == r.get('cited_grounded')]):>10.2f}"
              + f"{mean([r['n_search'] for r in d]):>10.2f}")

    dead = [r for r in recs if not r["judge"]]
    if dead:
        print(f"\n⚠️ {len(dead)}/{len(recs)} 题裁判连续失败/输出无法解析，已**排除**（不当 0 分）。")
    ok = [r for r in recs if r["judge"]]

    # 这把尺子最值钱的一格：老指标里这两类长得一模一样，现在能分开。
    para = [r for r in ok if r["judge"]["correct"] >= 1.0 and r["judge"]["sufficient"] == 0.0]
    unf = [r for r in ok if r["judge"]["faithful"] == 0.0]
    under = [r for r in unf if r["judge"]["sufficient"] >= 1.0]
    print(f"\n  ▸ **答对但没依据**（correct=1 且 sufficient=0）：{len(para)}/{len(ok)} 题。"
          f"\n    这是模型**凭参数化知识答对**的，不是检索的功劳 —— 老指标看不见这类："
          f"\n    `context_recall_fact` 把它记成检索失败，`correctness` 把它记成成功。")
    print(f"  ▸ **不忠诚**（faithful=0）：{len(unf)}/{len(ok)} 题，其中 {len(under)} 题 sufficient=1.0 "
          f"——\n    依据是够的，只是答案说了引用之外的话（**少引/添油加醋**）；"
          f"另 {len(unf) - len(under)} 题依据本就不够（**真的在无据发挥**）。"
          f"\n    注意 faithful 是拿答案跟**它自己引的那几句**比，不是跟全部检索上下文比。")

    cg = [r["cited_grounded"] for r in ok if r.get("cited_grounded") == r.get("cited_grounded")]
    if cg and mean(cg) < 0.8:
        print(f"\n⛔ 引用属实率仅 {mean(cg):.2f} —— agent 引的句子有相当一部分**不在它检索到的上下文里**。"
              f"\n   这说明它在改写或编造引用，**sufficient / faithful 两个分不可信**，先修 prompt 再看数。")
    nocite = [r for r in ok if not (r.get("cited") or []) and r.get("n_search", 0) > 0]
    if nocite:
        clean = [r for r in ok if r not in nocite]
        by = {t: sum(1 for r in nocite if r["type"] == t) for t in ("comparison", "inference", "temporal")}
        print(f"\n⚠️ {len(nocite)}/{len(ok)} 题查了却没吐 KEY EVIDENCE 块（{by}）—— 它们的 sufficient 被"
              f"机械判 0，是**格式问题不是检索问题**。剔掉后："
              + "  ".join(f"{k} {mean([r['judge'][k] for r in clean]):.3f}"
                          for k in ("correct", "sufficient", "faithful")))
    if any(r["type"] in ("comparison", "temporal") for r in recs):
        print("\n  correct 的绝对值**必须减地板**：本数据集 comparison 96% 是 Yes/No 且 59.8% 答 \"Yes\"，"
              "\n  恒答 \"Yes\" 的空壳就有 0.60。地板跑 `eval_answerability.py`（一次性常数）。")
    else:
        print("\n  MuSiQue 的猜测下限≈0（最高频答案仅占 1.4%），correct 可以直接读，不用减地板。")


def main() -> None:
    ap = argparse.ArgumentParser(description="裁判读 agent 自引的关键句，打 correct/sufficient/faithful 三分")
    ap.add_argument("dump", help="eval_agentic.py --local 产出的 JSONL（需含 answer/cited/contexts）")
    ap.add_argument("--baseline", default=None, help="另一个 dump，按 example_id **同题配对**比较")
    ap.add_argument("--benchmark", choices=["multihoprag", "musique"], default="multihoprag",
                    help="gold 从哪来。必须与产生 dump 的 eval_agentic.py --benchmark 一致")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None, help="把逐题判分写成 JSONL（便于复查裁判的 why）")
    args = ap.parse_args()

    gold, judge = _gold(args.benchmark), build_judge()
    recs = _score_dump(args.dump, gold, judge, args.concurrency, args.out)
    _table(recs, f"头号指标 · {args.dump}")
    if not args.baseline:
        return

    base = _score_dump(args.baseline, gold, judge, args.concurrency,
                       (args.out + ".base") if args.out else None)
    _table(base, f"头号指标 · {args.baseline}（基线）")

    by_id = {r["example_id"]: r for r in base if r["judge"]}
    pairs = [(r, by_id[r["example_id"]]) for r in recs if r["judge"] and r["example_id"] in by_id]
    print(f"\n{'=' * 78}\n同题配对：{args.dump} − {args.baseline}（n={len(pairs)}）\n{'=' * 78}")
    if len(pairs) < 3:
        print("  配对样本 <3，不出区间 —— 两个 dump 得抽同一批题（同 --per-type、同数据集）。")
        return
    print(f"{'指标':12s}{'差值':>10}{'  95% 置信区间':>22}")
    for k in ("correct", "sufficient", "faithful"):
        m, lo, hi = _ci([a["judge"][k] - b["judge"][k] for a, b in pairs])
        sig = "✅ 显著" if lo > 0 else ("⛔ 反向显著" if hi < 0 else "  跨 0，**不显著**")
        print(f"{k:12s}{m:>+10.3f}   [{lo:+.3f}, {hi:+.3f}]  {sig}")
    print("\n  区间跨 0 = 这批样本量分辨不出差异，**不能说谁更好**（实验15 就栽在把 n=15 的噪声当效应）。")


if __name__ == "__main__":
    main()
