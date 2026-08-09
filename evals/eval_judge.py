"""**头号指标**：一次裁判调用，三个分 —— 而三个分**各有各的参照系**。

    python evals/eval_judge.py runs/dumps/a.jsonl --benchmark musique
    python evals/eval_judge.py runs/dumps/新.jsonl --baseline runs/dumps/旧.jsonl   # 同题配对 + 95% CI

━━━ 2026-08 重构：为什么把 `faithful` 换成 `grounded` ━━━

旧的三个分里有**两个**是从 agent 的**自述引用**算出来的：

    sufficient  = 裁判读「agent 引的那几句」判够不够  → 同时在量"检索到没有"和"愿不愿意引"
    faithful    = 答案的论断能否追回「它自己引的那几句」→ **引得越少越容易满分**

于是这两个分在构造上就是**拮抗**的，任何改动只要碰到引用行为，两个分就一起动、方向相反。
实测三版提示词就是在同一条权衡曲线上滑：

    v1 什么都不要求          cited_grounded 0.94  faithful 0.524  n_search 1.81
    v2 "N 跳至少 N 行引用"    cited_grounded 0.77  faithful 0.619  ← 分涨了，因为它在**编引用**
    v3 给 `unsupported:` 出路 cited_grounded 0.87  faithful 0.643  n_search 2.14 ← 变省也变懒

**"忠诚度提不上去"不是模型不行，是这把尺子把被测对象和测量点绑在了一起。**

修法 —— 三个头号量各用一个**互不重叠**的参照系，且没有一个能被"多引/少引"操纵：

    correct    ← gold 答案             （裁判）
    grounded   ← **agent 实际检索到的全部上下文**（裁判）—— 不是它自选的那几句
    delivered  ← gold 证据 ∩ 检索上下文  （**确定性**，eval_agentic.py 已算好，零噪声）

`grounded` 拿全部检索上下文当参照系之后，"少引"不再能刷分；而"用参数化知识补一跳"
（实验26④ 那类答对但链条断的题）照样会被抓住 —— 因为那句话在检索上下文里根本不存在。

━━━ 降级为诊断的 ━━━

`sufficient`（自引句够不够）留着，但**只当引用行为的诊断**，不再当检索指标读；
`cited_grounded` 低时只作废 `sufficient` 一个分，`correct` / `grounded` 不受影响。
检索栈好不好，看确定性的 `delivered` / `fate_missing`，不必经过裁判。

━━━ 两条永远要记得 ━━━

裁判固定为 `endpoints.json` 的 `_roles.judge`（换裁判 = 换尺子，实验13）；
`correct` 的绝对值要减地板（`eval_ceiling.py` 一次性跑出地板/天花板）。
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
from rag.runctx import META_KEY, compare_meta, fmt_meta, read_dump

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
_CHUNK = re.compile(r"(?=\[source: )")
_JSON = re.compile(r"\{.*\}", re.S)

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

EVERYTHING THE SYSTEM ACTUALLY RETRIEVED (this is the full evidence available to it):
{retrieved}

Grade three things, each a number from 0.0 to 1.0. Judge them independently — an answer can be \
correct but ungrounded (the model knew it without retrieving), or well-grounded but wrong.

"correct" — does SYSTEM ANSWER match GOLD ANSWER in substance? Wording, extra explanation and \
formatting do not matter. 1.0 = same answer; 0.0 = different, missing, or a refusal. Use values \
in between only for partially-right answers.

"grounded" — is every factual claim in SYSTEM ANSWER supported by EVERYTHING THE SYSTEM ACTUALLY \
RETRIEVED? Grade against the retrieved passages, NOT against the sentences the system chose to \
quote — a claim that appears in the retrieved passages is grounded even if the system never \
quoted it. A claim the system could only have known from its own background knowledge is NOT \
grounded, however true it is. This is the anti-hallucination axis: it must not reward answering \
less or quoting less. 1.0 = every claim traceable to a retrieved passage; 0.0 = the central \
claim is nowhere in them. Ignore hedging, restating the question, and general framing.

"sufficient" — could a reader who saw ONLY the sentences the system says it relied on derive the \
GOLD ANSWER? Grade SUFFICIENCY, not coverage: do NOT require every gold evidence item to appear. \
If the question is conjunctive ("did both A and B ...", "which of X and Y ..."), every conjunct \
must be supported. Otherwise one sentence may well be enough. 1.0 = the cited sentences establish \
the gold answer on their own; 0.5 = they establish part of it and the rest is a plausible leap; \
0.0 = they do not support it at all, or there are no cited sentences.

Also list the claims in SYSTEM ANSWER that you could NOT find in the retrieved passages \
(verbatim fragments, at most 3, empty list if none) — this is what makes "grounded" auditable.

Reply with ONLY a JSON object, no prose and no code fence:
{{"correct": 0.0, "grounded": 0.0, "sufficient": 0.0, "ungrounded_claims": [], "why": "<one short sentence>"}}"""


def _gold(benchmark: str) -> dict[str, dict]:
    """{问句: {answer, evidence_list:[{fact}]}}。两个评测集拍成同一形状，裁判 prompt 不用改。

    ⚠️ benchmark 名要与 `eval_agentic.load_benchmark` **保持同一套**（含 `musique+1hop`）——
    这里曾经只认 `"musique"`，于是 `musique+1hop` 的 dump 一条都对不上 gold，
    直接报"没有可判的记录"。新增评测集时**两处都要改**。
    """
    if benchmark.startswith("musique"):
        from rag.corpus_musique import load_questions
        ex, _ = load_questions(with_1hop=benchmark.endswith("+1hop"))
        return {e.inputs["question"]: {"answer": e.outputs["reference"],
                                       "evidence_list": [{"fact": f} for f in
                                                         e.outputs["reference_contexts"]]}
                for e in ex}
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {r["query"]: r for r in raw}


def _retrieved_blob(contexts: list[str], cap: int) -> tuple[str, bool]:
    """把多轮检索的片段**去重**后拼成裁判要读的证据面。

    ⚠️ 为什么必须去重再截断，而不是直接截：k 大 / 搜得多的臂原始字符更多，一刀切截断会让它的
    `grounded` 系统性偏低 —— **尺子对被比的变量不中立**，正是实验19① 栽过的那个坑
    （`fact@k` 只查前 120 字符，于是小块系统性占便宜）。去重后绝大多数配置都进不到上限；
    真截到了会返回 True，逐题记下来，配对时若两臂截断率不同就不能比。
    """
    seen, out, n = set(), [], 0
    for ctx in contexts:
        for c in _CHUNK.split(ctx):
            c = c.strip()
            if not c or c in seen:
                continue
            seen.add(c)
            if n + len(c) > cap:
                return "\n\n".join(out), True
            out.append(c)
            n += len(c)
    return "\n\n".join(out), False


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
    for k in ("correct", "grounded", "sufficient"):
        v = d.get(k)
        if not isinstance(v, (int, float)):
            return None
        out[k] = min(1.0, max(0.0, float(v)))
    out["ungrounded_claims"] = [str(x)[:160] for x in (d.get("ungrounded_claims") or [])][:3]
    out["why"] = str(d.get("why", ""))[:200]
    return out


def _ci(d, boots: int = 20000) -> tuple[float, float, float]:
    a = np.asarray([x for x in d if x == x], float)
    if len(a) < 3:
        return (float(a.mean()) if len(a) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, len(a), size=(boots, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def score_dump(path: str, gold: dict, judge, concurrency: int, cap: int,
               out: str | None) -> tuple[dict, list[dict]]:
    meta, all_rows = read_dump(path)
    rows = [r for r in all_rows if "errored" not in r and r.get("type") != "null"]
    jobs = []
    for r in rows:
        g = gold.get(r["question"])
        if not g or "answer" not in r:
            continue
        facts = [e["fact"] for e in (g.get("evidence_list") or []) if e.get("fact")]
        blob, trunc = _retrieved_blob(r.get("contexts") or [], cap)
        r["ctx_truncated"] = trunc
        jobs.append((r, _PROMPT.format(
            question=r["question"], gold=g.get("answer", ""),
            facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)) or "(none)",
            answer=(r["answer"] or "(empty)").strip()[:4000],
            cited="\n".join(f"- {s}" for s in (r.get("cited") or [])) or "(the system cited nothing)",
            retrieved=blob or "(the system retrieved nothing)")))
    if not jobs:
        raise SystemExit(f"{path} 里没有可判的记录 —— 需要 answer/cited/contexts 字段。")

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
        # 空答案的确定性兜底：裁判会给 grounded=1.0（"没有论断，忠诚度平凡为真"），逻辑上说得通
        # 但在均值里就变成**打空的题反而加分**。三项一律归 0 —— 失败必须可见，且不能长得像成功。
        if j and not (r.get("answer") or "").strip():
            j = {**j, "correct": 0.0, "grounded": 0.0, "sufficient": 0.0,
                 "why": "[确定性兜底] agent 答案为空"}
        recs.append(r | {"judge": j})
    if out:
        # 判分文件也带 __meta__ 头（沿用被判 dump 的快照 + 记下裁判和上下文上限）——
        # 否则它下游再被 eval_ceiling.py --dump 读到时会被标成"旧格式、无法自证配置"。
        head = {META_KEY: {**meta, "judged_by": getattr(judge, "model_name", None) or "?",
                           "judge_ctx_chars": cap}}
        pathlib.Path(out).write_text("\n".join(
            [json.dumps(head, ensure_ascii=False)]
            + [json.dumps({k: v for k, v in r.items() if k != "contexts"}, ensure_ascii=False)
               for r in recs]), encoding="utf-8")
    return meta, recs


_MEAN = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")            # noqa: E731
_OK = lambda xs: [x for x in xs if x == x and x is not None]              # noqa: E731


# 「给出了具体答案」的判据：确定性、不经过裁判。**词表定义在 rag/agent.py，全项目唯一一份。**
from rag.agent import refused as _is_refusal


def answered(r: dict) -> bool:
    return not _is_refusal(r.get("answer"))


def _col(recs, key):
    if key in ("correct", "grounded", "sufficient"):
        return _OK([r["judge"][key] for r in recs if r["judge"]])
    if key == "answer_rate":
        return [1.0 if answered(r) else 0.0 for r in recs]
    if key == "correct_at_answered":
        return _OK([r["judge"]["correct"] for r in recs if r["judge"] and answered(r)])
    return _OK([r.get(key) for r in recs])


def table(recs: list[dict], meta: dict, title: str) -> None:
    order = ["2hop", "3hop", "4hop", "comparison", "inference", "temporal"]
    types = sorted({r["type"] for r in recs}, key=lambda t: (order.index(t) if t in order else 99, t))
    print(f"\n{'=' * 92}\n{title}\n  {fmt_meta(meta)}\n{'=' * 92}")
    print(f"{'':10s}{'':4s}│{'  ── 头号（各有各的参照系）──':>34s}│"
          f"{'  ── correct 拆开：肯不肯答 / 答了准不准 ──':>44s}│{'  ── 诊断 / 成本 ──':>26s}")
    print(f"{'type':10s}{'n':>4}│{'correct':>10}{'grounded':>10}{'delivered†':>13}│"
          f"{'给答案率':>11}{'correct@给了':>14}│{'引用属实':>10}{'n_search':>9}{'万字符':>8}")
    for t in [*types, "全部"]:
        d = [r for r in recs if (t == "全部" or r["type"] == t)]
        if not d:
            continue
        print(f"{t:10s}{len(d):>4}│"
              + "".join(f"{_MEAN(_col(d, k)):>{w}.3f}" for k, w in
                        (("correct", 10), ("grounded", 10), ("delivered", 13)))
              + "│" + "".join(f"{_MEAN(_col(d, k)):>{w}.3f}" for k, w in
                              (("answer_rate", 11), ("correct_at_answered", 14)))
              + "│" + f"{_MEAN(_col(d, 'cited_grounded')):>10.3f}"
              + f"{_MEAN(_col(d, 'n_search')):>9.3f}"
              + f"{_MEAN(_col(d, 'n_ctx_chars')) / 1e4:>8.2f}")
    print("  † delivered 是**确定性**的（gold 证据段 ∩ 检索上下文，零裁判噪声）——检索栈好不好看它，不必经过裁判。\n"
          "    correct ← gold 答案；grounded ← agent **实际检索到的全部上下文**（不是它自选的引用，"
          "所以「少引」不再能刷分）。\n"
          "  * **correct = 给答案率 × correct@给了**（近似）。分开看是必须的：「不肯答」和「答错了」"
          "在 correct 上都是 0、长得一模一样，\n"
          "    但前者该调**断言阈值**、后者该调**推理或检索**。实测同一个模型在 gold 上下文下拒答 19/88 题，"
          "而强制作答后其中多数是对的。")

    dead = [r for r in recs if not r["judge"]]
    if dead:
        print(f"\n⚠️ {len(dead)}/{len(recs)} 题裁判连续失败/输出无法解析，已**排除**（不当 0 分）。")
    ok = [r for r in recs if r["judge"]]
    if not ok:
        return

    para = [r for r in ok if r["judge"]["correct"] >= 1.0 and r["judge"]["grounded"] <= 0.5]
    print(f"\n  ▸ **答对但没依据**（correct=1 且 grounded≤0.5）：{len(para)}/{len(ok)} 题 —— "
          f"模型**凭参数化知识抄近路**补上了缺的那一跳，\n    答案碰巧对、推理链是断的。旧指标看不见这类："
          f"`context_recall_fact` 记成检索失败，`correct` 记成成功。")
    claims = [c for r in ok for c in (r["judge"].get("ungrounded_claims") or [])]
    if claims:
        print(f"  ▸ 裁判点名的**无依据论断** {len(claims)} 条，抽三条（grounded 是否判得住，看这里）：")
        for c in claims[:3]:
            print(f"      · {c[:110]}")

    trunc = [r for r in ok if r.get("ctx_truncated")]
    if trunc:
        print(f"\n⚠️ {len(trunc)}/{len(ok)} 题的检索上下文被截断后才送裁判 —— 这些题的 `grounded` 会**偏低**。"
              f"\n   配对比较时若两臂截断率不同，grounded 的差值不可信（--ctx-chars 调大重判）。")
    cg = _col(ok, "cited_grounded")
    if cg and _MEAN(cg) < 0.8:
        print(f"\n⛔ 引用属实率仅 {_MEAN(cg):.2f} —— agent 引的句子有相当一部分**不在它检索到的上下文里**（在编引用）。"
              f"\n   **只有 `sufficient` 因此作废**；`correct` / `grounded` / `delivered` 不依赖自述引用，照常读。")
    nocite = [r for r in ok if not (r.get("cited") or []) and r.get("n_search", 0) > 0]
    if nocite:
        print(f"\n⚠️ {len(nocite)}/{len(ok)} 题查了却没吐 KEY EVIDENCE 块 —— 它们的 `sufficient` 被机械判 0，"
              f"是**格式问题不是检索问题**（同样不影响 correct / grounded）。")
    print("\n  correct 的绝对值要减地板：`python evals/eval_ceiling.py` 一次跑出 地板/实测/天花板 三条线。")


_PAIRED = [("correct", "头号"), ("grounded", "头号"), ("delivered", "头号·确定性"),
           ("answer_rate", "correct 拆解"), ("correct_at_answered", "correct 拆解"),
           ("sufficient", "诊断"), ("cited_grounded", "诊断"),
           ("n_search", "成本"), ("n_chunks", "成本"), ("n_ctx_chars", "成本"), ("n_llm_calls", "成本")]


def paired(recs: list[dict], base: list[dict], meta_a: dict, meta_b: dict, a: str, b: str) -> None:
    fatal, varied = compare_meta(meta_a, meta_b)
    print(f"\n{'=' * 92}\n同题配对：{a} − {b}\n{'=' * 92}")
    if fatal:
        print("⛔ 两个 dump 的**尺子不同**，拒绝配对：\n  " + "\n  ".join(fatal))
        print("  （裁判/评测集不同 = 比的不是同一件事。用 eval_rescore.py 拿同一个裁判重打分再比。）")
        return
    print("  自变量（这次实验真正改了什么）：" + ("、".join(varied) if varied else
                                                 "⚠️ **没有任何差异** —— 两臂配置一模一样，差值只反映非确定性"))
    if not varied:
        print("     （若确实只想量重跑方差，忽略此条；否则先确认是不是拿错了 dump。）")

    by_id = {r["example_id"]: r for r in base if r["judge"]}
    pairs = [(r, by_id[r["example_id"]]) for r in recs if r["judge"] and r["example_id"] in by_id]
    print(f"\n  配对样本 n={len(pairs)}")
    if len(pairs) < 3:
        print("  <3 不出区间 —— 两个 dump 得抽同一批题（同 --per-type、同 --benchmark）。")
        return

    # ── 尺子中立性守卫：截断率不等 → `grounded` 的差值不可信 ────────────────────────
    # 交付更多字符的臂更容易被截断，而被截断的上下文会让它的 grounded **系统性偏低** ——
    # 这就是实验19① 那个坑的同型（`fact@k` 只查前 120 字符，于是小块系统性占便宜）。
    # 「先怀疑尺子，再怀疑被测对象」——所以这里宁可拒绝出这一格的数。
    ta = sum(1 for r, _ in pairs if r.get("ctx_truncated"))
    tb = sum(1 for _, r in pairs if r.get("ctx_truncated"))
    skip = set()
    if ta != tb:
        print(f"\n⛔ 两臂的上下文截断率不同（{ta}/{len(pairs)} vs {tb}/{len(pairs)}）——"
              f"交付更多的那臂被截得更狠，`grounded` 会**系统性偏低**。\n"
              f"   这一格的差值**不出**（先怀疑尺子）。修法：--ctx-chars 调大重判，或把两臂的 k 压到不触顶。")
        skip.add("grounded")

    print(f"\n{'指标':16s}{'类别':12s}{'差值':>10}{'  95% 置信区间':>22}")
    for k, tag in _PAIRED:
        if k in skip:
            print(f"{k:16s}{tag:12s}{'—':>10}   （截断率不等，拒绝出数）")
            continue
        vals = []
        for x, y in pairs:
            if k in ("correct", "grounded", "sufficient"):
                u, v = x["judge"].get(k), y["judge"].get(k)
            elif k == "answer_rate":
                u, v = float(answered(x)), float(answered(y))
            elif k == "correct_at_answered":
                # 只在**两边都给了答案**的题上比 —— 否则比的是"谁更敢答"，那是 answer_rate 的事。
                u, v = ((x["judge"].get("correct"), y["judge"].get("correct"))
                        if answered(x) and answered(y) else (None, None))
            else:
                u, v = x.get(k), y.get(k)
            if u is not None and v is not None and u == u and v == v:
                vals.append(u - v)
        if len(vals) < 3:
            continue
        m, lo, hi = _ci(vals)
        sig = "✅ 显著" if lo > 0 else ("⛔ 反向显著" if hi < 0 else "  跨 0，**不显著**")
        w = 10 if abs(m) < 100 else 10
        print(f"{k:16s}{tag:12s}{m:>+{w}.3f}   [{lo:+.3f}, {hi:+.3f}]  {sig}")
    print(f"\n  区间跨 0 = 这批样本量分辨不出差异，**不能说谁更好**（实验15 就栽在把 n=15 的噪声当效应）。")
    if len(pairs) < 60:
        print(f"  ⚠️ n={len(pairs)}：只够分辨 ~{1.96 * 0.35 / len(pairs) ** 0.5:.2f} 量级的效应。"
              f"要判 0.05 需 n≳200。")


def main() -> None:
    ap = argparse.ArgumentParser(description="裁判打 correct / grounded / sufficient 三分（各有各的参照系）")
    ap.add_argument("dump", help="eval_agentic.py 产出的 JSONL（需含 answer/cited/contexts）")
    ap.add_argument("--baseline", default=None, help="另一个 dump，按 example_id **同题配对**比较")
    ap.add_argument("--benchmark", choices=["musique", "multihoprag"], default=None,
                    help="gold 从哪来。缺省从 dump 的 __meta__ 里读（旧格式 dump 才需要手填）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--ctx-chars", type=int, default=150_000,
                    help="送裁判的检索上下文上限（去重后）。截断会让 grounded 偏低，逐题记录并告警")
    ap.add_argument("--out", default=None, help="把逐题判分写成 JSONL（便于复查裁判的 why / ungrounded_claims）")
    args = ap.parse_args()

    meta0, _ = read_dump(args.dump)
    bench = args.benchmark or meta0.get("benchmark") or "musique"
    gold, judge = _gold(bench), build_judge()
    meta_a, recs = score_dump(args.dump, gold, judge, args.concurrency, args.ctx_chars, args.out)
    table(recs, meta_a, f"头号指标 · {args.dump}")
    if not args.baseline:
        return
    meta_b, base = score_dump(args.baseline, gold, judge, args.concurrency, args.ctx_chars,
                              (args.out + ".base") if args.out else None)
    table(base, meta_b, f"头号指标 · {args.baseline}（基线）")
    paired(recs, base, meta_a, meta_b, args.dump, args.baseline)


if __name__ == "__main__":
    main()
