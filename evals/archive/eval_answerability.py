"""检索侧的**可答性**指标：只把检索到的上下文交给裁判，让它据此答题，再与 gold answer 比对。

为什么要换掉 `fact@k` / `context_recall_fact`（用户提出，实测支持）：

1. **它做的是对「某一条特定 gold 句子」的子串匹配**。可 gold 句只是数据集作者挑的某一句——
   同一篇文章里另一句话完全可能同样支撑答案，而这个指标会判 0 分。**过严。**
2. **冗余证据被同等计分**。实测 inference 类平均只有 **1.47 / 3.36** 条证据真正含答案，
   其余是氛围性铺垫；漏掉它们照样能答对，指标却照扣分。**权重错配。**
   （注：comparison / temporal 是"A 和 B 是不是都……"的**合取**问题，两条缺一不可，不算冗余。）
3. **它没有成本项**：搜 10 次 × 32 片段必然赢过搜 1 次 × 32，哪怕后 9 次全是重复。

本脚本改成直接量**「交付的上下文够不够答出这道题」**，并且判分环节保持**确定性**
（比对标准答案，而不是让裁判打主观分）——裁判只负责"读上下文、给答案"这一步。

**三条线必须一起量，否则数字没有意义**（这是本项目栽过多次的那类坑）：

    floor    不给任何上下文  → 瞎猜 + 参数化知识的下限
    ctx      只给检索到的上下文 → 实测
    ceiling  只给 gold 证据句   → 上限；裁判拿着标准证据都答不对的题，任何检索都救不了

为什么下限非量不可：本数据集 **comparison 96% 是 Yes/No 且 60% 答 "Yes"**、temporal 89% 是
Yes/No——**恒答 "Yes" 就有 0.60 / 0.49**；inference 的答案又多是 Google / Sam Altman 这种
**强模型不检索也知道**的实体。不减掉地板，0.85 里有多少是猜的、有多少是背下来的，完全不知道。

**裁判为什么要冻住**：让裁判"读上下文、给答案"，机制上确实和 RAG 的生成步一样——**但角色不同**。
若直接用 agent 自己的答案对错来衡量检索，换一个 agent 就**同时换掉了被测对象和尺子**
（实验13 踩过的坑），"它检索得更好"和"它本来就更会答"分不开。
**固定裁判之后，correctness 才变成一个检索指标。** 所以本脚本的裁判只负责读与答，判分是确定性的。

上下文来源有两种，对应两类实验：

    --source retrieval   单轮检索（原问句 k=32）→ 量**检索栈本身**，替换实验19-23 的 `fact@k`
    --source agentic     **agent 多跳检索的并集**（从 `eval_agentic.py --local` 的 dump 读）
                         → 量 **agentic 系统的检索**，替换实验21/24 的 `context_recall_fact`

    python eval_answerability.py --n 30 --types comparison,inference,temporal --model deepseek-v4-pro
    python eval_answerability.py --source agentic --from-dump ab_sol.jsonl --model deepseek-v4-pro
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))

import argparse
import json
import pathlib
import re
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rag.corpus_multihop import load_corpus
from rag.llm import build_model
from rag.retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

_ASK = (
    "Answer the question using ONLY the context below. Be concise: reply with just the answer "
    "(a name, an entity, or Yes/No). If the context does not contain enough information to answer, "
    "reply with exactly INSUFFICIENT.\n\n"
    "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
)
_ASK_NOCTX = (
    "Answer the question from your own knowledge. Be concise: reply with just the answer "
    "(a name, an entity, or Yes/No). If you do not know, reply with exactly INSUFFICIENT.\n\n"
    "Question: {q}\nAnswer:"
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _graded(pred: str, gold: str) -> float | None:
    """确定性判分。返回 None = 裁判自称信息不足（单独统计，**不当 0 分**）。"""
    p, g = _norm(pred), _norm(gold)
    if "insufficient" in p:
        return None
    if g in ("yes", "no", "true", "false", "consistent", "inconsistent"):
        first = next((w for w in p.split() if w in
                      ("yes", "no", "true", "false", "consistent", "inconsistent")), "")
        same = {"yes": "yes", "true": "yes", "consistent": "yes",
                "no": "no", "false": "no", "inconsistent": "no"}
        return float(same.get(first, "?") == same.get(g, "!"))
    return float(bool(g) and g in p)          # 开放答案：标准答案是否出现在裁判回答里


def _ci(d: list[float], boots: int = 20000) -> tuple[float, float, float]:
    a = np.asarray(d, float)
    if len(a) < 3:
        return (float(a.mean()) if len(a) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, len(a), size=(boots, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="每题型抽几题")
    ap.add_argument("--types", default="comparison,inference,temporal")
    ap.add_argument("--k", type=int, default=32, help="检索交付几个片段")
    ap.add_argument("--model", default=None, help="**裁判**模型（只负责读上下文给答案，判分是确定性的）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--source", default="retrieval", choices=["retrieval", "agentic"],
                    help="上下文从哪来：retrieval=现场单轮检索（量检索栈）；"
                         "agentic=从 eval_agentic.py --local 的 dump 里读 agent 多跳的并集（量 agentic 系统）")
    ap.add_argument("--from-dump", default=None, help="--source agentic 时的 JSONL（需含 contexts 字段）")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))

    if args.source == "agentic":
        if not args.from_dump:
            raise SystemExit("--source agentic 需要 --from-dump（先跑 eval_agentic.py --local）")
        by_q = {r["query"]: r for r in rows}
        dump = [json.loads(l) for l in
                pathlib.Path(args.from_dump).read_text(encoding="utf-8").splitlines() if l.strip()]
        qs, ctxs = [], []
        for d in dump:
            src = by_q.get(d.get("question"))
            if not src or "errored" in d or "contexts" not in d:
                continue
            facts = [e["fact"] for e in src.get("evidence_list", []) if e.get("fact")]
            if not (src.get("answer") and facts):
                continue
            qs.append((d["type"], d["question"], src["answer"], facts))
            ctxs.append("\n\n".join(d["contexts"]))          # agent 多跳检索的并集
        if not qs:
            raise SystemExit(f"{args.from_dump} 里没有可用记录（需要 contexts 字段；"
                             f"旧版 dump 没存，得用新版 eval_agentic.py --local 重跑）")
        print(f"{len(qs)} 题 · 上下文来自 agentic dump {args.from_dump}"
              f" · 裁判={args.model or 'RAG_MODEL'}", flush=True)
    else:
        qs = []
        for t in types:
            picked = [r for r in rows if r.get("question_type", "").startswith(t)
                      and r.get("answer") and [e for e in (r.get("evidence_list") or []) if e.get("fact")]]
            for r in picked[: args.n]:
                qs.append((t, r["query"], r["answer"],
                           [e["fact"] for e in r["evidence_list"] if e.get("fact")]))
        retr = HybridRetriever(load_corpus())
        print(f"{len(qs)} 题 · 单轮检索 k={args.k} · 裁判={args.model or 'RAG_MODEL'}"
              f"\n检索中（确定性、无 LLM）…", flush=True)
        ctxs = [" \n\n".join(h.doc.text for h in retr.search(q, k=args.k)) for _, q, _, _ in qs]

    types = sorted({t for t, *_ in qs}, key=lambda x: (types + [x]).index(x))
    llm = build_model(args.model)

    def _ask(prompt: str) -> str:
        for wait in (0, 5, 20):
            if wait:
                time.sleep(wait)
            try:
                return llm.invoke(prompt).content or ""
            except Exception:      # noqa: BLE001 —— 网关抽风
                pass
        return "__GATEWAY_FAILED__"

    jobs = []
    for i, ((t, q, a, facts), ctx) in enumerate(zip(qs, ctxs)):
        jobs += [(i, "floor", _ASK_NOCTX.format(q=q)),
                 (i, "ctx", _ASK.format(ctx=ctx, q=q)),
                 (i, "ceiling", _ASK.format(ctx="\n\n".join(facts), q=q))]
    print(f"裁判 {len(jobs)} 次调用（每题 3 条线：floor / ctx / ceiling）…", flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        outs = list(pool.map(lambda j: _ask(j[2]), jobs))

    res: dict = {}
    for (i, cond, _), out in zip(jobs, outs):
        res.setdefault(i, {})[cond] = out
    recs = []
    for i, (t, q, a, facts) in enumerate(qs):
        r = {"type": t, "question": q, "answer": a}
        for cond in ("floor", "ctx", "ceiling"):
            o = res[i][cond]
            r[cond] = None if o == "__GATEWAY_FAILED__" else _graded(o, a)
            r[cond + "_raw"] = o[:200]
        recs.append(r)
    if args.dump:
        pathlib.Path(args.dump).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")

    print(f"\n{'=' * 92}\n可答性：只凭这段上下文能否答对（判分确定性；INSUFFICIENT 单列，不当 0 分）\n{'=' * 92}")
    print(f"{'type':12s}{'n':>4}{'地板(无上下文)':>16}{'实测(检索)':>13}{'天花板(gold证据)':>17}"
          f"{'  归一化位置':>14}{'  实测-地板 [95%CI]':>26}")
    for t in [*types, "全部"] if len(types) > 1 else types:
        sub = [r for r in recs if t == "全部" or r["type"] == t]
        val = lambda c: [r[c] for r in sub if r[c] is not None]      # noqa: E731
        f_, c_, g_ = st.mean(val("floor")), st.mean(val("ctx")), st.mean(val("ceiling"))
        pair = [(r["ctx"], r["floor"]) for r in sub if r["ctx"] is not None and r["floor"] is not None]
        m, lo, hi = _ci([x - y for x, y in pair])
        sig = "✅" if lo > 0 else ("⛔" if hi < 0 else "  ")
        pos = (c_ - f_) / (g_ - f_) if g_ > f_ else float("nan")
        print(f"{t:12s}{len(sub):>4}{f_:>16.3f}{c_:>13.3f}{g_:>17.3f}{pos:>14.0%}"
              f"{m:>+16.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
    print("\n  归一化位置 = (实测−地板)/(天花板−地板)：100% = 检索已交付到 gold 证据同等的可答性；"
          "\n  ≤0% = 检索没带来任何超出「猜测+参数化知识」的信息。**绝对值不减地板毫无意义**。")

    ins = {c: sum(1 for r in recs if r[c] is None) for c in ("floor", "ctx", "ceiling")}
    print(f"\n  裁判自称信息不足/网关失败：floor {ins['floor']} · ctx {ins['ctx']} · ceiling {ins['ceiling']}（共 {len(recs)} 题）")
    bad = [r for r in recs if r["ceiling"] == 0.0]
    if bad:
        print(f"  ⚠️ {len(bad)}/{len(recs)} 题**拿着 gold 证据也答错** —— 这些题的上限就不是 1.0，"
              f"任何检索改进都到不了；把它们算进分母会系统性低估检索。")


if __name__ == "__main__":
    main()
