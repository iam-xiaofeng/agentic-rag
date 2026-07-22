"""P3 评测：在 MultiHop-RAG（真实语料，BM25 后端）上对比 agentic vs 单次检索。

    .venv/bin/python eval_multihop.py            # 默认 12 题
    .venv/bin/python eval_multihop.py --n 20     # 更大样本（更费 token）

需要模型凭据（OPENAI_BASE_URL / OPENAI_API_KEY / RAG_MODEL；若网关 WAF 拦 UA 再加 RAG_USER_AGENT）。

与玩具库的区别：这里 gold 证据分散在约 4k 个片段中的 2~4 篇，单次 BM25 查询在结构上
必然「欠覆盖」证据链，所以我们预期 agentic 的多轮改写能提升 **检索覆盖率(coverage)**，
进而提升 **正确率(correct)**。两项都报，并给出 agentic - 单次 的 delta。

说明：`correct` 是「参考答案子串」的近似判定，且部分 MultiHop 答案是 Yes/No 或日期，
所以正确率的绝对值有噪声 —— 真正的信号是**同一套打分下 agentic 相对单次的 delta**，而非绝对数。

鲁棒性：单题若因网关 5xx/超时失败，只跳过该题并继续（不让一次抖动毁掉整轮）。
"""

from __future__ import annotations

import argparse

from agent import build_agent
from corpus_multihop import load_corpus, load_examples
from eval_baseline import single_shot
from eval_dataset import Example
from eval_metrics import METRICS
from eval_run import run_agentic
from retriever_bm25 import BM25Retriever


def _pick(lst: list, m: int) -> list:
    """确定性等距抽样：把 m 个样本均匀铺在整个列表上（不受文件内分组影响）。"""
    if m >= len(lst):
        return list(lst)
    step = len(lst) / m
    return [lst[int(i * step)] for i in range(m)]


def _sample(n: int) -> list[Example]:
    exs = load_examples()
    answerable = [e for e in exs if e.kind == "multihop"]
    negatives = [e for e in exs if e.kind == "negative"]
    n_neg = min(len(negatives), max(2, n // 5))
    return _pick(answerable, n - n_neg) + _pick(negatives, n_neg)


def _coverage(ex: Example, out: dict) -> float | None:
    """检索覆盖率：整轮检索中 gold 来源文档被命中的比例（仅对 multihop 有意义）。"""
    if not ex.sources:
        return None
    got = set(out.get("sources", []))
    return len(set(ex.sources) & got) / len(set(ex.sources))


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _score(ex: Example, out: dict) -> dict:
    row = {k: f(ex, out) for k, f in METRICS.items()}
    row["n_search"] = out["n_search"]
    row["coverage"] = _coverage(ex, out)
    return row


def _summary(label: str, rows: list[dict]) -> dict:
    print(f"\n== {label} ==")
    agg = {}
    for k in METRICS:
        agg[k] = _avg([r[k] for r in rows])
        print(f"  {k:11s}: {agg[k]:.2f}")
    agg["coverage"] = _avg([r["coverage"] for r in rows if r["coverage"] is not None])
    agg["n_search"] = _avg([r["n_search"] for r in rows])
    print(f"  coverage   : {agg['coverage']:.2f}   (gold 文档被检索到的比例，仅 multihop)")
    print(f"  avg_search : {agg['n_search']:.2f}")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="评测题数")
    args = ap.parse_args()

    print("在 MultiHop-RAG 语料上构建 BM25 索引 ...")
    retriever = BM25Retriever(load_corpus())
    print(f"  已索引 {len(retriever.docs)} 个片段")
    data = _sample(args.n)
    agent = build_agent(retriever)

    ag_rows, bl_rows = [], []
    ok, failed = 0, 0
    print(f"\n{'kind':10s} {'n':>2s}  {'A.cov':>5s} {'S.cov':>5s}  {'A.ok':>4s} {'S.ok':>4s}  question")
    for ex in data:
        try:
            ao = run_agentic(ex.question, agent)
            bo = single_shot(ex.question, retriever)
        except Exception as e:  # 网关 5xx/超时等：跳过该题，保住整轮
            failed += 1
            print(f"{ex.kind:10s}  [skip] {type(e).__name__}: {str(e)[:70]}")
            continue
        ok += 1
        ar, br = _score(ex, ao), _score(ex, bo)
        ag_rows.append(ar)
        bl_rows.append(br)
        cov_a = f"{ar['coverage']:.2f}" if ar["coverage"] is not None else "  - "
        cov_b = f"{br['coverage']:.2f}" if br["coverage"] is not None else "  - "
        print(f"{ex.kind:10s} {ao['n_search']:>2d}  {cov_a:>5s} {cov_b:>5s}  "
              f"{ar['correct']:>4.0f} {br['correct']:>4.0f}  {ex.question[:56]}")

    print(f"\n[完成 {ok} 题 / 跳过 {failed} 题]")
    if not ag_rows:
        print("没有成功的样本，无法汇总（多半是网关持续超时，稍后重试或换模型）。")
        return

    a = _summary("AGENTIC（BM25，多跳）", ag_rows)
    b = _summary("SINGLE-SHOT（BM25，单次查询）", bl_rows)
    print("\n== DELTA (agentic - single) ==")
    for k in list(METRICS) + ["coverage"]:
        print(f"  {k:11s}: {a[k] - b[k]:+.2f}")


if __name__ == "__main__":
    main()
