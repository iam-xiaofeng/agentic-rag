"""检索融合/参数消融（**免费**，不调 LLM）：贪心坐标上升调 fusion → rrf_k → w_dense → pool。

用**确定性 fact 级** context_recall（gold 证据句是否真在检索文本里；比标题级准，见 EXPERIMENTS 实验9）衡量，每行报告：
  recall@pool  融合后候选池的召回（重排前的天花板，纯看「融合有没有把 gold 捞进池」）
  recall@4/@8  重排取 top-k 后真正交给模型的召回

复用同一个 HybridRetriever（语料只编码一次，走 .cache），逐配置翻 fusion/w/pool/rrf_k 属性，很快。

    python eval_ablation.py            # 默认 n=50 多跳，跑全套 sweep 并自动选优
    python eval_ablation.py --n 80
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))

import argparse
import json

import numpy as np

from rag.corpus_multihop import DATA, load_corpus
from evals.archive.eval_common import _norm
from rag.retriever_hybrid import HybridRetriever

KS = (4, 8)  # 报告的 top-k（交给模型的深度）


def _fact_examples(n: int, qtype: str | None = None) -> list[tuple[str, list[str]]]:
    """(问题, gold 证据句列表)；证据句来自 MultiHopRAG 的 evidence_list[].fact。
    `qtype` 给定则只取该题型——**参数的最优值可能因题型而异**，全类型聚合会把它抹平
    （实验15-16 的教训：聚合掩盖子群效应）。"""
    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    out: list[tuple[str, list[str]]] = []
    for r in rows:
        t = r.get("question_type", "")
        if t == "null_query" or (qtype and not t.startswith(qtype)):
            continue
        facts = [e["fact"] for e in (r.get("evidence_list") or []) if e.get("fact")]
        if facts:
            out.append((r["query"], facts))
    return out[:n]


def _fr(facts: list[str], docs) -> float:
    """fact 级召回：gold 证据句里有多少真出现在这批 chunk 文本里。"""
    blob = _norm(" ".join(d.text for d in docs))
    return sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts) if facts else 0.0


def _set(retr: HybridRetriever, fusion: str, w_dense: float, pool: int, rrf_k: int) -> None:
    retr.fusion, retr.w_dense, retr.w_bm25 = fusion, w_dense, round(1 - w_dense, 3)
    retr.pool, retr._rrf_k = pool, rrf_k


def _measure(retr: HybridRetriever, exps) -> dict:
    """对一组题跑当前配置，返回 {'pool':.., 4:.., 8:..} 的平均 fact 级 recall。"""
    tot = {"pool": 0.0, 4: 0.0, 8: 0.0}
    for q, facts in exps:
        cands = retr._fuse(q)                                 # 候选池（重排前）
        tot["pool"] += _fr(facts, cands)
        if cands:
            sc = np.asarray(retr.reranker.predict([(q, d.text) for d in cands]))
            order = np.argsort(-sc)
            for k in KS:
                tot[k] += _fr(facts, [cands[i] for i in order[:k]])
    n = len(exps)
    return {kk: v / n for kk, v in tot.items()}


def _row(label: str, m: dict) -> None:
    print(f"{label:22s}  recall@pool={m['pool']:.3f}   @4={m[4]:.3f}   @8={m[8]:.3f}")


_WS = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)


def _w_by_type(n: int) -> None:
    """按题型扫 w_dense。w=0 即纯 BM25、w=1 即纯 dense。
    实验16 发现 dense 在 temporal 上是明显弱腿（fact@8 0.344 vs BM25 0.522），
    且融合会让它把 BM25 的好候选挤出 pool（覆盖 0.889→0.867）——若如此，
    **降低 w_dense 应当对 temporal 有正收益**，而实验6 的『不敏感』只是聚合的错觉。"""
    retr = HybridRetriever(load_corpus())
    print(f"{'type':12s} {'w_dense':>8} " + " ".join(f"{'@'+str(k):>8}" for k in KS) + f"{'覆盖@pool':>10}")
    for t in ("temporal", "comparison", "inference"):
        exps = _fact_examples(n, t)
        for w in _WS:
            _set(retr, "rrf", w, 100, 60)
            m = _measure(retr, exps)
            print(f"{t:12s} {w:>8.1f} " + " ".join(f"{m[k]:>8.3f}" for k in KS) + f"{m['pool']:>10.3f}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="抽几道多跳题")
    ap.add_argument("--w-by-type", action="store_true",
                    help="只扫 w_dense、**按题型分开报**：验证『权重不敏感』是不是全类型聚合造成的错觉")
    args = ap.parse_args()

    if args.w_by_type:
        _w_by_type(args.n)
        return

    exps = _fact_examples(args.n)
    print(f"样本 {len(exps)} 道多跳；指标=fact 级 context_recall（gold 证据句在检索文本里；确定性、免费）\n")

    print("构建 hybrid 索引（编码语料走 .cache；只建一次，逐配置翻属性）...")
    retr = HybridRetriever(load_corpus())

    # Stage A · 融合方法（w_dense=0.5, pool=20, rrf_k=60）
    print("\n== Stage A · 融合方法（w_dense=0.5, pool=20, rrf_k=60）==")
    scoreA = {}
    for fusion in ("minmax", "rrf"):
        _set(retr, fusion, 0.5, 20, 60)
        m = _measure(retr, exps)
        scoreA[fusion] = m[8]
        _row(f"fusion={fusion}", m)
    fusion = "rrf" if scoreA["rrf"] >= scoreA["minmax"] else "minmax"
    print(f"→ 选 fusion={fusion}（recall@8 rrf={scoreA['rrf']:.3f} vs minmax={scoreA['minmax']:.3f}）")

    # Stage B · rrf_k（仅 rrf；pool=50, w_dense=0.5）
    rrf_k = 60
    if fusion == "rrf":
        print("\n== Stage B · rrf_k（pool=50, w_dense=0.5）==")
        scoreB = {}
        for k in (10, 30, 60, 100):
            _set(retr, "rrf", 0.5, 50, k)
            m = _measure(retr, exps)
            scoreB[k] = m[8]
            _row(f"rrf_k={k}", m)
        rrf_k = max(scoreB, key=lambda k: (round(scoreB[k], 4), -k))
        print(f"→ 选 rrf_k={rrf_k}")

    # Stage C · w_dense（fusion 定, pool=50, rrf_k 定）
    print(f"\n== Stage C · w_dense（fusion={fusion}, pool=50, rrf_k={rrf_k}）==")
    scoreC = {}
    for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        _set(retr, fusion, w, 50, rrf_k)
        m = _measure(retr, exps)
        scoreC[round(w, 1)] = m[8]
        _row(f"w_dense={w:.1f}", m)
    w_dense = max(scoreC, key=scoreC.get)
    print(f"→ 选 w_dense={w_dense}（w_bm25={round(1 - w_dense, 1)}）")

    # Stage D · pool（其余定；找召回停止上涨的拐点 = 瓶颈）
    print(f"\n== Stage D · pool（fusion={fusion}, w_dense={w_dense}, rrf_k={rrf_k}）==")
    scoreD = {}
    for pool in (20, 50, 100, 150, 200):
        _set(retr, fusion, w_dense, pool, rrf_k)
        scoreD[pool] = _measure(retr, exps)
        _row(f"pool={pool}", scoreD[pool])
    best_pool = max(scoreD, key=lambda p: (round(scoreD[p][8], 4), -p))  # 并列取小 pool
    print(f"→ 选 pool={best_pool}（recall@8 并列时取更小 pool）")

    m = scoreD[best_pool]
    print("\n== 最终选定 ==")
    print(f"fusion={fusion}  w_dense={w_dense}  w_bm25={round(1 - w_dense, 1)}  pool={best_pool}  rrf_k={rrf_k}")
    _row("final", m)


if __name__ == "__main__":
    main()
