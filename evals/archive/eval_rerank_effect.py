"""reranker 到底是增益还是减益？——直接盯**含 gold 证据的 chunk 的排名**在重排前后怎么变。

之前只测过"重排后 fact@8"比"融合后 fact@8"高（实验11 ②，+0.088，全题型平均），
但那是**聚合数字**，掩盖了两件事：
  1. 分题型看，重排可能对某些题型是**减益**的（temporal 的问句与证据对不齐，见实验14）；
  2. 就算平均是增益，也可能是"把本来第 9 名的推到第 3 名"与"把本来第 2 名的推到第 20 名"相抵消的结果。

本脚本对每道题，找出候选池里**真含 gold 证据句**的 chunk，记录它在
  融合后（RRF 顺序，reranker 未介入）与 重排后 两个列表里的名次，直接看重排把它**推前还是推后**。

    python eval_rerank_effect.py --n 15 --types temporal,comparison,inference
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
import statistics as st

import numpy as np

from rag.corpus_multihop import load_corpus
from rag.retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
_KS = (4, 8, 16)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _rrf_merge(lists: list[list], k: int = 60) -> list:
    """把多个排序列表按 RRF 合并（只看名次）——用来把『重排名次』与『融合名次』折中。"""
    fused: dict[str, list] = {}
    for lst in lists:
        for rank, d in enumerate(lst, start=1):
            slot = fused.setdefault(d.id, [d, 0.0])
            slot[1] += 1.0 / (k + rank)
    return [d for d, _ in sorted(fused.values(), key=lambda v: v[1], reverse=True)]


def _interleave(lists: list[list]) -> list:
    """轮流从每个列表取下一个未见过的：把 top-k 名额在几个排序之间对半分。"""
    out, seen = [], set()
    for rank in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if rank < len(lst) and lst[rank].id not in seen:
                seen.add(lst[rank].id)
                out.append(lst[rank])
    return out


def _fact_recall(docs: list, facts: list[str], k: int) -> float:
    blob = _norm(" ".join(d.text for d in docs[:k]))
    return sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--types", default="temporal,comparison,inference")
    ap.add_argument("--reranker", default="bge", choices=["bge", "qwen"])
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    qs = []
    for t in types:
        picked = [r for r in rows if r.get("question_type", "").startswith(t)
                  and [e for e in (r.get("evidence_list") or []) if e.get("fact")]]
        for r in picked[: args.n]:
            qs.append((t, r["query"], [e["fact"] for e in r["evidence_list"] if e.get("fact")]))

    if args.reranker == "qwen":
        from rag.reranker_qwen import QwenReranker
        retr = HybridRetriever(load_corpus(), reranker=QwenReranker())
    else:
        retr = HybridRetriever(load_corpus())

    # per-type: 融合/重排 的 fact@k，以及证据 chunk 的名次变化
    agg: dict[str, dict] = {}
    for t, q, facts in qs:
        cands = retr._fuse(q)                                   # 融合顺序（reranker 未介入）
        sc = np.asarray(retr.reranker.predict([(q, d.text) for d in cands]))
        order = np.argsort(-sc)
        ranked = [cands[i] for i in order]                       # 重排后顺序

        # 单路：BM25 / Dense（各取 pool 个，与融合同量级才可比）
        bm25_only = [h.doc for h in retr.bm25.search(q, k=retr.pool)]
        dense_only = [h.doc for h in retr.dense.search(q, k=retr.pool)]

        # 混合策略：不让 reranker 独占 top-k 的决定权（免费、无需判断题型）
        mix_rrf = _rrf_merge([ranked, cands])          # 把「重排名次」与「融合名次」按 RRF 合并
        mix_alt = _interleave([ranked, cands])         # 交替取：重排第1、融合第1、重排第2…

        lists = {"bm25": bm25_only, "dense": dense_only, "fuse": cands,
                 "rr": ranked, "mix_rrf": mix_rrf, "mix_alt": mix_alt}
        slot = agg.setdefault(t, {name: {k: [] for k in (*_KS, "pool")} for name in lists}
                              | {"delta": [], "pushed_back": 0, "pushed_fwd": 0, "n_ev": 0,
                                 "fuse_rank": [], "rr_rank": []})
        for name, lst in lists.items():
            for k in _KS:
                slot[name][k].append(_fact_recall(lst, facts, k))
            slot[name]["pool"].append(_fact_recall(lst, facts, len(lst)))

        # 找出"真含证据句"的 chunk，看它在两个列表里的名次
        fuse_pos = {d.id: i for i, d in enumerate(cands)}
        rr_pos = {d.id: i for i, d in enumerate(ranked)}
        for d in cands:
            txt = _norm(d.text)
            if any(_norm(f)[:120] in txt for f in facts):
                a, b = fuse_pos[d.id], rr_pos[d.id]
                slot["fuse_rank"].append(a)
                slot["rr_rank"].append(b)
                slot["delta"].append(b - a)                      # >0 = 被推后
                slot["n_ev"] += 1
                if b > a:
                    slot["pushed_back"] += 1
                elif b < a:
                    slot["pushed_fwd"] += 1

    print(f"\n== reranker={args.reranker}｜fact 级召回：融合后(未重排) vs 重排后 ==")
    print(f"{'type':12s} " + " ".join(f"{'融合@'+str(k):>9}{'重排@'+str(k):>9}{'Δ':>7}" for k in _KS))
    for t in types:
        d = agg.get(t)
        if not d:
            continue
        cells = []
        for k in _KS:
            f, r = st.mean(d["fuse"][k]), st.mean(d["rr"][k])
            cells.append(f"{f:>9.3f}{r:>9.3f}{r - f:>+7.3f}")
        print(f"{t:12s} " + " ".join(cells))

    print(f"\n== 检索栈分层 · fact 级召回 · 按题型（reranker={args.reranker}）==")
    names = {"bm25": "① BM25 单路", "dense": "② Dense 单路", "fuse": "③ 融合(RRF)",
             "rr": "④ 融合+重排", "mix_rrf": "⑤ 混合RRF(重排,融合)", "mix_alt": "⑥ 混合交替取"}
    print(f"{'type':12s} {'策略':22s} " + " ".join(f"{'@'+str(k):>8}" for k in _KS) + f"{'覆盖':>9}")
    for t in types:
        d = agg.get(t)
        if not d:
            continue
        for key, label in names.items():
            print(f"{t:12s} {label:22s} " + " ".join(f"{st.mean(d[key][k]):>8.3f}" for k in _KS)
                  + f"{st.mean(d[key]['pool']):>9.3f}")
        print()

    print(f"\n== 含 gold 证据的 chunk：重排把它推前还是推后？（名次 0 起，越小越靠前）==")
    print(f"{'type':12s} {'证据chunk数':>10} {'融合中位名次':>12} {'重排中位名次':>12} {'被推后':>8} {'被推前':>8}")
    for t in types:
        d = agg.get(t)
        if not d or not d["n_ev"]:
            continue
        print(f"{t:12s} {d['n_ev']:>10} {st.median(d['fuse_rank']):>12.1f} {st.median(d['rr_rank']):>12.1f} "
              f"{d['pushed_back'] / d['n_ev']:>7.0%} {d['pushed_fwd'] / d['n_ev']:>7.0%}")


if __name__ == "__main__":
    main()
