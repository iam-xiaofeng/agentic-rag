"""全链路排查：召回率低，到底断在**哪一层**？（确定性、免费、不调 LLM）

把链路拆成四层，每层单独问"这一层的上限是多少"，第一个塌下来的就是元凶：

  L0 证据句在不在语料里     对每条 gold fact，在**全部 chunk** 里做子串搜索。
                            找不到 → 切分/文本层问题，下游再怎么优化都是徒劳。
  L1 embedding 自检         **用证据句本身当 query** 去检索，看含它的 chunk 排第几。
                            这是给 embedding 的**最有利输入**（query 就是答案原文）。
                            连这个都排不进前几 → 编码或索引坏了，与"语义鸿沟"无关。
  L2 BM25 自检              同上，换 BM25。用来判断是"向量的问题"还是"两路都的问题"。
  L3 真实问题检索           换回**真实多跳问句**，看同一个 chunk 排第几。
                            L1 好而 L3 差 = 纯粹的 query↔证据 语义鸿沟（实验14-16 的结论）。

L1 与 L3 的落差，就是"问句写得不对"要背的锅；L0/L1 若塌，才是数据或模型的锅。

    python eval_chain.py --n 15 --types temporal,comparison,inference
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics as st

import numpy as np

from corpus_multihop import load_corpus
from retriever_bm25 import BM25Retriever
from retriever_dense import DenseRetriever

DATA = pathlib.Path(__file__).resolve().parent / "data"
_PROBE = 200          # 排名探测深度；超出记为 >200（用 _MISS 表示）
_MISS = 10**6


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _rank_of(hits, target_ids: set[str]) -> int:
    for i, h in enumerate(hits):
        if h.doc.id in target_ids:
            return i
    return _MISS


def _fmt(ranks: list[int]) -> str:
    """把一组排名汇总成 中位数 / top1 / top10 / 未命中率。"""
    if not ranks:
        return f"{'-':>8}{'-':>8}{'-':>8}{'-':>8}"
    hit = [r for r in ranks if r < _MISS]
    med = f"{st.median(hit):.0f}" if hit else "—"
    top1 = sum(1 for r in ranks if r == 0) / len(ranks)
    top10 = sum(1 for r in ranks if r < 10) / len(ranks)
    miss = sum(1 for r in ranks if r >= _MISS) / len(ranks)
    return f"{med:>8}{top1:>8.0%}{top10:>8.0%}{miss:>8.0%}"


# 等信息量：chunk 越小，同样的 k 交付的**字符数**越少。实验11 拿固定 fact@8 比不同 chunk，
# 等于让 300 的块只交付 2400 字符、1200 的块交付 9600 字符——**信息量差 4 倍**，小块必输。
# 这里按「总字符数」对齐：(chunk, k) 组合使 chunk*k ∈ {4800, 9600}。
_CHUNK_KS = [(1200, 8), (1200, 16), (600, 16), (600, 32)]


def _chunk_sweep(types: list[str], n: int) -> None:
    import corpus_multihop as cm
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from retriever_hybrid import HybridRetriever

    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    qs = []
    for t in types:
        got = [r for r in rows if r.get("question_type", "").startswith(t)
               and [e for e in (r.get("evidence_list") or []) if e.get("fact")]][:n]
        qs += [(t, r["query"], [e["fact"] for e in r["evidence_list"] if e.get("fact")]) for r in got]

    def fr(facts, docs, k):
        blob = _norm(" ".join(d.text for d in docs[:k]))
        return sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts)

    res: dict = {}
    for size in sorted({c for c, _ in _CHUNK_KS}, reverse=True):
        cm._SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=size // 8,
            separators=["\n\n", "\n", ". ", " ", ""], length_function=len)
        docs = cm.load_corpus()
        retr = HybridRetriever(docs)
        print(f"[chunk={size}] 片段 {len(docs)}，检索中…", flush=True)
        for t, q, facts in qs:
            cands = retr._fuse(q)
            sc = np.asarray(retr.reranker.predict([(q, d.text) for d in cands]))
            ranked = [cands[i] for i in np.argsort(-sc)]
            for c, k in _CHUNK_KS:
                if c != size:
                    continue
                slot = res.setdefault((t, c, k), {"fuse": [], "rr": []})
                slot["fuse"].append(fr(facts, cands, k))
                slot["rr"].append(fr(facts, ranked, k))

    print(f"\n== chunk 大小 × 交付深度（**等信息量**对齐）· fact 级召回 ==")
    print(f"{'type':12s} {'chunk':>6}{'k':>4}{'总字符':>8}{'融合后':>9}{'重排后':>9}")
    for t in types:
        for c, k in _CHUNK_KS:
            d = res.get((t, c, k))
            if d:
                print(f"{t:12s} {c:>6}{k:>4}{c * k:>8}{st.mean(d['fuse']):>9.3f}{st.mean(d['rr']):>9.3f}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--types", default="temporal,comparison,inference")
    ap.add_argument("--chunk-sweep", action="store_true",
                    help="重做 chunk 大小对比，但按**等信息量**对齐（见 _chunk_sweep 注释）")
    args = ap.parse_args()

    if args.chunk_sweep:
        _chunk_sweep([t.strip() for t in args.types.split(",") if t.strip()], args.n)
        return

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    docs = load_corpus()
    blobs = [(d.id, _norm(d.text)) for d in docs]
    bm25, dense = BM25Retriever(docs), DenseRetriever(docs)
    print(f"语料 {len(docs)} 个 chunk；排名探测深度 {_PROBE}\n")

    stats: dict[str, dict] = {}
    for t in types:
        picked = [r for r in rows if r.get("question_type", "").startswith(t)
                  and [e for e in (r.get("evidence_list") or []) if e.get("fact")]][: args.n]
        s = stats.setdefault(t, {"n_fact": 0, "in_corpus": 0,
                                 "self_dense": [], "self_bm25": [], "q_dense": [], "q_bm25": []})
        for r in picked:
            q = r["query"]
            for e in r["evidence_list"]:
                fact = e.get("fact")
                if not fact:
                    continue
                s["n_fact"] += 1
                key = _norm(fact)[:120]
                owners = {did for did, b in blobs if key in b}      # L0：哪些 chunk 真含这条证据
                if not owners:
                    continue
                s["in_corpus"] += 1
                # L1/L2：用证据句自己当 query（对检索器最有利的输入）
                s["self_dense"].append(_rank_of(dense.search(fact, k=_PROBE), owners))
                s["self_bm25"].append(_rank_of(bm25.search(fact, k=_PROBE), owners))
            # L3：用真实问句查（每题只算一次，取所有 owner chunk 的最好名次）
            all_owners = set()
            for e in r["evidence_list"]:
                if e.get("fact"):
                    all_owners |= {did for did, b in blobs if _norm(e["fact"])[:120] in b}
            if all_owners:
                s["q_dense"].append(_rank_of(dense.search(q, k=_PROBE), all_owners))
                s["q_bm25"].append(_rank_of(bm25.search(q, k=_PROBE), all_owners))

    print("L0 证据句在不在语料里（切分/文本层的上限）")
    print(f"{'type':12s} {'证据句数':>8} {'在语料里':>9}")
    for t in types:
        s = stats[t]
        print(f"{t:12s} {s['n_fact']:>8} {s['in_corpus'] / max(s['n_fact'], 1):>9.0%}")

    hdr = f"{'中位名次':>8}{'top1':>8}{'top10':>8}{'>200':>8}"
    print(f"\nL1/L2 **用证据句自己当 query**（检索器的能力上限；理想 top1≈100%）\n"
          f"{'type':12s} {'检索器':>8} {hdr}")
    for t in types:
        for label, key in (("dense", "self_dense"), ("BM25", "self_bm25")):
            print(f"{t:12s} {label:>8} " + _fmt(stats[t][key]))

    print(f"\nL3 **用真实多跳问句**查同一批 chunk（与 L1 的落差 = 问句↔证据的语义鸿沟）\n"
          f"{'type':12s} {'检索器':>8} {hdr}")
    for t in types:
        for label, key in (("dense", "q_dense"), ("BM25", "q_bm25")):
            print(f"{t:12s} {label:>8} " + _fmt(stats[t][key]))


if __name__ == "__main__":
    main()
