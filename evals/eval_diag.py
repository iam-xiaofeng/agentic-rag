"""诊断召回：title 级 vs fact 级（**免费**、确定性、不调 LLM）——揭穿漏水的代理指标。

`context_recall` 是**文章标题级**（gold 文章有没有被检索到）。但「检索到对的文章」≠「交付了对的证据句」。
本脚本对同一批多跳题、在几种检索配置下同时测：
  title_recall  gold 文章标题 ∩ 检索到的 source ÷ gold 标题数            （现用的代理指标）
  fact_recall   gold 证据句(evidence_list[].fact)真出现在检索文本里的比例（更贴近 correctness）
  pool(100)     天花板：证据到底在不在候选池

关键发现（EXPERIMENTS 实验8-9）：source 去重能刷高 title_recall、却拉低 fact_recall（Goodhart）——
所以 `HybridRetriever` 默认不去重（`max_per_source=None`），真正管用的是 k=4→8（多交付证据）。

    python eval_diag.py --n 50
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

import numpy as np

from rag.corpus_multihop import load_corpus
from rag.retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _dedup(ranked, cap: int, k: int):
    """沿重排后的列表走，每篇文章最多取 cap 个，凑够 k 个。"""
    out, c = [], {}
    for d in ranked:
        if c.get(d.source, 0) >= cap:
            continue
        c[d.source] = c.get(d.source, 0) + 1
        out.append(d)
        if len(out) >= k:
            break
    return out


# 每个配置 = 从同一次重排的候选列表里怎么选（同一次重排、多种选法，省算力）
_CONFIGS = {
    "nodedup k=4": lambda r: r[:4],
    "nodedup k=8": lambda r: r[:8],
    "dedup1  k=8": lambda r: _dedup(r, 1, 8),
    "dedup2  k=8": lambda r: _dedup(r, 2, 8),
    "pool(100)":   lambda r: r,          # 天花板
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--reranker", default="bge", choices=["bge", "qwen"],
                    help="重排器：bge(默认) 或 qwen(Qwen3-Reranker-4B, instruction-aware)")
    ap.add_argument("--by-type", action="store_true",
                    help="按 question_type 分组报 fact 级召回随 k 的曲线（定位『哪类题的证据交付不上去』）")
    ap.add_argument("--pool", type=int, default=100, help="融合后候选池大小（实验12：加大池子只对强 reranker 有意义）")
    args = ap.parse_args()

    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    qs = []
    for r in rows:
        if r.get("question_type") == "null_query":
            continue
        facts = [e["fact"] for e in (r.get("evidence_list") or []) if e.get("fact")]
        titles = {e["title"] for e in (r.get("evidence_list") or []) if e.get("title")}
        if facts and titles:
            qs.append((r["query"], facts, titles, r.get("question_type", "?").split("_")[0]))

    if args.by_type:                      # 每类均衡抽 n//3，避免某类主导均值
        per, buckets = max(1, args.n // 3), {}
        for item in qs:
            buckets.setdefault(item[3], []).append(item)
        qs = [x for t in ("comparison", "inference", "temporal") for x in buckets.get(t, [])[:per]]
    else:
        qs = qs[: args.n]

    if args.reranker == "qwen":
        from rag.reranker_qwen import QwenReranker
        retr = HybridRetriever(load_corpus(), reranker=QwenReranker(), pool=args.pool)
    else:
        retr = HybridRetriever(load_corpus(), pool=args.pool)

    if args.by_type:
        _by_type(retr, qs, f"{args.reranker} pool={args.pool}")
        return

    agg = {name: [0.0, 0.0] for name in _CONFIGS}
    for q, facts, titles, _t in qs:
        cands = retr._fuse(q)
        sc = np.asarray(retr.reranker.predict([(q, d.text) for d in cands]))
        ranked = [cands[i] for i in np.argsort(-sc)]
        for name, sel in _CONFIGS.items():
            docs = sel(ranked)
            srcs = {d.source for d in docs}
            blob = _norm(" ".join(d.text for d in docs))
            agg[name][0] += len(titles & srcs) / len(titles)
            agg[name][1] += sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts)

    n = len(qs)
    print(f"样本 {n} 道多跳；title=标题级(代理)，fact=证据句真在检索文本里(贴近 correctness)\n")
    print(f"{'config':14s} {'title_recall':>13} {'fact_recall':>12}")
    for name in _CONFIGS:
        print(f"{name:14s} {agg[name][0] / n:>13.3f} {agg[name][1] / n:>12.3f}")


_KS = (4, 8, 16)


def _by_type(retr, qs, tag: str) -> None:
    """按题型 × top-k 报 fact 级召回。用来回答：**换 reranker 到底救了哪一类题**——
    端到端 correctness 只有 temporal 还没满分，若 temporal 的 fact 交付也上不去，
    说明瓶颈在检索交付；若 temporal fact 上去了而 correctness 不动，瓶颈就在推理端。"""
    agg: dict[str, dict[str, list]] = {}
    for q, facts, titles, t in qs:
        cands = retr._fuse(q)
        sc = np.asarray(retr.reranker.predict([(q, d.text) for d in cands]))
        ranked = [cands[i] for i in np.argsort(-sc)]
        slot = agg.setdefault(t, {f"@{k}": [] for k in _KS} | {"pool": [], "n": []})
        for k in _KS:
            blob = _norm(" ".join(d.text for d in ranked[:k]))
            slot[f"@{k}"].append(sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts))
        blob = _norm(" ".join(d.text for d in ranked))
        slot["pool"].append(sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts))
        slot["n"].append(1)

    cols = [f"@{k}" for k in _KS] + ["pool"]
    print(f"\n== fact 级召回 · 按题型 · reranker={tag} ==")
    print(f"{'type':12s} " + " ".join(f"{c:>8}" for c in cols) + f"{'n':>6}")
    for t in ("comparison", "inference", "temporal"):
        d = agg.get(t)
        if not d:
            continue
        print(f"{t:12s} " + " ".join(f"{sum(d[c]) / len(d[c]):>8.3f}" for c in cols) + f"{len(d['n']):>6}")


if __name__ == "__main__":
    main()
