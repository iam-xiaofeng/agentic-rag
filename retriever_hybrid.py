"""Hybrid 检索：BM25(词法) + bge(向量) 加权融合 → bge-reranker 交叉编码重排。

本项目的新核心——一条生产级检索栈：
  1. 召回：BM25 和 Dense 各取 top-`pool`（两种「看法」互补，先把候选池撑大）。
  2. 融合：各自分数 min-max 归一化后按权重相加（对应「算出权重」）。同一片段两边都命中则叠加。
  3. 重排：把融合后的候选池丢给 cross-encoder（bge-reranker）——它把 (query, passage) 一起读、
     算一个更准的相关性分，比双塔向量更强但更贵，所以只重排少量候选。
  4. 返回重排后的 top-k。

实现同一套 `Retriever` 协议，所以对上层（生成 / 评测）完全透明。
"""

from __future__ import annotations

import numpy as np

from retriever import Doc, Hit
from retriever_bm25 import BM25Retriever
from retriever_dense import DenseRetriever

_RERANKER = "BAAI/bge-reranker-v2-m3"


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class HybridRetriever:
    """BM25 + Dense 加权融合，再用 bge-reranker 重排。"""

    def __init__(
        self,
        docs: list[Doc],
        w_bm25: float = 0.5,
        w_dense: float = 0.5,
        pool: int = 20,
        reranker: str = _RERANKER,
    ) -> None:
        from sentence_transformers import CrossEncoder  # 延迟导入

        self.docs = docs
        self.bm25 = BM25Retriever(docs)
        self.dense = DenseRetriever(docs)
        self.w_bm25, self.w_dense, self.pool = w_bm25, w_dense, pool
        self.reranker = CrossEncoder(reranker)

    def _fuse(self, query: str) -> list[Doc]:
        """两路召回 → 归一化加权融合 → 候选池（按融合分降序）。"""
        fused: dict[str, list] = {}  # id -> [doc, fused_score]
        for retr, w in ((self.bm25, self.w_bm25), (self.dense, self.w_dense)):
            hits = retr.search(query, k=self.pool)
            if not hits:
                continue
            norm = _minmax(np.array([h.score for h in hits]))
            for h, s in zip(hits, norm):
                slot = fused.setdefault(h.doc.id, [h.doc, 0.0])
                slot[1] += w * float(s)
        ranked = sorted(fused.values(), key=lambda v: v[1], reverse=True)
        return [doc for doc, _ in ranked[: self.pool]]

    def search(self, query: str, k: int = 4) -> list[Hit]:
        cands = self._fuse(query)
        if not cands:
            return []
        scores = self.reranker.predict([(query, d.text) for d in cands])
        order = np.argsort(-np.asarray(scores))[:k]
        return [Hit(doc=cands[i], score=float(scores[i])) for i in order]
