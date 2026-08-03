"""Hybrid 检索：BM25(词法) + bge(向量) 加权融合 → bge-reranker 交叉编码重排。

本项目的新核心——一条生产级检索栈：
  1. 召回：BM25 和 Dense 各取 top-`pool`（两种「看法」互补，先把候选池撑大）。
  2. 融合：RRF（倒数排名融合，默认）——只看两路的**排名**、不看分数，对 BM25(无上界)/余弦([-1,1])
     不可比的量纲免疫，是 Elasticsearch/Weaviate 的标配；另留 min-max 加权（fusion="minmax"）做对照。
     同一片段两路都命中则分数叠加。
  3. 重排：把融合后的候选池丢给 cross-encoder（bge-reranker）——它把 (query, passage) 一起读、
     算一个更准的相关性分，比双塔向量更强但更贵，所以只重排少量候选。
  4. 返回重排后的 top-k。

实现同一套 `Retriever` 协议，所以对上层（生成 / 评测）完全透明。
"""

from __future__ import annotations

import os

import numpy as np

from rag.retriever import Doc, Hit
from rag.retriever_bm25 import BM25Retriever
from rag.retriever_dense import DenseRetriever

_RERANKER = "BAAI/bge-reranker-v2-m3"
# 池的**信息量**要与 chunk 匹配：pool×chunk ≈ 恒定（实验19 ⑥）。chunk=600 下 100→200 有收益、
# 200→300 只涨池覆盖不涨交付。可用 RAG_POOL 在 shell 层覆盖，用于新旧配置整体 A/B。
_POOL = int(os.environ.get("RAG_POOL", 200))


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
        pool: int = _POOL,
        reranker=_RERANKER,  # 模型名(→CrossEncoder) 或已构建的重排器对象(有 .predict，如 QwenReranker)
                             # 或 None = 不加载重排器（只用 _fuse，给融合层的对照实验省显存）
        fusion: str = "rrf",
        rrf_k: int = 60,
        max_per_source: int | None = None,
        bm25=None,
        dense=None,
    ) -> None:
        """`bm25` / `dense` 可传入任何满足 `Retriever` 协议的对象来替换默认两条腿——
        参数实验里用它注入「精确余弦」的轻量 dense 探针，免得每个 chunk 配置都建一个 Chroma 库。"""
        self.docs = docs
        self.bm25 = bm25 if bm25 is not None else BM25Retriever(docs)
        self.dense = dense if dense is not None else DenseRetriever(docs)
        self.w_bm25, self.w_dense, self.pool = w_bm25, w_dense, pool
        self.fusion, self._rrf_k = fusion, rrf_k
        self.max_per_source = max_per_source
        if reranker is None:                      # 不重排（只测召回/融合）
            self.reranker = None
        elif hasattr(reranker, "predict"):         # 已构建的重排器对象（如 QwenReranker）→ 直接用
            self.reranker = reranker
        else:                                     # 模型名字符串 → sentence-transformers CrossEncoder
            from sentence_transformers import CrossEncoder  # 延迟导入
            self.reranker = CrossEncoder(reranker)

    def _fuse(self, query: str) -> list[Doc]:
        return self._fuse_rrf(query) if self.fusion == "rrf" else self._fuse_minmax(query)

    def _fuse_rrf(self, query: str) -> list[Doc]:
        """RRF（倒数排名融合）：score(d) = Σ_路 w / (rrf_k + rank)。只看排名不看分数——
        BM25(无上界) 与余弦([-1,1]) 的分数量纲不可比，min-max 遇离群分会失真；RRF 免疫。"""
        fused: dict[str, list] = {}  # id -> [doc, score]
        for retr, w in ((self.bm25, self.w_bm25), (self.dense, self.w_dense)):
            for rank, h in enumerate(retr.search(query, k=self.pool), start=1):
                slot = fused.setdefault(h.doc.id, [h.doc, 0.0])
                slot[1] += w / (self._rrf_k + rank)
        ranked = sorted(fused.values(), key=lambda v: v[1], reverse=True)
        return [doc for doc, _ in ranked[: self.pool]]

    def _fuse_minmax(self, query: str) -> list[Doc]:
        """两路分数各自 min-max 归一化后按权重相加（对照基线）。同一片段两边都命中则叠加。"""
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
        """融合 → 重排 → 取 top-k。

        `max_per_source` 设为整数则启用 **source 去重**（每篇文章≤该数个名额），能顶高交付的
        *不同文章数*（title 级 recall）；**但**去重会挤掉同一篇里的证据 chunk、拉低 **fact 级 recall**
        与 correctness（实验8 实测：dedup=1 让 title 0.77→0.885 却把 fact 0.645→0.530）——所以
        **默认 None（不去重）**，靠 k=8 提高证据覆盖。去重仅作为可选项保留。（见 EXPERIMENTS 实验8-9。）"""
        cands = self._fuse(query)
        if not cands:
            return []
        if self.reranker is None:                           # 构造时 reranker=None → 直接交付融合顺序
            return [Hit(doc=d, score=0.0) for d in cands[:k]]
        scores = np.asarray(self.reranker.predict([(query, d.text) for d in cands]))
        order = np.argsort(-scores)
        if not self.max_per_source:                         # None/0 → 不去重，直接取相关性 top-k
            return [Hit(doc=cands[i], score=float(scores[i])) for i in order[:k]]
        hits: list[Hit] = []
        per_source: dict[str, int] = {}
        for i in order:                                     # 启用去重：每篇文章名额上限 max_per_source
            src = cands[i].source
            if per_source.get(src, 0) >= self.max_per_source:
                continue
            per_source[src] = per_source.get(src, 0) + 1
            hits.append(Hit(doc=cands[i], score=float(scores[i])))
            if len(hits) >= k:
                break
        return hits
