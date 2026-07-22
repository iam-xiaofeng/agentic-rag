"""非 agentic 基线：只检索一次 top-k、只生成一次。

这是回答「值不值」的对照组 —— 迭代式 agentic 检索，到底比朴素的 retrieve-then-generate
流水线强多少？评测报告的就是这两者的 delta。
"""

from __future__ import annotations

from agent import build_model
from retriever import InMemoryRetriever, Retriever

# 喂给模型的功能性系统提示（保留英文：评测数字基于此文案）。
_BASELINE_SYSTEM = (
    "Answer ONLY from the provided context and cite sources as [source: ...]. "
    "If the context does not contain the answer, say you don't know — do not guess."
)


def single_shot(question: str, retriever: Retriever | None = None) -> dict:
    """一次检索、一次生成。无迭代、无改写。"""
    retriever = retriever or InMemoryRetriever()
    hits = retriever.search(question, k=4)
    sources = [h.doc.source for h in hits]
    context = "\n\n".join(
        f"[source: {h.doc.source}]\n{h.doc.text}" for h in hits
    ) or "(no results)"
    msg = build_model().invoke(
        [
            ("system", _BASELINE_SYSTEM),
            ("user", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    return {"answer": msg.content or "", "sources": sources, "n_search": 1}
