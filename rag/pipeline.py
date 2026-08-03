"""检索 → 生成一次（grounded，非 agentic）—— 纯检索流水线的生成端。

和旧的 agentic 循环不同：这里**一次检索、一次生成**，把复杂度全压在检索侧（hybrid + reranker）。
返回 answer / sources / contexts，其中 contexts 供 RAGAS 算 faithfulness、context precision/recall。
"""

from __future__ import annotations

from rag.llm import build_model
from rag.retriever import Hit, Retriever

# 喂给模型的功能性系统提示（英文——只用检索到的上下文、标 [source:]、查不到就认怂）。
_SYSTEM = (
    "Answer ONLY from the provided context and cite sources as [source: ...]. "
    "If the context does not contain the answer, say you don't know — do not guess."
)


def answer(
    question: str,
    retriever: Retriever,
    k: int = 4,
    hits: list[Hit] | None = None,
    model=None,
) -> dict:
    """检索 top-k（或复用传入的 hits）→ 生成 grounded 答案。

    hits：外部已检索好就传进来，避免重复检索。
    model：批量评测时 build_model() 一次传进来，避免每题重建。
    """
    if hits is None:
        hits = retriever.search(question, k=k)
    sources = [h.doc.source for h in hits]
    contexts = [h.doc.text for h in hits]
    block = (
        "\n\n".join(f"[source: {h.doc.source}]\n{h.doc.text}" for h in hits)
        or "(no results)"
    )
    msg = (model or build_model()).invoke(
        [
            ("system", _SYSTEM),
            ("user", f"Context:\n{block}\n\nQuestion: {question}"),
        ]
    )
    return {
        "answer": msg.content or "",
        "sources": sources,
        "contexts": contexts,
        "hits": hits,
    }
