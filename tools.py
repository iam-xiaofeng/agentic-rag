"""模型唯一能看到的 agent 工具：rag_search。

这是暴露给模型的**唯一**工具。「建库」不是工具 —— 它是检索器背后的离线基础设施，
agent 只管检索。

注意：下面的 `description` 和 `_NO_RESULTS` 是**喂给模型的功能性文本**（模型靠它们决定行为），
且报告里的评测数字是基于这版英文文案得到的，所以保留英文；人类看的 docstring/注释用中文。
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from retriever import Retriever

# 明确的「查无」哨兵：引导模型换关键词重试，或在试过几个角度后如实告知库里没有答案。
_NO_RESULTS = (
    "NO_RESULTS: nothing in the knowledge base matched this query. "
    "Reformulate with different keywords, or — if you have already tried a few "
    "angles — tell the user the knowledge base does not contain the answer."
)


def make_rag_search(retriever: Retriever) -> StructuredTool:
    """把一个 retriever 绑成 `rag_search` LangChain 工具。"""

    def rag_search(query: str) -> str:
        hits = retriever.search(query, k=4)
        if not hits:
            return _NO_RESULTS
        return "\n\n".join(
            f"[source: {h.doc.source}] (score={h.score:.0f})\n{h.doc.text}"
            for h in hits
        )

    return StructuredTool.from_function(
        rag_search,
        name="rag_search",
        description=(
            "Search the knowledge base and return cited snippets. Call this before "
            "answering any factual question. If the snippets are insufficient, call "
            "AGAIN with a refined or alternative query — multi-hop is expected."
        ),
    )
