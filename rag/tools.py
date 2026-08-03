"""模型唯一能看到的 agent 工具：rag_search。

这是暴露给模型的**唯一**工具。「建库」不是工具 —— 它是检索器背后的离线基础设施，
agent 只管检索。

注意：下面的 `description` 和 `_NO_RESULTS` 是**喂给模型的功能性文本**（模型靠它们决定行为），
且报告里的评测数字是基于这版英文文案得到的，所以保留英文；人类看的 docstring/注释用中文。
"""

from __future__ import annotations

import os

from langchain_core.tools import StructuredTool

from rag.retriever import Retriever

# k=32 而不是 16：chunk 从 1200 缩到 600 后（实验19-20），同样的 k 只交付一半字符。
# 32×516 ≈ 16.5k 字符 ≈ 旧配置 16×1042，**上下文预算不变**，而重排后 fact 级召回三类均值
# 0.749 → 0.779（每型 60 题实测，三个题型齐涨）。RAG_TOPK 可在 shell 层覆盖，用于新旧配置 A/B。
_TOPK = int(os.environ.get("RAG_TOPK", 32))

# 明确的「查无」哨兵：引导模型换关键词重试，或在试过几个角度后如实告知库里没有答案。
_NO_RESULTS = (
    "NO_RESULTS: nothing in the knowledge base matched this query. "
    "Reformulate with different keywords, or — if you have already tried a few "
    "angles — tell the user the knowledge base does not contain the answer."
)


def make_rag_search(retriever: Retriever) -> StructuredTool:
    """把一个 retriever 绑成 `rag_search` LangChain 工具。"""

    def rag_search(query: str) -> str:
        hits = retriever.search(query, k=_TOPK)
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
