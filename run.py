"""CLI：在真实语料（MultiHop-RAG，BM25 检索）上提一个问题，观察 agentic 检索过程。

    python run.py "Who is the individual associated with the cryptocurrency industry facing a criminal trial?"

打印的 "rag_search #N" 行 + 末尾计数，让你直接**看见过程层**：检索了几次、怎么改写的、
什么时候决定停。

首次运行会在约 6194 个语料片段上建 BM25 索引（几秒）。完整逐步 trace（每次工具调用、
延迟、token）在设了 LANGSMITH_TRACING=true 和 LANGSMITH_API_KEY 时会出现在 LangSmith。
"""

from __future__ import annotations

import sys

from agent import build_agent
from corpus_multihop import load_corpus
from retriever_bm25 import BM25Retriever


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Q: ").strip()
    print("构建 BM25 索引（MultiHop-RAG 语料）...", file=sys.stderr)
    agent = build_agent(BM25Retriever(load_corpus()))

    n_search = 0
    for chunk in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
        msg = chunk["messages"][-1]
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] == "rag_search":
                n_search += 1
                print(f"  ↳ rag_search #{n_search}: {tc['args'].get('query')!r}")
        msg.pretty_print()

    print(f"\n[过程] 本次 rag_search 调用次数: {n_search}")


if __name__ == "__main__":
    main()
