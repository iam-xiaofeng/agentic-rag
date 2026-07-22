"""CLI：agentic RAG —— 模型自己决定 该不该查 / 查几次 / 何时停（对照 run.py 的单次强检索流水线）。

    python run_agentic.py "Who is the individual ... facing a criminal trial?"

打印每次 rag_search 的改写 query + 末尾次数，让「过程层」可见。检索用 BM25（词法的欠覆盖正好逼出多跳）。
"""

from __future__ import annotations

import sys

from agent import build_agent
from corpus_multihop import load_corpus
from retriever_bm25 import BM25Retriever


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Q: ").strip()
    print("构建 BM25 索引（agentic 侧用词法，好让多跳可观测）...", file=sys.stderr)
    agent = build_agent(BM25Retriever(load_corpus()))

    n = 0
    for chunk in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
        m = chunk["messages"][-1]
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] == "rag_search":
                n += 1
                print(f"  ↳ rag_search #{n}: {tc['args'].get('query')!r}")
        m.pretty_print()

    print(f"\n[过程] rag_search 次数: {n}")


if __name__ == "__main__":
    main()
