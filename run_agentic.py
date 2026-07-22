"""CLI：agentic RAG —— 模型自己决定 该不该查 / 查几次 / 何时停（对照 run.py 的单次强检索流水线）。

    python run_agentic.py "Who is the individual ... facing a criminal trial?"
    python run_agentic.py --retriever bm25 "..."      # 换后端，看"弱检索逼出更多多跳"

默认用 **hybrid**（BM25 + bge 向量 + reranker）——和流水线同一套强检索。想观察词法欠覆盖如何逼出多跳，
就 `--retriever bm25`。打印每次 rag_search 的改写 query + 末尾次数，让「过程层」可见。
"""

from __future__ import annotations

import argparse
import sys

from agent import build_agent
from corpus_multihop import load_corpus
from retriever_bm25 import BM25Retriever
from retriever_dense import DenseRetriever
from retriever_hybrid import HybridRetriever

_RETRIEVERS = {"bm25": BM25Retriever, "dense": DenseRetriever, "hybrid": HybridRetriever}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+", help="要问的问题")
    ap.add_argument("--retriever", choices=list(_RETRIEVERS), default="hybrid",
                    help="检索后端（默认 hybrid = BM25 + bge 向量 + reranker）")
    args = ap.parse_args()
    question = " ".join(args.question).strip()

    print(f"构建 {args.retriever} 索引（首次会下 bge 模型 / 编码语料）...", file=sys.stderr)
    agent = build_agent(_RETRIEVERS[args.retriever](load_corpus()))

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
