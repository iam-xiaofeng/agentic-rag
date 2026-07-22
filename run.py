"""CLI：在真实语料（MultiHop-RAG）上跑 hybrid 检索（BM25 + bge 向量 + reranker），可选生成答案。

    python run.py "Who is Sam Bankman-Fried?"     # 检索 + grounded 生成
    python run.py --topk 8 "..."                  # 看更多候选
    python run.py --no-gen "..."                  # 只看检索排序，不调模型

首次运行会下 bge 模型并对 ~6194 个片段编码（之后走 .cache 缓存，秒级）。
"""

from __future__ import annotations

import argparse
import sys

from corpus_multihop import load_corpus
from retriever_hybrid import HybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser(description="hybrid(BM25+bge)+reranker 检索流水线")
    ap.add_argument("question", nargs="+", help="要问的问题")
    ap.add_argument("--topk", type=int, default=4, help="返回片段数（默认 4）")
    ap.add_argument("--no-gen", action="store_true", help="只检索、不调模型生成")
    args = ap.parse_args()
    question = " ".join(args.question).strip()

    print("构建 hybrid 索引（BM25 + bge 向量 + reranker；首次会下模型 / 编码语料）...", file=sys.stderr)
    retriever = HybridRetriever(load_corpus())
    hits = retriever.search(question, k=args.topk)

    print(f"\n=== 检索 top-{args.topk}（reranker 相关性分）===")
    for i, h in enumerate(hits, 1):
        print(f"{i:>2}. {h.score:6.2f}  {h.doc.source}")

    if args.no_gen:
        return

    from rag import answer

    print("\n=== grounded 生成（需模型 key）===", file=sys.stderr)
    out = answer(question, retriever, k=args.topk, hits=hits)
    print(out["answer"])


if __name__ == "__main__":
    main()
