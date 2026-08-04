"""CLI：agentic RAG —— 看**一题**的完整过程（对照 run.py 的单次强检索流水线）。

    python run_agentic.py "Who is the spouse of the Green performer?"
    python run_agentic.py --agent planner "..."          # 看 plan→逐跳搜→合成 的链条
    python run_agentic.py --model gpt-5.6-luna "..."     # 换答题模型（查 endpoints.json）
    python run_agentic.py --corpus multihoprag "..."     # 换语料（⚠️ 该数据集已被实验25 证伪）

两条控制流的差别在这里最直观：**react 拿到桥接实体后经常直接停**（MuSiQue 实测 21 题里 6 题
只搜 1 次），planner 的第二跳是控制流保证的。谁更好由 `evals/run_matrix.py` 配对跑数据决定，
这个 CLI 只负责让过程可见。
"""

from __future__ import annotations

import argparse
import sys

from rag.agent import build_runner
from rag.retriever_bm25 import BM25Retriever
from rag.retriever_dense import DenseRetriever
from rag.retriever_hybrid import HybridRetriever
from rag.tools import topk

_RETRIEVERS = {"bm25": BM25Retriever, "dense": DenseRetriever, "hybrid": HybridRetriever}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+", help="要问的问题")
    ap.add_argument("--retriever", choices=list(_RETRIEVERS), default="hybrid",
                    help="检索后端（默认 hybrid = BM25 + bge 向量 + reranker）")
    ap.add_argument("--corpus", choices=["musique", "multihoprag"], default="musique")
    ap.add_argument("--agent", choices=["react", "planner"], default=None)
    ap.add_argument("--model", default=None, help="答题模型名（查 endpoints.json）")
    args = ap.parse_args()
    question = " ".join(args.question).strip()

    if args.corpus == "musique":
        from rag.corpus_musique import load_corpus
    else:
        from rag.corpus_multihop import load_corpus
    print(f"构建 {args.retriever} 索引（首次会下 bge 模型 / 编码语料）...", file=sys.stderr)
    run = build_runner(_RETRIEVERS[args.retriever](load_corpus()), args.model, args.agent)
    print(f"[{run.kind}] k={topk()}  模型={args.model or '(endpoints.json 的 _roles.answer)'}",
          file=sys.stderr)

    out = run(question)

    for i, ctx in enumerate(out["contexts"], start=1):
        srcs = [ln for ln in ctx.splitlines() if ln.startswith("[source:")][:4]
        print(f"\n  ↳ 检索 #{i} → {len(ctx)} 字符，前几个 source：")
        for s in srcs:
            print(f"      {s[:100]}")
    if out.get("plan"):
        print("\n=== 链条（planner）===")
        for i, c in enumerate(out["plan"], start=1):
            print(f"  {i}. {c['goal']}\n     query: {c['query']!r}\n     → {c['answer'] or 'NOT FOUND'}")

    print(f"\n=== 答案 ===\n{out['answer']}")
    print("\n=== KEY EVIDENCE（agent 自认最关键的几句）===")
    for s in out["cited"]:
        print(f"  - {s[:160]}")
    print(f"\n[过程] 检索 {out['n_search']} 次 / LLM 调用 {out['n_llm_calls']} 次 / "
          f"引用 {len(out['cited'])} 句 / 认怂 {out['n_unsupported']} 环")


if __name__ == "__main__":
    main()
