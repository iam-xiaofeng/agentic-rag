"""在 LangSmith 数据集（multihop-rag）上评 **agentic RAG（多跳）**，按 question_type 均衡抽样，结果推 LangSmith。

    python eval_agentic.py --per-type 25       # 每类 25 题（comparison/inference/temporal/null）= 100
    python eval_agentic.py --per-type 1         # 冒烟：每类 1 题

target = agentic agent（多跳，检索后端 hybrid）；每题把**所有 hop** 检索到的 source 取并集 → 算 context_recall。
评估器：
  correctness / groundedness / retrieval_relevance / helpfulness  —— openevals LLM-judge
  context_recall  —— 确定性（gold 文章标题 ∩ 检索并集），null 无 gold 则跳过
  refused         —— 确定性（null_query 该拒答；答得出来的题里 refused=1 反而是漏答）

⚠️ 成本大：每题多跳 ~5-9 次生成 + 4 次裁判，100 题约 1~2 小时、上千次 grok 调用，网关偶发 522/524 会让个别题 errored（不影响整轮）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

from langsmith import Client, evaluate

from agent import build_agent
from corpus_multihop import load_corpus
from eval_rag import _SPECS, _call, _context_recall, _judges
from llm import build_model
from retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parent / "data"
_TYPES = ["comparison_query", "inference_query", "temporal_query", "null_query"]
_REFUSAL = ("don't know", "do not know", "not contain", "no information", "cannot find",
            "not available", "no relevant", "isn't specified", "is not specified",
            "does not specify", "not mention", "no answer")
_SRC = re.compile(r"\[source:\s*([^\]]+)\]")


def _type_map() -> dict[str, str]:
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {r["query"]: r.get("question_type", "?") for r in raw}


def _refused(run, example) -> dict:
    ans = (run.outputs.get("answer") or "").lower()
    return {"key": "refused", "score": 1.0 if any(c in ans for c in _REFUSAL) else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="multihop-rag")
    ap.add_argument("--per-type", type=int, default=25, help="每种 question_type 抽几题")
    args = ap.parse_args()

    qtype = _type_map()
    client = Client()
    by_type: dict[str, list] = defaultdict(list)
    for e in client.list_examples(dataset_name=args.dataset):
        by_type[qtype.get(e.inputs.get("question"), "?")].append(e)

    picked = []
    for t in _TYPES:
        pool = by_type.get(t, [])
        step = max(1, len(pool) // args.per_type)
        picked += pool[::step][:args.per_type]
    print("抽样：" + " | ".join(f"{t.split('_')[0]} {min(len(by_type.get(t, [])), args.per_type)}"
                                for t in _TYPES) + f"  = {len(picked)} 题")

    retr = HybridRetriever(load_corpus())
    agent = build_agent(retr)
    judges = _judges(build_model())

    def target(inputs: dict) -> dict:
        got, contexts, answer = set(), [], ""
        try:
            for chunk in agent.stream({"messages": [("user", inputs["question"])]}, stream_mode="values"):
                m = chunk["messages"][-1]
                if getattr(m, "type", "") == "tool" and getattr(m, "name", None) == "rag_search":
                    got |= set(_SRC.findall(m.content or ""))
                    contexts.append(m.content or "")
                if getattr(m, "type", "") == "ai" and (m.content or "").strip() and not getattr(m, "tool_calls", None):
                    answer = m.content
        except Exception:  # 网关 522/524 等：给个空结果，别毁整轮
            return {"answer": "", "sources": [], "contexts": []}
        return {"answer": answer, "sources": sorted(got), "contexts": contexts}

    def _adapt(name, fn):
        def scorer(run, example):
            ref = example.outputs.get("reference", "")
            if name == "correctness" and not ref:   # null_query 无 gold 答案 → 跳过 correctness
                return {"key": "correctness", "score": None}
            r = _call(name, fn, example.inputs["question"], run.outputs, ref)
            return {"key": name, "score": r["score"], "comment": r.get("comment")}
        return scorer

    evaluators = [_adapt(k, judges[k]) for k in _SPECS] + [_context_recall, _refused]
    results = evaluate(target, data=picked, evaluators=evaluators,
                       experiment_prefix=f"agentic-on-{args.dataset}", max_concurrency=4, client=client)

    # 本地按 question_type 聚合（去 LangSmith 也能看，这里省得点）
    try:
        agg: dict = defaultdict(lambda: defaultdict(list))
        for row in results:
            t = qtype.get(row["example"].inputs.get("question"), "?").split("_")[0]
            for er in (row["evaluation_results"]["results"] or []):
                if er.score is not None:
                    agg[t][er.key].append(er.score)
        metrics = ["correctness", "groundedness", "retrieval_relevance", "helpfulness", "context_recall", "refused"]
        print("\n== 按 question_type 均值 ==")
        print(f"{'type':11s} " + " ".join(f"{m[:9]:>10}" for m in metrics))
        for t in ["comparison", "inference", "temporal", "null"]:
            d = agg.get(t, {})
            print(f"{t:11s} " + " ".join(
                f"{sum(d[m]) / len(d[m]):>10.2f}" if d.get(m) else f"{'-':>10}" for m in metrics))
    except Exception as e:
        print(f"（本地聚合失败，去 LangSmith 看即可：{type(e).__name__}: {str(e)[:60]}）")
    print(f"\n[langsmith] agentic 实验已推到数据集 {args.dataset!r}（前缀 agentic-on-{args.dataset}）。")


if __name__ == "__main__":
    main()
