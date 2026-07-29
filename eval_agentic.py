"""在 LangSmith 数据集（multihop-rag）上评 **agentic RAG（多跳）**，按 question_type 均衡抽样，结果推 LangSmith。

    python eval_agentic.py --per-type 8        # 每类 8 题（comparison/inference/temporal/null）= 32（默认，快）
    python eval_agentic.py --per-type 25        # 每类 25 = 100（更稳但更慢 ~1-2h）

target = agentic agent（多跳，检索后端 hybrid）；每题把**所有 hop** 检索到的证据文本取并集 → 算召回。
评估器：
  correctness / groundedness / retrieval_relevance / helpfulness  —— openevals LLM-judge
  context_recall_fact  —— 确定性（gold 证据句是否真在检索文本里；比标题级准，见 EXPERIMENTS 实验9），null 跳过
  refused              —— 确定性（null_query 该拒答；答得出来的题里 refused=1 反而是漏答）

⚠️ 成本：每题多跳 ~5-9 次生成 + 4 次裁判；每类 8（=32）约 20-40 分钟，每类 25（=100）约 1~2 小时。网关偶发 522/524 会让个别题 errored（不影响整轮）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from collections import defaultdict

from langsmith import Client, evaluate

from agent import build_agent
from corpus_multihop import load_corpus
from eval_common import SPECS, call_judge, context_recall_fact, make_judges, refused
from llm import build_judge, build_model
from retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parent / "data"
_TYPES = ["comparison_query", "inference_query", "temporal_query", "null_query"]
_SRC = re.compile(r"\[source:\s*([^\]]+)\]")


def _type_map() -> dict[str, str]:
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {r["query"]: r.get("question_type", "?") for r in raw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="multihop-rag")
    ap.add_argument("--per-type", type=int, default=8, help="每种 question_type 抽几题（默认 8=32 题，快；25=100 更稳）")
    ap.add_argument("--reranker", default="bge", choices=["bge", "qwen"],
                    help="重排器：bge(默认,快) 或 qwen(Qwen3-Reranker-4B,instruction-aware,慢但 fact 召回更高)")
    ap.add_argument("--concurrency", type=int, default=4, help="evaluate 并发数（Qwen3-4B 慢+网关限流时降到 1）")
    ap.add_argument("--types", default="", help="只评这些题型（逗号分隔，如 temporal）。默认全四类。"
                                               "用于把样本预算压在**有区分度**的题型上——comparison/inference 已双双 1.00，多抽也没信息量")
    ap.add_argument("--model", default=None,
                    help="**答题**模型（缺省读 .env 的 RAG_MODEL）。裁判模型固定为 RAG_MODEL 不受影响——"
                         "换裁判等于换尺子，分数就不可比了")
    args = ap.parse_args()

    types = [f"{t.strip()}_query" if not t.strip().endswith("_query") else t.strip()
             for t in args.types.split(",") if t.strip()] or _TYPES

    qtype = _type_map()
    client = Client()
    by_type: dict[str, list] = defaultdict(list)
    for e in client.list_examples(dataset_name=args.dataset):
        by_type[qtype.get(e.inputs.get("question"), "?")].append(e)

    picked = []
    for t in types:
        pool = by_type.get(t, [])
        step = max(1, len(pool) // args.per_type)
        picked += pool[::step][:args.per_type]
    print("抽样：" + " | ".join(f"{t.split('_')[0]} {min(len(by_type.get(t, [])), args.per_type)}"
                                for t in types) + f"  = {len(picked)} 题")

    if args.reranker == "qwen":
        from reranker_qwen import QwenReranker
        retr = HybridRetriever(load_corpus(), reranker=QwenReranker())
    else:
        retr = HybridRetriever(load_corpus())
    agent = build_agent(retr, args.model)
    judges = make_judges(build_judge())        # 裁判恒为 RAG_JUDGE_MODEL/RAG_MODEL，跨实验可比

    def _run_once(question: str) -> dict:
        got, contexts, answer = set(), [], ""
        for chunk in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
            m = chunk["messages"][-1]
            if getattr(m, "type", "") == "tool" and getattr(m, "name", None) == "rag_search":
                got |= set(_SRC.findall(m.content or ""))
                contexts.append(m.content or "")
            if getattr(m, "type", "") == "ai" and (m.content or "").strip() and not getattr(m, "tool_calls", None):
                answer = m.content
        return {"answer": answer, "sources": sorted(got), "contexts": contexts}

    def target(inputs: dict) -> dict:
        """网关 502/524 会整题打空 → **绝不能吞掉**：吞了就变成 correctness=0 计入均值，
        把两轮实验读成「模型变差了」（实测 qwen 轮 temporal 8 题空了 7 题）。这里长退避重试
        （Cloudflare 502 建议 retry_after=60，SDK 内置退避只有秒级、扛不住源站分钟级抽风），
        仍失败就 **raise** —— LangSmith 把该题标 errored 并**排除出均值**，而不是当 0 分。"""
        last: Exception | None = None
        for wait in (0, 30, 90):                      # 3 次尝试，总退避 ~2 分钟
            if wait:
                time.sleep(wait)
            try:
                return _run_once(inputs["question"])
            except Exception as e:                    # noqa: BLE001 —— 网关各类 5xx/超时
                last = e
                print(f"  ⚠️ 网关失败({type(e).__name__})，{wait or 0}s 后已重试…", flush=True)
        raise RuntimeError(f"网关连续 3 次失败，该题记 errored（不计入均值）：{type(last).__name__}: {str(last)[:120]}")

    def _adapt(name, fn):
        def scorer(run, example):
            ref = example.outputs.get("reference", "")
            if name == "correctness" and not ref:   # null_query 无 gold 答案 → 跳过 correctness
                return {"key": "correctness", "score": None}
            r = call_judge(name, fn, example.inputs["question"], run.outputs, ref)
            return {"key": name, "score": r["score"], "comment": r.get("comment")}
        return scorer

    evaluators = [_adapt(k, judges[k]) for k in SPECS] + [context_recall_fact, refused]
    results = evaluate(target, data=picked, evaluators=evaluators,
                       experiment_prefix=f"agentic-{args.reranker}-{(args.model or 'default').split('/')[-1]}-on-{args.dataset}",
                       max_concurrency=args.concurrency, client=client)

    # 本地按 question_type 聚合（去 LangSmith 也能看，这里省得点）
    try:
        agg: dict = defaultdict(lambda: defaultdict(list))
        n_err = 0
        for row in results:
            t = qtype.get(row["example"].inputs.get("question"), "?").split("_")[0]
            if not (row["run"].outputs or {}).get("answer"):   # 网关打空/errored：不计入均值
                n_err += 1
                continue
            for er in (row["evaluation_results"]["results"] or []):
                if er.score is not None:
                    agg[t][er.key].append(er.score)
        metrics = ["correctness", "groundedness", "retrieval_relevance", "helpfulness", "context_recall_fact", "refused"]
        print("\n== 按 question_type 均值 ==")
        print(f"{'type':11s} " + " ".join(f"{m[:9]:>10}" for m in metrics) + "     n有效")
        for t in ["comparison", "inference", "temporal", "null"]:
            d = agg.get(t, {})
            n = max((len(v) for v in d.values()), default=0)
            print(f"{t:11s} " + " ".join(
                f"{sum(d[m]) / len(d[m]):>10.2f}" if d.get(m) else f"{'-':>10}" for m in metrics) + f"   {n:>5}")
        if n_err:
            print(f"\n⚠️ {n_err} 题网关连续失败被判 errored，已**排除**出均值（不当 0 分算）——"
                  f"若 n有效 明显小于抽样数，这轮的对比不可信，重跑。")
    except Exception as e:
        print(f"（本地聚合失败，去 LangSmith 看即可：{type(e).__name__}: {str(e)[:60]}）")
    print(f"\n[langsmith] agentic 实验已推到数据集 {args.dataset!r}（前缀 agentic-{args.reranker}-on-{args.dataset}）。")


if __name__ == "__main__":
    main()
