"""跑评测：agentic agent vs 单次基线，在数据集上对比。

    .venv/bin/python eval_run.py               # 本地对比表（需要模型凭据）
    .venv/bin/python eval_run.py --langsmith   # 另把 dataset+experiment 推到 LangSmith

分别报每个指标的均值，以及 agentic-vs-单次 的 delta。设了 LANGSMITH_TRACING=true 时，
每次运行还会 trace 到 LANGSMITH_PROJECT 下。

关于玩具库：只有 7 篇 + k=4，单次检索就已看到大半语料，所以这里的 delta **天生就小**。
这本身就是结论 —— agentic 迭代要在**大语料**上、单次查询捞不出多跳链时才显威力（这正是 P3
的真实语料 eval_multihop.py 要做的）。
"""

from __future__ import annotations

import re
import sys

from agent import build_agent
from eval_baseline import single_shot
from eval_dataset import DATASET, Example
from eval_metrics import METRICS
from retriever import InMemoryRetriever


def run_agentic(question: str, agent) -> dict:
    """跑 agentic agent，抽出 answer + 检索到的 sources + 检索次数。"""
    answer, sources, n = "", [], 0
    for chunk in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
        m = chunk["messages"][-1]
        mtype = getattr(m, "type", "")
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] == "rag_search":
                n += 1
        if mtype == "tool" and getattr(m, "name", None) == "rag_search":
            sources += re.findall(r"\[source:\s*([^\]]+)\]", m.content or "")
        if mtype == "ai" and (m.content or "").strip() and not getattr(m, "tool_calls", None):
            answer = m.content
    return {"answer": answer, "sources": sources, "n_search": n}


def _score(ex: Example, out: dict) -> dict:
    return {k: f(ex, out) for k, f in METRICS.items()} | {"n_search": out["n_search"]}


def _avg(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def _summary(label: str, rows: list[dict]) -> dict:
    print(f"\n== {label} ==")
    agg = {}
    for k in METRICS:
        agg[k] = _avg(rows, k)
        print(f"  {k:11s}: {agg[k]:.2f}")
    print(f"  avg_search : {_avg(rows, 'n_search'):.2f}")
    return agg


def main() -> None:
    retriever = InMemoryRetriever()
    agent = build_agent(retriever)
    ag_rows, bl_rows = [], []

    print(f"{'kind':12s} {'question':50s} {'agentic':>8s} {'single':>7s}")
    for ex in DATASET:
        ao = run_agentic(ex.question, agent)
        bo = single_shot(ex.question, retriever)
        ar, br = _score(ex, ao), _score(ex, bo)
        ag_rows.append(ar)
        bl_rows.append(br)
        print(f"{ex.kind:12s} {ex.question[:50]:50s} "
              f"n={ao['n_search']} c={ar['correct']:.0f}   c={br['correct']:.0f}")

    a = _summary("AGENTIC", ag_rows)
    b = _summary("SINGLE-SHOT baseline", bl_rows)
    print("\n== DELTA (agentic - single) ==")
    for k in METRICS:
        print(f"  {k:11s}: {a[k] - b[k]:+.2f}")

    if "--langsmith" in sys.argv:
        _push_to_langsmith(agent)


def _push_to_langsmith(agent) -> None:
    """地道的 LangSmith 评测：dataset + experiment + evaluators 面板。

    面向 langsmith>=0.1 的 evaluate()。不同版本的 evaluator/target 签名略有差异，
    装的版本不同就照着调一下。
    """
    from langsmith import Client, evaluate

    client = Client()
    name = "agentic-rag-eval"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name)
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"question": e.question} for e in DATASET],
            outputs=[{"reference": e.reference, "sources": e.sources, "kind": e.kind}
                     for e in DATASET],
        )

    def target(inputs: dict) -> dict:
        return run_agentic(inputs["question"], agent)

    def _make(metric_name, fn):
        def scorer(run, example):
            ex = Example(example.inputs["question"], example.outputs["reference"],
                         example.outputs["sources"], example.outputs["kind"])
            return {"key": metric_name, "score": fn(ex, run.outputs)}
        return scorer

    evaluate(
        target,
        data=name,
        evaluators=[_make(k, f) for k, f in METRICS.items()],
        experiment_prefix="agentic-rag",
        client=client,
    )
    print("\n[langsmith] 已推送 dataset + experiment 'agentic-rag-eval'。")


if __name__ == "__main__":
    main()
