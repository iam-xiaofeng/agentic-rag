"""RAG 评测：在 MultiHop-RAG 上评 hybrid(BM25+bge+reranker) 流水线，用 openevals 的 LLM-as-judge，可推 LangSmith。

四个裁判（judge = grok 网关；prompt 来自 openevals，业界现成的评估器，非自研）：
  correctness          答案 vs 参考答案 —— 对不对
  groundedness         答案是否只由检索到的上下文支撑 —— 忠实度 / 不幻觉
  retrieval_relevance  检索到的上下文与问题相关吗 —— 检索质量
  helpfulness          答案是否真正回应了问题

每个指标是 [0,1] 连续分（continuous=True），judge 还给出打分理由（comment）。

    python eval_rag.py --n 8                          # 本地评分 + 打印
    python eval_rag.py --n 8 --upload                 # 造个本地抽样 dataset + experiment 推 LangSmith
    python eval_rag.py --dataset multihop-rag --n 12  # 在已上传的 LangSmith 数据集上跑实验（用 gold 证据）

需要模型凭据（OPENAI_* / RAG_MODEL，llm.py 会自动读 .env）；--upload 另需 LANGSMITH_API_KEY。
成本提醒：每题 = 1 次生成 + 4 次裁判调用，n 越大越费 token，先用小 n 试。
"""

from __future__ import annotations

import argparse

from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

from corpus_multihop import load_corpus, load_examples
from eval_dataset import Example
from llm import build_model
from rag import answer
from retriever_hybrid import HybridRetriever

# 每个裁判需要的变量（决定调用时传哪些 kwargs）。
_SPECS = {
    "correctness": (CORRECTNESS_PROMPT, {"inputs", "outputs", "reference_outputs"}),
    "groundedness": (RAG_GROUNDEDNESS_PROMPT, {"outputs", "context"}),
    "retrieval_relevance": (RAG_RETRIEVAL_RELEVANCE_PROMPT, {"inputs", "context"}),
    "helpfulness": (RAG_HELPFULNESS_PROMPT, {"inputs", "outputs"}),
}


def _sample(n: int) -> list[Example]:
    """从可答的 multihop 题里确定性等距抽 n 题。"""
    ans = [e for e in load_examples() if e.kind == "multihop"]
    step = max(1, len(ans) // n)
    return ans[::step][:n]


def _judges(judge):
    return {
        name: create_llm_as_judge(prompt=prompt, feedback_key=name, judge=judge, continuous=True)
        for name, (prompt, _needs) in _SPECS.items()
    }


def _call(name, fn, question, out, reference):
    """按该裁判所需变量装 kwargs 后调用，返回 [0,1] 分。"""
    needs = _SPECS[name][1]
    kw = {}
    if "inputs" in needs:
        kw["inputs"] = question
    if "outputs" in needs:
        kw["outputs"] = out["answer"]
    if "reference_outputs" in needs:
        kw["reference_outputs"] = reference
    if "context" in needs:
        kw["context"] = "\n\n".join(out["contexts"])
    return fn(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="评测题数")
    ap.add_argument("--topk", type=int, default=4, help="每题检索片段数")
    ap.add_argument("--upload", action="store_true", help="造本地抽样 dataset + experiment 推 LangSmith")
    ap.add_argument("--dataset", default=None,
                    help="在已有 LangSmith 数据集（如 multihop-rag）上跑实验，用它的 gold 证据评测")
    args = ap.parse_args()

    print("构建 hybrid 索引（BM25 + bge 向量 + reranker；首次会下模型/编码语料）...")
    retriever = HybridRetriever(load_corpus())
    model = build_model()
    judges = _judges(model)

    if args.dataset:
        _run_on_dataset(retriever, judges, model, args.dataset, args.n, args.topk)
        return

    data = _sample(args.n)
    if args.upload:
        _upload(retriever, data, judges, model, args.topk)
        return

    keys = list(_SPECS)
    rows = []
    print(f"\n{'correct':>8} {'ground':>7} {'retr':>6} {'help':>6}  question")
    for ex in data:
        try:
            out = answer(ex.question, retriever, k=args.topk, model=model)
            s = {k: _call(k, judges[k], ex.question, out, ex.reference)["score"] for k in keys}
        except Exception as e:  # 网关 5xx/超时等：跳过该题，保住整轮
            print(f"   [skip] {type(e).__name__}: {str(e)[:60]}")
            continue
        rows.append(s)
        print(f"{s['correctness']:8.2f} {s['groundedness']:7.2f} {s['retrieval_relevance']:6.2f} "
              f"{s['helpfulness']:6.2f}  {ex.question[:52]}")

    if not rows:
        print("\n没有成功的样本（多半是网关持续超时，稍后重试或换模型）。")
        return
    print(f"\n== 平均（openevals LLM-judge，judge=grok；{len(rows)} 题）==")
    for k in keys:
        print(f"  {k:20s}: {sum(r[k] for r in rows) / len(rows):.2f}")


def _upload(retriever, data: list[Example], judges, model, topk: int) -> None:
    """openevals 评估器直接作为 LangSmith evaluators：上传 dataset，evaluate() 跑 target + 打分。"""
    from langsmith import Client, evaluate

    client = Client()
    name = "agentic-rag-hybrid"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="MultiHop-RAG sample · hybrid(BM25+bge+reranker) · openevals")
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"question": e.question} for e in data],
            outputs=[{"reference": e.reference} for e in data],
        )

    def target(inputs: dict) -> dict:
        out = answer(inputs["question"], retriever, k=topk, model=model)
        return {"answer": out["answer"], "sources": out["sources"], "contexts": out["contexts"]}

    def _adapt(name, fn):
        def scorer(run, example):
            r = _call(name, fn, example.inputs["question"], run.outputs,
                      example.outputs.get("reference", ""))
            return {"key": name, "score": r["score"], "comment": r.get("comment")}
        return scorer

    evaluate(
        target,
        data=name,
        evaluators=[_adapt(k, judges[k]) for k in _SPECS],
        experiment_prefix="hybrid-openevals",
        client=client,
    )
    print(f"\n[langsmith] 已推送 dataset + experiment '{name}'。")


def _context_recall(run, example) -> dict:
    """确定性 context recall：gold 文章标题里有多少被检索到（不花 LLM，用上传的 gold 证据）。"""
    gold = set(example.outputs.get("gold_titles") or [])
    got = set(run.outputs.get("sources") or [])
    return {"key": "context_recall", "score": (len(gold & got) / len(gold)) if gold else None}


def _run_on_dataset(retriever, judges, model, dataset: str, n: int, topk: int) -> None:
    """在已有的 LangSmith 数据集（如 multihop-rag）上跑实验：
    target = 检索流水线；评估器 = 4 个 openevals LLM-judge + 确定性 context_recall（用 gold 证据）。"""
    from langsmith import Client, evaluate

    client = Client()
    examples = [e for e in client.list_examples(dataset_name=dataset)
                if (e.outputs or {}).get("kind") == "multihop"]
    if not examples:
        print(f"数据集 {dataset!r} 里没有 multihop 例（或数据集不存在）。")
        return
    step = max(1, len(examples) // n)
    examples = examples[::step][:n]
    print(f"在 {dataset!r} 上取 {len(examples)} 道 multihop 跑实验（每题 1 生成 + 4 裁判）...")

    def target(inputs: dict) -> dict:
        out = answer(inputs["question"], retriever, k=topk, model=model)
        return {"answer": out["answer"], "sources": out["sources"], "contexts": out["contexts"]}

    def _adapt(name, fn):
        def scorer(run, example):
            r = _call(name, fn, example.inputs["question"], run.outputs,
                      example.outputs.get("reference", ""))
            return {"key": name, "score": r["score"], "comment": r.get("comment")}
        return scorer

    evaluators = [_adapt(k, judges[k]) for k in _SPECS] + [_context_recall]
    results = evaluate(target, data=examples, evaluators=evaluators,
                       experiment_prefix=f"hybrid-on-{dataset}", max_concurrency=4, client=client)
    try:
        agg: dict[str, list] = {}
        for row in results:
            for er in (row["evaluation_results"]["results"] or []):
                if er.score is not None:
                    agg.setdefault(er.key, []).append(er.score)
        print(f"\n== 平均（{dataset}，{len(examples)} 题）==")
        for k in list(_SPECS) + ["context_recall"]:
            v = agg.get(k, [])
            print(f"  {k:20s}: {sum(v) / len(v):.2f}" if v else f"  {k:20s}: n/a")
    except Exception as e:  # 聚合读取失败不影响实验已落库
        print(f"（本地聚合失败，去 LangSmith 看即可：{type(e).__name__}: {str(e)[:60]}）")
    print(f"[langsmith] 已在数据集 {dataset!r} 上跑完实验（前缀 hybrid-on-{dataset}）。")


if __name__ == "__main__":
    main()
