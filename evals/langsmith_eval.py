"""把评测集 + **同一个 LLM-as-judge** 上传到 LangSmith，跑成一个可翻可比的 experiment。

    python evals/langsmith_eval.py --per-type 10                # 8 类型 × 10 = 80 题
    python evals/langsmith_eval.py --per-type 10 --dataset-only # 只建/更新数据集，不跑

━━━ 这个脚本和 `eval_judge.py` 的分工 ━━━

**统计结论仍然只认 `eval_judge.py`**（确定性 dump + 同题配对 bootstrap，见 README §5）。
本脚本的产出是**可翻的过程**：LangSmith 上一条 example 一行，点进去能看到
agent 发了哪几个 query、每次 `rag_search` 返回什么、裁判给了什么理由。
**它解决的是"这一题为什么错"，不是"这个改动值多少分"。**

━━━ 一条不许违反的约束 ━━━

**裁判提示词直接 import `eval_judge._PROMPT`，绝不在这里重写一份。**
本项目今天已经栽过两次"一个概念两处定义"：拒答词表（两份词表 → 同一批答案得到互相矛盾的
拒答率）、`__meta__` 的 topk 默认值（实际 k=8 记成 k=32）。
**再抄一份裁判提示词，就是让 LangSmith 上的分和 `eval_judge.py` 的分永远对不上。**

━━━ 评估器 ━━━

| 评估器 | 参照系 | 谁算的 |
|---|---|---|
| `correct` | gold 答案 | LLM 裁判（与 eval_judge 同一份提示词、同一个裁判模型） |
| `grounded` | agent **实际检索到的全部上下文** | 同上 |
| `delivered` | gold 证据 ∩ 检索上下文 | **确定性**，纯子串匹配、零裁判噪声 |
| `n_search` / `n_ctx_chars` | — | 成本项，**必须与分数同表**（README §5.5） |

⚠️ `null_query`（不可答题）没有 gold 答案，`correct` 对它无意义 ——
改用 `refused_ok`：**拒答或明确自曝证据不足才算对**（三分类见 `rag/agent.answer_stance`）。
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys

if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import json
import time

from langsmith import Client, evaluate

from evals.eval_agentic import _fact_fate, build_retriever, load_benchmark, sample
from evals.eval_judge import _JSON, _PROMPT, _retrieved_blob
from rag.agent import answer_stance, build_runner
from rag.llm import build_judge
from rag.runctx import fmt_meta, snapshot

_DS = "agentic-rag-8types"


def collect(per_type: int) -> list[dict]:
    """8 种题型各 `per_type` 道 —— MuSiQue 的 1/2/3/4hop + MultiHop-RAG 的四类。"""
    rows = []
    for bench, types in (("musique+1hop", ["1hop", "2hop", "3hop", "4hop"]),
                         ("multihoprag", ["comparison_query", "inference_query",
                                          "temporal_query", "null_query"])):
        ex, qtype, _ = load_benchmark(bench)
        for e in sample(ex, qtype, types, per_type):
            t = qtype.get(e.inputs["question"], "?").split("_")[0]
            rows.append({"benchmark": bench, "type": t, "example_id": str(e.id),
                         "question": e.inputs["question"],
                         "reference": e.outputs.get("reference", ""),
                         "reference_contexts": e.outputs.get("reference_contexts") or []})
    return rows


def upsert_dataset(client: Client, rows: list[dict]) -> str:
    """建/更新数据集。**按 example_id 幂等** —— 重复跑不会把同一题灌进去两份。"""
    try:
        ds = client.read_dataset(dataset_name=_DS)
    except Exception:                                              # noqa: BLE001
        ds = client.create_dataset(_DS, description=(
            "8 类题型各 N 道：MuSiQue 1/2/3/4hop（真桥接多跳）+ MultiHop-RAG "
            "comparison/inference/temporal/null。null_query 无 gold 答案，用 refused_ok 评。"))
    have = {(e.metadata or {}).get("example_id") for e in client.list_examples(dataset_id=ds.id)}
    new = [r for r in rows if r["example_id"] not in have]
    if new:
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"question": r["question"]} for r in new],
            outputs=[{"reference": r["reference"],
                      "reference_contexts": r["reference_contexts"]} for r in new],
            metadata=[{"example_id": r["example_id"], "type": r["type"],
                       "benchmark": r["benchmark"]} for r in new])
    print(f"数据集 {_DS}：已有 {len(have)} 题，本次新增 {len(new)} 题，共 {len(have) + len(new)}")
    return _DS


# ── 评估器 ────────────────────────────────────────────────────────────────
_judge = None


def _grade(run, example) -> dict:
    """LLM 裁判 —— 提示词与裁判模型**与 eval_judge.py 完全一致**（见文件头的约束）。

    ⚠️ 两种情况**返回 None 而不是 0**（`None` 会被 LangSmith 排除出均值，`0` 会拉低它）：
      · **run 报错**（网关 429/5xx 打穿重试）—— 那是基础设施失败，不是模型答错。
        本项目栽过三次的同一个坑：**失败必须可见，且不能长得像失败之外的东西。**
      · **题目没有 gold 答案**（`null_query` 不可答题）—— `correct` 对它**在定义上就无意义**，
        硬判必然是 0。首轮 80 题里 10 道 null 全被记 0，把总分从 0.636 拖到 0.556。
        不可答题该看的是 `refused_ok`。
    """
    if run.error:
        return {"results": [{"key": k, "score": None} for k in ("correct", "grounded", "sufficient")]}
    ref = example.outputs or {}
    if not (ref.get("reference") or "").strip():          # null_query：只评 refused_ok
        return {"results": [{"key": k, "score": None} for k in ("correct", "grounded", "sufficient")]}

    global _judge                                                   # noqa: PLW0603
    if _judge is None:
        _judge = build_judge()
    out = run.outputs or {}
    blob, _trunc = _retrieved_blob(out.get("contexts") or [], 150_000)
    prompt = _PROMPT.format(question=(example.inputs or {}).get("question", ""),
                            gold=ref.get("reference", ""),
                            facts="\n".join(f"- {f}" for f in (ref.get("reference_contexts") or [])),
                            answer=(out.get("answer") or "")[:4000],
                            cited="\n".join(f"- {c}" for c in (out.get("cited") or [])) or "(none)",
                            retrieved=blob)
    try:
        m = _JSON.search(_judge.invoke(prompt).content or "")
        d = json.loads(m.group(0)) if m else {}
    except Exception:                                               # noqa: BLE001
        return {"key": "judge_failed", "score": None}               # 失败记 None，**不当 0 分**
    clamp = lambda v: min(1.0, max(0.0, float(v))) if isinstance(v, (int, float)) else None  # noqa: E731
    return {"results": [{"key": k, "score": clamp(d.get(k)),
                         "comment": (d.get("why") or "")[:500] if k == "correct" else None}
                        for k in ("correct", "grounded", "sufficient")]}


def _delivered(run, example) -> dict:
    """**确定性**：gold 证据段 ∩ 检索上下文。零裁判噪声，检索栈好不好看它。"""
    out, ref = run.outputs or {}, example.outputs or {}
    facts = ref.get("reference_contexts") or []
    if run.error or not facts:                            # 报错 / null_query 没有 gold 证据
        return {"key": "delivered", "score": None}
    f = _fact_fate(facts, out.get("contexts") or [], out.get("cited") or [])
    return {"key": "delivered", "score": f["delivered"]}


def _refused_ok(run, example) -> dict:
    """**只对不可答题有意义**：拒答 或 给候选但自曝证据不足，都算对；无免责断言才算错。

    二值的"拒答率"会把「透明地标注证据不足」和「睁眼编」归成一类 —— 前者是诚实行为。
    """
    if run.error or (example.outputs or {}).get("reference_contexts"):
        return {"key": "refused_ok", "score": None}                 # 可答题 / 报错都不评这一项
    stance = answer_stance((run.outputs or {}).get("answer"))
    return {"key": "refused_ok", "score": 0.0 if stance == "asserted" else 1.0,
            "comment": f"stance={stance}"}


def _cost(run, example) -> dict:
    """成本项。**必须与分数同表**（README §5.5：没有成本项的指标必然奖励"塞得更多"）。"""
    if run.error:
        return {"results": [{"key": k, "score": None} for k in ("n_search", "n_ctx_chars")]}
    out = run.outputs or {}
    return {"results": [{"key": "n_search", "score": float(out.get("n_search") or 0)},
                        {"key": "n_ctx_chars", "score": float(
                            sum(len(c) for c in (out.get("contexts") or [])))}]}


def main() -> None:
    ap = argparse.ArgumentParser(description="上传评测集 + 同一个裁判到 LangSmith，跑成 experiment")
    ap.add_argument("--per-type", type=int, default=10, help="数据集里每种题型几道（8 类型 × 它）")
    ap.add_argument("--run-per-type", type=int, default=None,
                    help="**只跑**每类前 N 道（数据集本身不变）。不给则跑整个数据集。"
                         "⚠️ `evaluate(data=数据集名)` 会跑**全部** example，"
                         "所以想跑子集必须显式把挑好的 example 列表传进去。")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--agent", default="react", choices=["react", "planner"])
    ap.add_argument("--concurrency", type=int, default=2,
                    help="**最多 3** —— 再高中转网关会 429（实测并发 3 时 80 题里 7 题被打穿）")
    ap.add_argument("--dataset-only", action="store_true", help="只建/更新数据集，不跑评测")
    args = ap.parse_args()

    client = Client()
    rows = collect(args.per_type)
    ds_name = upsert_dataset(client, rows)
    if args.dataset_only:
        return

    # 要跑哪些 example：整个数据集，还是按题型挑的子集
    data: object = ds_name
    if args.run_per_type:
        ds = client.read_dataset(dataset_name=ds_name)
        by_type: dict[str, list] = {}
        for e in sorted(client.list_examples(dataset_id=ds.id), key=lambda x: str(x.id)):
            by_type.setdefault((e.metadata or {}).get("type", "?"), []).append(e)
        picked = [e for v in by_type.values() for e in v[:args.run_per_type]]
        data = picked
        print(f"只跑子集：{len(by_type)} 类型 × {args.run_per_type} = {len(picked)} 题")

    # 两个语料各建一次检索器（重排器加载贵），按 example 的 benchmark 分派。
    runners = {}
    for bench in ("musique+1hop", "multihoprag"):
        _, _, corpus = load_benchmark(bench)
        runners[bench] = build_runner(build_retriever(corpus, "bge", args.model),
                                      args.model, args.agent)
    by_q = {r["question"]: r for r in rows}

    def target(inputs: dict) -> dict:
        """⚠️ **必须自带退避重试。** 模型网关的 429/5xx 会整题打空，而 `evaluate()` 不重试 ——
        一次限流就废掉一条 example，而且在 LangSmith 上长得像"模型答不出来"。
        退避与 `eval_agentic.execute()` 保持一致（0/30/90 秒）。
        """
        q = inputs["question"]
        bench = by_q.get(q, {}).get("benchmark", "musique+1hop")
        last = None
        for wait in (0, 30, 90):
            if wait:
                time.sleep(wait)
            try:
                out = runners[bench](q)
                return {"answer": out["answer"], "cited": out["cited"],
                        "contexts": out["contexts"], "queries": out.get("queries"),
                        "n_search": out["n_search"], "n_llm_calls": out.get("n_llm_calls")}
            except Exception as e:                       # noqa: BLE001 —— 网关 429/5xx
                last = e
                print(f"  ⚠️ 网关失败({type(e).__name__})，退避重试…", flush=True)
        raise RuntimeError(f"三次重试后仍失败: {type(last).__name__}: {str(last)[:120]}")

    meta = snapshot(benchmark="8types", per_type=args.per_type, answer_model=args.model,
                    agent=args.agent)
    n_run = len(data) if isinstance(data, list) else len(rows)
    print(f"跑 experiment：{n_run} 题，并发 {min(args.concurrency, 3)}\n   {fmt_meta(meta)}")
    res = evaluate(target, data=data,
                   evaluators=[_grade, _delivered, _refused_ok, _cost],
                   experiment_prefix=f"{args.agent}-{args.model}",
                   metadata={k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
                   max_concurrency=min(args.concurrency, 3))
    print(f"\n✅ 完成。到 LangSmith 的 `{ds_name}` 数据集下看这次 experiment。")
    print("   ⚠️ 统计结论仍以 eval_judge.py 的确定性 dump + 同题配对 CI 为准；"
          "这里的价值是**可翻的过程**（每题的 query 序列、检索片段、裁判理由）。")
    try:
        print(f"   {res}")
    except Exception:                                               # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
