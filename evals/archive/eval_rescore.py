"""用**新裁判**给 LangSmith 上**已存的 run** 重新打分——不重跑 agent，只重打分（省钱、也让旧实验可比）。

为什么需要它：裁判模型会消失（本项目实测：网关某天起不再提供 grok-4.5，而实验1-12 全是它判的）。
一旦换裁判，新旧分数就**不可比**——除非把旧 run 用**同一个新裁判**重打一遍。本脚本干的就是这件事：
run 的输出（answer / contexts）都存在 LangSmith 里，重打分不需要再跑一次 agent。

    python eval_rescore.py --judge deepseek-reasoner \
        --exp agentic-on-multihop-rag-241732c0 \
        --exp agentic-bge-deepseek-v4-pro-on-multihop-rag-eecdbb93

⚠️ 只重打 **LLM-judge** 四轴；确定性指标（context_recall_fact / refused）不依赖裁判，原样可比。
⚠️ 空产出的 run（网关故障）**排除**，不当 0 分算（教训见 EXPERIMENTS 实验12）。
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))

import argparse
import json
import pathlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from langsmith import Client

from evals.archive.eval_common import SPECS, call_judge, make_judges
from rag.llm import build_judge

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
_AXES = ["correctness", "groundedness", "retrieval_relevance", "helpfulness"]


def _type_map() -> dict[str, str]:
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {r["query"]: r.get("question_type", "?") for r in raw}


def _gold(client: Client, dataset: str) -> dict[str, str]:
    """question -> gold answer（correctness 需要参考答案；run 里不存 gold）。"""
    return {e.inputs.get("question"): (e.outputs or {}).get("reference", "")
            for e in client.list_examples(dataset_name=dataset)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, help="新裁判模型名（对所有 --exp 用同一个，才可比）")
    ap.add_argument("--exp", action="append", required=True, help="LangSmith experiment 名，可给多个")
    ap.add_argument("--dataset", default="multihop-rag")
    ap.add_argument("--axes", default="correctness", help=f"重打哪些轴，逗号分隔（可选：{','.join(_AXES)}）")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip() in SPECS]
    qtype, client = _type_map(), Client()
    gold = _gold(client, args.dataset)
    judges = make_judges(build_judge(args.judge))
    print(f"裁判 = {args.judge}｜重打轴 = {', '.join(axes)}\n")

    for exp in args.exp:
        rows = [r for r in client.list_runs(project_name=exp, is_root=True)
                if ((r.outputs or {}).get("answer") or "").strip()]      # 排除空产出
        total = len(list(client.list_runs(project_name=exp, is_root=True)))

        def score(r):
            q = (r.inputs or {}).get("question", "")
            out, ref = r.outputs or {}, gold.get(q, "")
            got = {}
            for a in axes:
                if a == "correctness" and not ref:      # null_query 无 gold 答案 → 跳过
                    continue
                try:
                    got[a] = call_judge(a, judges[a], q, out, ref)["score"]
                except Exception as e:                  # 单题失败不毁整轮，但**记为缺失**而不是 0
                    print(f"  ⚠️ {a} 打分失败：{type(e).__name__}: {str(e)[:70]}")
            return qtype.get(q, "?").split("_")[0], got

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            results = list(ex.map(score, rows))

        agg: dict = defaultdict(lambda: defaultdict(list))
        for t, got in results:
            for a, s in got.items():
                if s is not None:
                    agg[t][a].append(s)

        print(f"== {exp} ==（有效 {len(rows)}/{total} 题，空产出已排除）")
        print(f"{'type':12s} " + " ".join(f"{a[:11]:>12}" for a in axes) + f"{'n':>5}")
        for t in ("comparison", "inference", "temporal", "null"):
            d = agg.get(t, {})
            n = max((len(v) for v in d.values()), default=0)
            print(f"{t:12s} " + " ".join(
                f"{sum(d[a]) / len(d[a]):>12.2f}" if d.get(a) else f"{'-':>12}" for a in axes) + f"{n:>5}")
        print()


if __name__ == "__main__":
    main()
