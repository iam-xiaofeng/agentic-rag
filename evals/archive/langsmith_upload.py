"""把 MultiHop-RAG 全量传成 LangSmith 数据集，每条附 gold 证据（evidence 的 fact 摘录）。

    python langsmith_upload.py                  # 传全部 2556 题（2255 多跳 + 301 拒答负例）
    python langsmith_upload.py --name my-ds     # 指定数据集名
    python langsmith_upload.py --n 200          # 只传前 200（等距抽样）

每条 example：
  inputs  = {question}
  outputs = {reference(标准答案), kind, gold_titles, reference_contexts(evidence 的 fact 摘录)}
拒答负例（null_query）：reference="" 且 gold 为空 —— 用来判「该拒答时拒不拒答」。

纯数据上传，不调模型、不花钱；只需 LANGSMITH_API_KEY（llm.py 自动读 .env）。
有了 reference_contexts（原文里的支撑句），就能在 LangSmith 上对着 ground truth 判忠实度 / context recall。
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import json
import pathlib

import rag.llm as llm  # noqa: F401  —— 触发 .env 加载（LANGSMITH_API_KEY）

from langsmith import Client

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def _rows() -> list[dict]:
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    out = []
    for r in raw:
        ev = r.get("evidence_list") or []
        kind = "negative" if r.get("question_type") == "null_query" else "multihop"
        out.append({
            "question": r["query"],
            "reference": "" if kind == "negative" else (r.get("answer") or "").strip(),
            "kind": kind,
            "gold_titles": sorted({e.get("title", "").strip() for e in ev if e.get("title")}),
            "reference_contexts": [e.get("fact", "").strip() for e in ev if e.get("fact")],
        })
    return out


def _pick(rows: list[dict], n: int) -> list[dict]:
    if n <= 0 or n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="multihop-rag", help="LangSmith 数据集名")
    ap.add_argument("--n", type=int, default=0, help="只传前 n 题（等距抽样）；0 = 全部")
    args = ap.parse_args()

    rows = _pick(_rows(), args.n)
    client = Client()
    if client.has_dataset(dataset_name=args.name):
        print(f"数据集 {args.name!r} 已存在；换个 --name，或先在 LangSmith 上删掉再传。")
        return

    ds = client.create_dataset(
        args.name,
        description="MultiHop-RAG · question + gold answer + gold evidence(fact) + gold titles · kind=multihop/negative",
    )
    batch = 100
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"question": r["question"]} for r in chunk],
            outputs=[{
                "reference": r["reference"],
                "kind": r["kind"],
                "gold_titles": r["gold_titles"],
                "reference_contexts": r["reference_contexts"],
            } for r in chunk],
        )
        print(f"  上传 {min(i + batch, len(rows))}/{len(rows)}")
    print(f"[langsmith] 数据集 {args.name!r} 已上传 {len(rows)} 题（含 gold 证据 fact）。")


if __name__ == "__main__":
    main()
