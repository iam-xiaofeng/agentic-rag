"""查询侧诊断：**问句怎么写**比**用什么 reranker** 更决定召回吗？（确定性 + 几乎免费）

EXPERIMENTS 实验11-12 把检索侧走到了头：候选池覆盖能补到 100%、换最强的 instruction-aware reranker
（Qwen3-4B）也只能把 temporal 的交付从 0.42 抬到 0.53——**证据在池子里，就是排不出来**。
原因是 reranker 无论多强都只是 `(query, doc) → 分数`：**query 里没有的信号，打分器变不出来**。
temporal 的问句是**元层面追问**（"A 在 X 日的报道与 B 在 Y 日的报道是否一致？"），
gold 证据却是**具体事实陈述**，两者不共享词汇也不共享语义焦点。

所以本脚本把变量换成**问句本身**，对比四种策略（检索器/reranker 全程不变，唯一变量 = 用什么 query 去查）：

  1. baseline        原问句直接查
  2. multiquery      LLM 生成 N 个**同义改写**（LangChain MultiQueryRetriever 的做法）→ 各自召回 → RRF 合并
  3. decompose       LLM **分解**成事实性子问句（"A 在 X 日报道了什么"）→ 各自召回 → RRF 合并 → 用原问句重排
  4. decompose-each  同上分解，但**每个子问句各自重排**取名额再拼（真正的"分而治之"，不让原问句的
                     元层面措辞再把证据压回去）

3 vs 4 是关键对照：如果 3 不涨而 4 涨，说明**光换召回不够，重排也必须跟着子问句走**。
2 vs 3 回答"同义改写够不够，还是必须降到事实层"。

    python eval_query.py --n 15 --types temporal
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

from corpus_multihop import load_corpus
from llm import build_model
from retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parent / "data"
_KS = (4, 8, 16)

_MULTI = ("Rewrite the following question in {n} different ways to maximize retrieval coverage. "
          "Keep the same meaning; vary wording and phrasing. "
          "Output ONLY the {n} rewrites, one per line, no numbering.\n\nQuestion: {q}")
_DECOMP = ("Decompose the following multi-hop question into {n} INDEPENDENT, FACT-SEEKING sub-questions "
           "that can each be answered by a single news article. Do NOT ask about consistency, comparison, "
           "or agreement between reports — instead ask directly about the underlying FACTS each report states "
           "(e.g. 'What did <outlet> report about <entity> on <date>?'). "
           "Output ONLY the {n} sub-questions, one per line, no numbering.\n\nQuestion: {q}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _lines(text: str, n: int) -> list[str]:
    out = [re.sub(r"^\s*[-*\d.)]+\s*", "", ln).strip() for ln in (text or "").splitlines()]
    return [ln for ln in out if len(ln) > 8][:n]


def _rrf(lists: list[list], k: int = 60) -> list:
    """把多个子问句各自的候选列表按 RRF 合并（只看排名，跨问句可比）。"""
    fused: dict[str, list] = {}
    for lst in lists:
        for rank, d in enumerate(lst, start=1):
            slot = fused.setdefault(d.id, [d, 0.0])
            slot[1] += 1.0 / (k + rank)
    return [d for d, _ in sorted(fused.values(), key=lambda v: v[1], reverse=True)]


def _rerank(retr, query: str, docs: list) -> list:
    if not docs:
        return []
    sc = np.asarray(retr.reranker.predict([(query, d.text) for d in docs]))
    return [docs[i] for i in np.argsort(-sc)]


def _score(docs: list, facts: list[str], ks=_KS) -> dict:
    out = {}
    for k in ks:
        blob = _norm(" ".join(d.text for d in docs[:k]))
        out[f"@{k}"] = sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts)
    blob = _norm(" ".join(d.text for d in docs))
    out["pool"] = sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="每种题型抽几题")
    ap.add_argument("--types", default="temporal", help="题型，逗号分隔（comparison/inference/temporal）")
    ap.add_argument("--sub", type=int, default=3, help="生成几个改写/子问句")
    ap.add_argument("--reranker", default="bge", choices=["bge", "qwen"],
                    help="重排器；用 qwen 可测『查询分解 × 强 reranker』能否叠加（实验15：两者修的是同一环的不同侧面）")
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    qs = []
    for t in types:
        picked = [r for r in rows if r.get("question_type", "").startswith(t)
                  and [e for e in (r.get("evidence_list") or []) if e.get("fact")]]
        for r in picked[: args.n]:
            qs.append((t, r["query"], [e["fact"] for e in r["evidence_list"] if e.get("fact")]))

    if args.reranker == "qwen":
        from reranker_qwen import QwenReranker
        retr = HybridRetriever(load_corpus(), reranker=QwenReranker())
    else:
        retr = HybridRetriever(load_corpus())
    llm = build_model()
    agg: dict[str, dict[str, list]] = {}

    for i, (t, q, facts) in enumerate(qs, 1):
        try:
            multi = _lines(llm.invoke(_MULTI.format(n=args.sub, q=q)).content, args.sub) or [q]
            decomp = _lines(llm.invoke(_DECOMP.format(n=args.sub, q=q)).content, args.sub) or [q]
        except Exception as e:
            print(f"  ⚠️ 第{i}题改写失败，跳过：{type(e).__name__}: {str(e)[:60]}")
            continue

        strategies = {
            "1 baseline": _rerank(retr, q, retr._fuse(q)),
            "2 multiquery": _rerank(retr, q, _rrf([retr._fuse(s) for s in multi])),
            "3 decompose": _rerank(retr, q, _rrf([retr._fuse(s) for s in decomp])),
        }
        # 4: 每个子问句各自重排，按名额轮流取（不让原问句的元层面措辞把证据压回去）
        per = [_rerank(retr, s, retr._fuse(s)) for s in decomp]
        inter, seen = [], set()
        for rank in range(max((len(p) for p in per), default=0)):
            for p in per:
                if rank < len(p) and p[rank].id not in seen:
                    seen.add(p[rank].id)
                    inter.append(p[rank])
        strategies["4 decompose-each"] = inter

        for name, docs in strategies.items():
            slot = agg.setdefault(name, {})
            for k, v in _score(docs, facts).items():
                slot.setdefault(k, []).append(v)
        if i == 1:
            print(f"[样例] 原问句：{q[:100]}\n  改写→ {multi[0][:90]}\n  分解→ {decomp[0][:90]}\n")
        print(f"  {i}/{len(qs)} done", flush=True)

    cols = [f"@{k}" for k in _KS] + ["pool"]
    n = len(next(iter(agg.values()))["@8"]) if agg else 0
    print(f"\n== fact 级召回 · 按**查询策略** · {'+'.join(types)} · n={n} · 子问句 {args.sub} 个 · reranker={args.reranker} ==")
    print(f"{'策略':18s} " + " ".join(f"{c:>8}" for c in cols))
    for name in ("1 baseline", "2 multiquery", "3 decompose", "4 decompose-each"):
        d = agg.get(name)
        if d:
            print(f"{name:18s} " + " ".join(f"{sum(d[c]) / len(d[c]):>8.3f}" for c in cols))


if __name__ == "__main__":
    main()
