"""查询分解检索器：把多跳问句降到事实层，**每个子问句各自重排**，再轮流取名额。

为什么是这个形态、而不是 `MultiQueryRetriever`（实验 22-23，每型 100 题 + 配对 bootstrap CI）：

| 做法 | Δ fact@16 vs 原问句直查 |
|---|---|
| 同义改写（MultiQueryRetriever 的做法）→ RRF 合并 → **用原问句重排** | −0.012 [−0.031,+0.005] **零** |
| 分解到事实层 → RRF 合并 → **用原问句重排** | −0.009 [−0.026,+0.006] **零** |
| **分解到事实层 → 每个子问句重排自己的候选 → 轮流取** | **+0.050 [+0.014,+0.085]** ✅ |

**决定性的是最后一步，而它恰恰是直觉上会省掉的那一步。** 后两行用的是**完全相同的子问句、
完全相同的候选池**（池覆盖数字一模一样），唯一差别是 top-k 名额怎么分配——纯机制对照
**+0.059 [+0.026,+0.093]**，比"要不要分解"本身还强。

机制：reranker 只是 `(query, doc) → 分数`。**若最后仍拿原问句去打分，子问句捞回来的证据会被
原问句的元层面措辞重新压下去**——绕一圈回到起点。所以重排必须也跟着子问句走。

⚠️ **默认关闭**。代价是每次检索多 1 次 LLM 调用 + N 倍重排开销；而 agent 本身已经在**自发**做
事实层改写（实验21 ③、24），这份收益可能与之重叠——**检索侧证实了不等于端到端拿得到**。
用 `RAG_DECOMPOSE=1` 打开，先测端到端还剩多少再决定要不要设成默认。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time

import numpy as np

from rag.retriever import Doc, Hit

_CACHE = pathlib.Path(__file__).resolve().parents[1] / ".cache" / "decompose.json"

# 与 eval_query.py 的 _DECOMP 完全一致——实验22-23 的数字是在这版提示词上得到的，别随手改。
_PROMPT = (
    "Decompose the following multi-hop question into {n} INDEPENDENT, FACT-SEEKING sub-questions "
    "that can each be answered by a single news article. Do NOT ask about consistency, comparison, "
    "or agreement between reports — instead ask directly about the underlying FACTS each report states "
    "(e.g. 'What did <outlet> report about <entity> on <date>?'). "
    "Output ONLY the {n} sub-questions, one per line, no numbering.\n\nQuestion: {q}"
)


def _lines(text: str, n: int) -> list[str]:
    out = [re.sub(r"^\s*[-*\d.)]+\s*", "", ln).strip() for ln in (text or "").splitlines()]
    return [ln for ln in out if len(ln) > 8][:n]


class DecomposeRetriever:
    """包住一个 `HybridRetriever`，实现同一套 `Retriever` 协议，所以上层一行不动。

    需要内层暴露 `_fuse()` 与 `.reranker`（即 HybridRetriever），因为"逐子问句各自重排"
    必须拿到融合候选、再用**子问句**去打分——这是整个做法的关键，不能只调 `search()`。
    """

    def __init__(self, inner, llm=None, n_sub: int = 3, cache: bool = True) -> None:
        if not (hasattr(inner, "_fuse") and getattr(inner, "reranker", None) is not None):
            raise TypeError("DecomposeRetriever 需要一个带 _fuse() 和 reranker 的 HybridRetriever")
        self.inner, self.n_sub, self._use_cache = inner, n_sub, cache
        if llm is None:
            from rag.llm import build_model
            llm = build_model()
        self.llm = llm
        self._cache: dict[str, list[str]] = {}
        if cache and _CACHE.exists():
            self._cache = json.loads(_CACHE.read_text(encoding="utf-8"))

    def _decompose(self, query: str) -> list[str]:
        """分解失败时**退回原问句**，但把失败**打印出来**——静默降级正是实验12/23 栽过的坑。"""
        key = hashlib.md5(f"{self.n_sub}|{query}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        last = None
        for wait in (0, 5, 20):
            if wait:
                time.sleep(wait)
            try:
                subs = _lines(self.llm.invoke(_PROMPT.format(n=self.n_sub, q=query)).content, self.n_sub)
                if subs:
                    self._cache[key] = subs
                    if self._use_cache:
                        _CACHE.parent.mkdir(parents=True, exist_ok=True)
                        _CACHE.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
                    return subs
                last = ValueError("模型没输出可用的子问句")
            except Exception as e:      # noqa: BLE001 —— 网关 5xx/429/超时
                last = e
        print(f"  ⚠️ 查询分解失败，本次退回原问句直查：{type(last).__name__}: {str(last)[:80]}", flush=True)
        return [query]

    def _rank(self, sub: str) -> list[Doc]:
        """用**子问句**去融合、也用**子问句**去重排 —— 后半句才是有效的那一半。"""
        cands = self.inner._fuse(sub)
        if not cands:
            return []
        sc = np.asarray(self.inner.reranker.predict([(sub, d.text) for d in cands]))
        return [cands[i] for i in np.argsort(-sc)]

    def search(self, query: str, k: int = 4) -> list[Hit]:
        subs = self._decompose(query)
        if subs == [query]:                       # 分解失败 → 完全等价于原来的行为
            return self.inner.search(query, k=k)
        per = [self._rank(s) for s in subs]
        out, seen = [], set()
        # 轮流取：每个子问句的第 1 名、第 2 名……名额在子问句之间**均分**，
        # 而不是让某一个子问句的长尾把别人的证据挤出 top-k。
        for rank in range(max((len(p) for p in per), default=0)):
            for p in per:
                if rank < len(p) and p[rank].id not in seen:
                    seen.add(p[rank].id)
                    out.append(p[rank])
                    if len(out) >= k:
                        return [Hit(doc=d, score=float(len(out) - i)) for i, d in enumerate(out)]
        return [Hit(doc=d, score=float(len(out) - i)) for i, d in enumerate(out)]


def maybe_wrap(retriever, llm=None):
    """`RAG_DECOMPOSE=1` 时把 retriever 包上分解层，否则**原样返回**（默认关闭）。"""
    if os.environ.get("RAG_DECOMPOSE", "").strip() not in ("1", "true", "yes"):
        return retriever
    n = int(os.environ.get("RAG_DECOMPOSE_N", 3))
    print(f"[decompose] 已启用查询分解检索（{n} 个子问句，各自重排后轮流取名额）", flush=True)
    return DecomposeRetriever(retriever, llm=llm, n_sub=n)
