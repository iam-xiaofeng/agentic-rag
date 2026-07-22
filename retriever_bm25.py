"""真实语料上的 BM25 检索器（P3）—— 与 InMemoryRetriever 同一套 `Retriever` 协议。

因为它实现的是完全相同的 `search(query, k) -> list[Hit]` 接口，agent.py / tools.py /
eval_* 全都**不用改**，只换后端。这正是「把检索藏在协议后面」的意义。

为什么第一次真实跑先用 BM25（词法）而不是向量：
  - 零重依赖（不用 torch、不用下模型）-> 几秒钟就能起来；
  - 它是一等的检索方式（检索 != embedding，见 项目/07）；
  - 正因为是词法匹配，跨跳的词面差异会让单次查询「欠覆盖」证据链 —— 这恰恰是 agentic
    改写该赢的地方。
稠密的 `VectorRetriever`（chroma + bge）可在同一协议下直接替换升级。
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from retriever import Doc, Hit

_WORD = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _WORD.findall((s or "").lower())


class BM25Retriever:
    """在一批片段上做 Okapi BM25。建索引是 O(N)，几千条片段瞬间完成。"""

    def __init__(self, docs: list[Doc]) -> None:
        self.docs = docs
        self._bm25 = BM25Okapi([_tok(d.text) for d in docs])

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores = self._bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Hit(doc=self.docs[i], score=float(scores[i])) for i in order if scores[i] > 0]
