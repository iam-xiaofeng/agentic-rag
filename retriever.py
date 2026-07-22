"""agentic RAG demo 的检索后端。

agent 只依赖 `Retriever` 协议，所以后端可以随意替换：

- InMemoryRetriever（本文件）：在样本库上做关键词检索。零依赖、现在就能跑。
- VectorRetriever（P3，待办）：真实语料上的 chroma + bge 向量检索，**同一套接口** →
  直接替换，tools.py / agent.py 一行都不用改。
- BM25Retriever（P3，见 retriever_bm25.py）：真实语料上的词法检索，同样实现本协议。

P1 的重点是 **agentic 过程层**（该不该查 / 查几次 / 会不会停 / 别幻觉），不是建库。
这里的检索器故意做得「笨」，好逼着 agent 自己多做功。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sample_kb import SAMPLE_DOCS


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    source: str


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[Hit]:
        ...


# 一个小停用词表，免得 "what / is / the ..." 主导了词面重叠得分。
_STOP = {
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "in", "on", "at", "by", "to", "and", "or", "for", "with",
    "does", "do", "did", "used", "using", "use", "that", "this",
    "it", "its", "they", "their", "as", "from", "into",
}


def _tokens(text: str) -> set[str]:
    return {w.strip(".,?!()[]{}'\":;").lower() for w in text.split()} - _STOP - {""}


class InMemoryRetriever:
    """在一个内存小知识库上做关键词重叠检索。

    故意做得「笨」（不上向量）：单次查询通常只能返回部分答案，从而逼着 agent 改写查询、
    做多跳检索 —— 这正是我们想观察、想评测的 agentic 行为。
    """

    def __init__(self, docs: list[Doc] | None = None) -> None:
        self.docs = docs or [Doc(*d) for d in SAMPLE_DOCS]

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = _tokens(query)
        hits = [
            Hit(doc=d, score=float(len(q & _tokens(d.text))))
            for d in self.docs
        ]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


if __name__ == "__main__":  # 快速手验，无外部依赖
    r = InMemoryRetriever()
    for probe in ["Nimbus database", "Quartz language", "Nimbus source language"]:
        print(f"\nQ: {probe}")
        for h in r.search(probe):
            print(f"  {h.score:.0f}  [{h.doc.source}] {h.doc.text[:60]}...")
