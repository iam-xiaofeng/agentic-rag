"""检索后端的协议与数据类型。

agent 只依赖 `Retriever` 协议，后端可随意替换而上层（agent.py / tools.py / eval_*）不动：
- BM25Retriever（retriever_bm25.py）：真实语料上的词法检索（当前使用）。
- VectorRetriever（待做）：chroma + bge 稠密检索，同一协议直接替换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
