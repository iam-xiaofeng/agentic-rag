"""评测用的共享数据类型：一条带参考答案 + 期望行为的样例。

`kind` 决定该给哪种过程层行为打分：
  multihop  -> 答案分散在多篇文档（应检索，最好多跳）
  negative  -> 语料里查不到（应拒答，别编）

真实语料的样例由 rag.corpus_multihop.load_examples() 从 MultiHop-RAG 产出。
（原在 evals/eval_dataset.py —— 但 rag.corpus_multihop 依赖它，等于**核心依赖评测**；
移进 rag/ 后依赖方向恢复单向：evals → rag。）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    question: str
    reference: str          # 答案里必须出现的关键事实（negative 为 ""）
    sources: list[str]      # gold 来源文档（用于 hit / coverage；negative 为 []）
    kind: str               # multihop | negative
