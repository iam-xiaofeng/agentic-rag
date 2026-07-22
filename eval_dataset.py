"""评测数据集：带参考答案 + 期望行为的探针问题。

`kind` 决定我们给哪种过程层行为打分：
  multihop     -> 事实题，答案拆在多篇文档里   （应检索，最好 >1 次）
  single       -> 事实题，单篇文档就有          （应检索 >=1 次）
  no_retrieve  -> 不靠知识库就能答             （不该检索）
  negative     -> 知识库里根本没有            （应拒答，别编）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    question: str
    reference: str          # 答案里必须出现的关键事实（negative 为 ""）
    sources: list[str]      # gold 来源文档（用于 retrieval-hit；没有则为 []）
    kind: str               # multihop | single | no_retrieve | negative


DATASET: list[Example] = [
    Example("What language is the database used by Nimbus written in?", "Rust",
            ["nimbus/architecture.md", "quartz/overview.md"], "multihop"),
    Example("Which office is the team behind Nimbus based in?", "Berlin",
            ["handbook.md", "teams/stream.md"], "multihop"),
    Example("What database does Nimbus use?", "Quartz",
            ["nimbus/architecture.md"], "single"),
    Example("When did the Berlin office open?", "2019",
            ["offices/berlin.md"], "single"),
    Example("Who leads the Stream team?", "Mia Chen",
            ["teams/stream.md"], "single"),
    Example("What is 15 + 27?", "42", [], "no_retrieve"),
    Example("What language is Nimbus itself written in?", "", [], "negative"),
    Example("What is the capital of France?", "", [], "negative"),
]
