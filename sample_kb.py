"""一个极小、自包含的知识库（虚构的 "Aurora Labs"）。

这是**占位语料**。它唯一的作用，是让 agentic 过程层在**没有真实索引**的情况下也能被测出来，
故意埋了三样东西：

- 多跳链    -> 答案拆在 2+ 篇文档里（逼模型改写 + 跳）
- 干扰文档  -> 无关但词面重叠（测噪声鲁棒）
- 故意留空  -> 有些问题这里根本没答案（测拒答 / negative rejection）

将来搭真实项目文件夹时，删掉本文件、把 VectorRetriever 指向真实语料即可 —— agent/工具代码不变。
"""

# （id, text, source）
SAMPLE_DOCS: list[tuple[str, str, str]] = [
    ("handbook",
     "Aurora Labs' flagship product is Nimbus, a real-time analytics platform. "
     "Nimbus is built and maintained by the Stream team.",
     "handbook.md"),
    ("nimbus-arch",
     "Nimbus stores incoming events in the Quartz database and caches hot query "
     "results in Redis.",
     "nimbus/architecture.md"),
    ("quartz",
     "Quartz is a columnar database written in Rust. It supports vector similarity "
     "search and was open-sourced in 2024.",
     "quartz/overview.md"),
    ("stream-team",
     "The Stream team is led by Mia Chen and is based in the Berlin office.",
     "teams/stream.md"),
    ("berlin",
     "The Berlin office opened in 2019 and focuses on data infrastructure.",
     "offices/berlin.md"),
    ("redis-note",
     "Redis at Aurora Labs runs in cluster mode with six shards.",
     "nimbus/caching.md"),
    ("perk",  # 干扰项：在 "Aurora Labs / office" 上词面重叠，却答不出任何有用信息
     "Aurora Labs offers a coffee perk; the office espresso machine is a La Marzocco Linea.",
     "handbook.md"),
]

# 探针问题（见 README），以及各自考察的过程层行为：
#   多跳    : "What language is the database used by Nimbus written in?"  (nimbus-arch -> quartz)
#   多跳    : "Which office is the team behind Nimbus based in?"          (handbook -> stream-team -> berlin)
#   不检索  : "What is 15 + 27?"                                          (直接答，别检索)
#   拒答    : "What language is Nimbus itself written in?"               (库里没有 -> 说不知道)
#   干扰    : "What database does Nimbus use?"                            (无视 espresso 那篇)
