# agentic-rag · 过程层优先的 Agentic RAG（可跑 · 可评测）

> 一个**面试可讲、可跑、可评测**的 agentic RAG 最小实现。
> **重点不是「建库」，而是 agentic 过程层**：模型自己决定 **该不该查 / 查几次 / 会不会停 / 值不值（别幻觉）**。
> 检索后端藏在一个协议后面、**可替换**：真实语料（BM25 词法）→ 稠密向量（bge，待做），换后端时上层一行不动。
>
> 这份 README 力求**自包含**：读完它，你应该能了解本项目的每个文件、`data/` 里是什么、怎么跑、评测怎么设计、
> 实测结果如何、以及怎么迁进 SlotFlow。

---

## 一、为什么这么设计（核心动机）

RAG = **检索(retrieve) → 拼上下文(augment) → 生成(generate)**。它有两段：

- 🏗️ **建库/索引**（chunk → embed → 存）—— 离线基础设施，**不是 agent 工具**。
- 🔍 **检索**（`rag_search`）—— **唯一暴露给模型的工具**。

**「agentic」的价值全在检索这一侧的决策链**，不在建库。所以本项目**先把过程层做扎实**，把语料/索引这个重活
**解耦**在一个协议后面：`Retriever` 是个 interface，当前后端是真实语料上的 **`BM25Retriever`**（词法检索），
将来可升级 `VectorRetriever`（chroma + bge）—— **每次只换后端，工具/agent/评测都不动**。

> **为什么用 BM25 词法、而不是一上来就上向量**：词法检索**看词面**，跨跳的措辞差异让单次查询天然「欠覆盖」证据链——
> 正好逼模型改写 query、多跳检索，**过程层行为才可观测、可评**。向量检索一次 top-k 常把分散证据一把糊上来，
> 反而**看不出 agentic**。（这也是本项目早期用一个「故意做笨的关键词玩具库」做冒烟的原因，现已并入真实语料。）

---

## 二、目录结构（每个文件干什么）

```
agentic-rag/
├── README.md            # 本文件（完整说明）
├── requirements.txt     # 依赖：langchain-openai / langgraph / rank-bm25 / python-dotenv / langsmith
├── .env.example         # 配置模板（复制成 .env 填真实值；真 key 别写这里）
├── .env                 # 真实密钥（已 gitignore，不提交；模型 + LangSmith 凭据）
│
│  ── 核心：工具 + 检索接口 + agentic loop ──
├── retriever.py         # Retriever 协议 + Doc/Hit 数据类（检索后端的地基）
├── retriever_bm25.py    # BM25Retriever：真实语料词法检索（实现 Retriever 协议，零重依赖、无 torch）
├── tools.py             # rag_search：唯一暴露给模型的工具（query → 带 [source:] 引用的片段）
├── prompts.py           # 系统提示 = agentic 四条策略（过程层写在这一处，不是硬编码控制流）
├── agent.py             # create_react_agent：model →(rag_search → model)* → stop；含 build_model + .env 自动加载
├── run.py               # CLI：在真实语料上提一个问题，逐步打印每次检索 / 改写 / 停
│
│  ── 语料 + 评测 ──
├── corpus_multihop.py   # 加载 data/ 的 MultiHop-RAG（609 篇 → 6194 片段 + 2556 问）
├── eval_dataset.py      # Example 数据类（评测样例的共享类型）
├── eval_baseline.py     # single_shot：检索一次答一次（非 agentic 对照组）
├── eval_metrics.py      # 确定性指标：correct / faithful / hit / discipline
├── eval_multihop.py     # 真实语料上 agentic vs 单次 + coverage；含 run_agentic + 可 --langsmith 推送
└── data/                # ← MultiHop-RAG 语料，见「第三节 data/ 详解」（已 gitignore）
    ├── corpus.json          # 609 篇新闻全文
    └── MultiHopRAG.json     # 2556 个多跳问题 + gold 证据
```

---

## 三、`data/` 详解（这里到底是什么）

`data/` 放的是 **MultiHop-RAG** —— 一个**多跳 RAG 评测基准**，自带语料 + 问题 + gold 证据。
我们用它证明「在真实、装不下上下文的语料上，agentic 多跳检索确实比单次强」。

- **来源**：HuggingFace 数据集 [`yixuantt/MultiHopRAG`](https://huggingface.co/datasets/yixuantt/MultiHopRAG)（配套论文 *MultiHop-RAG*，2024）。
- **许可**：ODC-BY（开放、允许使用，署名即可）。
- **为什么选它**：① 证据被**故意分散在 2~4 篇文章**里 → 单次检索结构上必然「欠覆盖」→ 正是 agentic 迭代该赢的地方；
  ② **自带语料 + gold 证据**（不用自己去爬维基）；③ 只有 609 篇，**笔记本几分钟建完索引**。
- **是否入库**：`data/` 已在 `.gitignore`，**不提交**（12 MB 语料不该进 git；下面给一键下载命令）。

### 3.1 两个文件

| 文件 | 大小 | 内容 | 顶层结构 |
|---|---|---|---|
| `corpus.json` | ~6.6 MB | **609 篇新闻全文**（被检索的语料） | 一个 JSON 数组，每元素 = 一篇文章 |
| `MultiHopRAG.json` | ~5.0 MB | **2556 个多跳问题** + 每问的 gold 证据 | 一个 JSON 数组，每元素 = 一个问题 |

### 3.2 `corpus.json`（语料 = 被检索对象）

每篇文章字段：

| 字段 | 含义 |
|---|---|
| `title` | 标题 —— **我们用它当「来源 id」**（gold 证据也用 title 对齐，便于算 hit / coverage） |
| `body` | 正文全文（真正被切块检索的内容；长度 min 4770 / 中位 **7836** / max 71034 字符） |
| `source` | 媒体来源（如 Mashable、CNBC、TechCrunch…） |
| `author` / `published_at` / `category` / `url` | 作者 / 发布时间 / 分类 / 原文链接（元数据，本项目未直接用） |

真实样本（截断）：

```json
{
  "title": "200+ of the best deals from Amazon's Cyber Monday sale",
  "author": null,
  "source": "Mashable",
  "published_at": "2023-11-27T08:45:59+00:00",
  "category": "entertainment",
  "url": "https://mashable.com/article/cyber-monday-deals-amazon-2023",
  "body": "Table of Contents ... （全文约数千字）"
}
```

### 3.3 `MultiHopRAG.json`（问题 + gold 证据 = 评测标准答案）

每个问题字段：

| 字段 | 含义 |
|---|---|
| `query` | 多跳问题 |
| `answer` | 标准答案（短，如 `"Sam Bankman-Fried"`；我们用**子串匹配**判 `correct`） |
| `question_type` | 问题类型（见下表；`null_query` = 语料里查不到 → 我们当 negative） |
| `evidence_list` | **gold 证据数组**（2~4 条），每条含 `title`（哪篇文章）+ `fact`（支撑句）等 |

真实样本（截断）：

```json
{
  "query": "Who is the individual associated with the cryptocurrency industry facing a criminal trial ...",
  "answer": "Sam Bankman-Fried",
  "question_type": "inference_query",
  "evidence_list": [
    { "title": "The FTX trial is bigger than Sam Bankman-Fried",
      "fact": "Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto ...",
      "source": "...", "url": "...", "author": "...", "published_at": "...", "category": "..." },
    { "...": "... 共 3 条，分散在 3 篇不同文章 ..." }
  ]
}
```

`question_type` 分布（共 2556）：

| 类型 | 数量 | 含义 | 我们映射成 |
|---|---|---|---|
| `comparison_query` | 856 | 跨文档比较 | `multihop` |
| `inference_query` | 816 | 跨文档推断 | `multihop` |
| `temporal_query` | 583 | 跨文档时序 | `multihop` |
| `null_query` | 301 | **语料里信息不足** | `negative`（应拒答） |

### 3.4 我们怎么用这两个文件（`corpus_multihop.py`）

- `load_corpus()`：读 `corpus.json` → 把每篇 `body` 按 **1200 字符 / 重叠 150** 切块，每块拼上标题 →
  得到 **6194 个片段**（`Doc`，`source = 文章标题`）。
- `load_examples()`：读 `MultiHopRAG.json` → 每个问题转成一条 `Example`：
  `question=query`，`reference=answer`，`sources=evidence 里的所有 title`（= gold 来源），
  `kind = null_query?negative:multihop`。
- 因为**检索片段的 `source` 和 gold 证据都用文章标题**，所以能直接算 `hit@k` 和 `coverage`（多跳覆盖率）。

### 3.5 如何重新下载（`data/` 丢了/换机器时）

```bash
cd agentic-rag && mkdir -p data
curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json      -o data/corpus.json
curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json -o data/MultiHopRAG.json
# 校验（离线，无需模型）：
.venv/bin/python -c "from corpus_multihop import load_corpus, load_examples; print(len(load_corpus()), '片段', len(load_examples()), '例')"   # 应打印 6194 片段 2556 例
```

---

## 四、快速开始

```bash
cd agentic-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # 填 OPENAI_API_KEY / OPENAI_BASE_URL / RAG_MODEL / LANGSMITH_*（真 key 放 .env，别放 .env.example）
# 注：agent.py 启动时自动加载 .env（python-dotenv），无需手动 source。

# 1) 免费离线自检（不调模型）：语料加载 + 分块
python -c "from corpus_multihop import load_corpus, load_examples; print(len(load_corpus()), '片段', len(load_examples()), '例')"

# 2) 跑 agentic 检索，看过程层（需要模型 key）
python run.py "Who is the individual associated with the cryptocurrency industry facing a criminal trial?"

# 3) P3 评测：真实语料 MultiHop-RAG 上 agentic vs 单次（含 coverage）
python eval_multihop.py --n 10                # 本地 delta 表
python eval_multihop.py --n 10 --langsmith    # 同上 + 推 dataset+experiment 到 LangSmith
```

> **网关提示**：某些中转网关 WAF 拦未知 UA → 设 `RAG_USER_AGENT=claude-code/2.1.214`；
> 偶发 5xx/超时 → `RAG_MAX_RETRIES` / `RAG_TIMEOUT`（`build_model` 已读取）。

---

## 五、架构与关键设计

**一句话**：`create_react_agent` 给我 loop，`Retriever` 协议给我可替换后端，`rag_search` 是模型唯一的抓手。

```python
# retriever.py —— 后端只需实现这个协议，其余代码全不动
@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[Hit]: ...

# 两个后端，同一协议：
#   BM25Retriever      (当前, 真实语料词法检索, retriever_bm25.py)
#   VectorRetriever    (待做, chroma + bge)

# agent.py —— 一行 loop；何时查/查几次/何时停由模型在策略下决定，不是写死的控制流
def build_agent(retriever):
    return create_react_agent(build_model(), [make_rag_search(retriever)], prompt=AGENTIC_RAG_SYSTEM)
```

> **为什么用 BM25 词法而非向量**：向量检索一次 top-k 常把分散证据一把糊上来，反而**看不出 agentic**。词法检索
> 的跨跳措辞差异让单次只给部分链，**逼模型改写、多跳**——过程层行为才可观测、可评。

---

## 六、四条 agentic 策略 = 四个评测轴（★ 全篇核心）

`prompts.py` 里的系统提示写了四条策略，一一对应过程层的四个问题、和 `eval_metrics.py` 的指标：

| 过程层问题 | 策略 | 指标 | 说明 |
|---|---|---|---|
| **该不该查** | `1. DECIDE` | `discipline` | 需要知识才查；问候/算术/已知的**直接答**（省 token） |
| **查几次** | `2. ITERATE` | `hit` + `coverage` + `avg_search` | 一次拿不全，**用刚学到的改写 query 再查** |
| **会不会停** | `3. STOP` | `avg_search`（不该膨胀） | 证据够就停；**上限 4 次** |
| **值不值·别幻觉** | `4. GROUND & CITE` | `faithful` + `correct` + `refusal` | 只用检索到的答、标 `[source:]`；查不到就认怂 |

---

## 七、评测设计（怎么把「过程层」量出来）

- **数据集**：真实库 MultiHop-RAG（`corpus_multihop.py`，从 2556 题里等距抽样），`kind`（multihop / negative）决定该考的行为。
- **对照组**：`single_shot`（`eval_baseline.py`）= 检索一次 top-4 → 生成一次。**无迭代、无改写**，用来算 delta。
- **确定性指标**（`eval_metrics.py`，无额外 LLM 成本 → 可复现、免费、离线可跑）：

| 指标 | 怎么算 | 考什么 |
|---|---|---|
| `correct` | 参考答案是否出现在答案里（negative 则 = 正确拒答） | 结果对不对 |
| `faithful` | **引用合法性代理**：cite 的 source ⊆ 实际检索到的 source | 有没有编来源 |
| `hit` | 至少一个 gold source 被检索到（hit@k） | 检索质量 |
| `discipline` | `no_retrieve → 0 次`，否则 `≥1 次` | 该不该查 |
| `coverage` | gold 文档被检索到的**比例** | **多跳覆盖：agentic 的真实增益** |

> 生产会补 **RAGAS / LLM-as-judge** 做语义级忠实度；这里的启发式是它的**低成本、可复现代理**。

---

## 八、实测结果

### 真实语料 MultiHop-RAG（BM25 · grok-4.5 · n=6 试点，0 跳过）

| 指标 | AGENTIC | 单次 | delta |
|---|---|---|---|
| **coverage**（多跳覆盖） | 0.79 | 0.42 | **+0.38** ✅ |
| **hit** | 1.00 | 0.83 | **+0.17** ✅ |
| correct | 0.67 | 0.67 | +0.00 |
| **faithful** | 0.50 | 0.67 | **−0.17** ❌ |
| avg_search | 5.50 | 1.00 | — |

**读法（有赢有输，都讲清）**：
- ✅ **多跳覆盖大赢（+0.38 / hit +0.17）**——证据分散在 2~4 篇文章里，单次 BM25 只捞到 42% 的 gold 文档，agentic 迭代改写捞到 79%。这是单次检索在结构上给不了的信号。
- ❌ **忠实度下滑（−0.17）**——跳得越多、引的来源越多，越容易引到检索集之外，或长标题严格匹配对不上。
- ⚠️ **avg_search=5.5，越过 4 次上限**——明细里 negative 各搜 6 次才拒答、一道多跳搜 8 次；注意 `discipline` 指标只查「该不该搜」、查不到「搜太多」，是 `avg_search` 暴露了模型不守 STOP 上限。
- 🔑 **收口**：**过程层的收益与风险都被模型能力放大**——多跳补覆盖，但忠实度与停机纪律要靠**硬 STOP + 忠实度守卫**兜底；agentic 不是免费午餐。
- 局限：n=6 偏小（4 多跳 + 2 拒答），只看方向；`faithful` 是长标题严格字符串匹配（有度量假象）；`correct` 是子串近似。

> **工程坑**：grok-4.5 在中转网关上多跳单次调用可能 **>120s → Cloudflare 524 超时**；`eval_multihop.py` 因此做了
> 「单题失败只跳过、不毁整轮」+ 模型层 `max_retries`。本次 n=6 试点用 `RAG_TIMEOUT` / `RAG_MAX_RETRIES` 快速失败设置，0 跳过跑通。

---

## 九、怎么迁进 SlotFlow

**只搬核心，不搬脚手架**：

| agentic-rag 的东西 | 进 SlotFlow？ | 说明 |
|---|---|---|
| `rag_search` 工具 + `Retriever`/`Doc`/`Hit` 协议 | ✅ 搬（约几十行 + 改 import） | 注册进 `tool_spaces.py` 一个工具空间 |
| 过程层策略（prompt 四条） | ✅ 搬（下沉成工具 description + 一句 system prompt） | |
| `VectorRetriever` + 真实语料 | ✅ 但要新写/接真语料 | 这才是「让它有用」的主要工作量 |
| `agent.py` / `run.py`（ReAct loop + CLI） | ❌ 不搬 | SlotFlow 自己就是 loop |
| `eval_*` + 真实语料评测台 | ❌ 不搬 | 独立评测台；指标逻辑可借鉴 |

> **为什么先独立做**：隔离环境能**逼出并观测过程层行为**、**干净评测**、**快迭代不碰生产**，并**对着干净接口先探路**——
> 所以最后迁移才机械、低风险。「易迁移」是分割的**回报**，不是矛盾。

---

## 十、路线图

- **过程层（✅）**：`rag_search` 工具 + agentic loop + 四条策略 + LangSmith tracing。
- **真实语料评测（✅）**：`corpus_multihop.py` + `retriever_bm25.py` + `eval_multihop.py`，MultiHop-RAG 上 agentic vs 单次；grok-4.5 · n=6 试点 **coverage +0.38 / hit +0.17**，并暴露 faithful −0.17、avg_search 5.5（越过停机上限）。
- **LangSmith 评测（✅）**：`eval_multihop.py --langsmith` 把自定义指标（含 coverage）包装成 LangSmith evaluator，一键推 dataset + experiment。
- **向量后端（可选）**：`BM25Retriever` → `VectorRetriever`（chroma + bge），同协议、上层不动；再跑 MuSiQue（对抗多跳）加码。

---

## 十一、一句话面试话术

> 「我把 RAG 拆两段：建库是离线基础设施、不是工具；真正 agentic 的是**检索侧的决策**。所以我把**过程层**做扎实——
> 模型自己决定该不该查、多跳改写、什么时候停、查不到就拒答——再用确定性指标 + LangSmith 把这些**量化**（检索次数、
> 忠实度、agentic vs 单次的 delta）。真实语料（MultiHop-RAG）上多跳覆盖 **+0.38 / hit +0.17**，但也实测到多跳会把
> 忠实度带崩、越过停机上限——**收益和风险都被模型能力放大**。检索后端是可替换接口，换语料/换向量时上层零改动。
> **知道什么时候不该检索，比会检索更值钱。**」
