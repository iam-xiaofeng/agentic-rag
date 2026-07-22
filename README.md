# 混合检索 RAG 流水线 · BM25 + bge 向量 + 交叉编码重排 · LLM-as-judge 评测

> 一个**面试可讲、可跑、可评测**的生产级 RAG **检索流水线**：
> **召回**（BM25 词法 + bge 稠密向量，加权融合）→ **重排**（bge-reranker 交叉编码）→ **生成**（grounded、带引用）
> → **评测**（openevals 的 LLM-as-judge，可一键推 LangSmith）。
>
> 重点是**检索质量**：单一检索方式都有盲区，词法看词面、向量看语义，**混合 + 重排**是业界最扎实的组合。
> 检索后端全部藏在一个 `Retriever` 协议后面、**可替换**，换后端时上层（生成 / 评测）一行不动。

> **项目沿革**：本仓库早期是一个 **agentic RAG**（模型自己决定该不该查 / 查几次 / 何时停）+「agentic vs 单次」对比实验，
> 已完整冻结在 git tag **`v1-agentic-comparison`**（`git checkout v1-agentic-comparison` 可复现）。当前 `master` 聚焦
> **检索流水线**方向。

---

## 一、为什么这么设计（核心动机）

RAG = **检索(retrieve) → 拼上下文(augment) → 生成(generate)**。**答案质量的上限，几乎全由检索决定**——
检索不到，再强的模型也只能幻觉。所以本项目把功夫下在检索侧，用业界公认最稳的三段式：

1. 🔤 **词法召回（BM25）**——看**词面重叠**。命中精确实体 / 稀有词很强，但换个说法就抓瞎。
2. 🧠 **向量召回（bge dense）**——看**语义相近**。措辞不同也能召回，但对精确实体 / 数字不敏感。
3. 🎯 **交叉编码重排（bge-reranker）**——把 (query, passage) **一起**读，算一个比双塔向量更准的相关性分。贵，所以只重排少量候选。

> **为什么两路召回要融合、还要重排**：词法和向量的盲区**互补**——加权融合先把两种「看法」的候选并起来（提召回），
> 再用 cross-encoder 精排（提精度）。这就是 “hybrid retrieval + reranking” 成为生产标配的原因。

检索后端解耦在 `Retriever` 协议后面：当前实现 `BM25Retriever` / `DenseRetriever` / `HybridRetriever` 三个后端，
**同一套接口**，`rag.py`（生成）和 `eval_rag.py`（评测）对换后端无感。

---

## 二、目录结构（每个文件干什么）

```
agentic-rag/
├── README.md            # 本文件
├── requirements.txt     # 依赖：sentence-transformers / rank-bm25 / openevals / langchain-openai / langsmith
├── .env.example         # 配置模板（复制成 .env 填真实值）
├── .env                 # 真实密钥（已 gitignore；模型 + LangSmith 凭据）
│
│  ── 检索栈（核心）：一个协议 + 三个后端 ──
├── retriever.py         # Retriever 协议 + Doc/Hit 数据类（后端地基）
├── retriever_bm25.py    # BM25Retriever：词法检索（rank_bm25，零重依赖）
├── retriever_dense.py   # DenseRetriever：bge 向量检索（sentence-transformers，向量缓存到 .cache/）
├── retriever_hybrid.py  # HybridRetriever：BM25+向量 加权融合 → bge-reranker 重排（★ 新核心）
│
│  ── 生成 + 运行 ──
├── llm.py               # build_model：OpenAI 兼容 chat 模型（grok/glm/…）+ .env 自动加载
├── rag.py               # answer：检索 → 一次 grounded 生成（带 [source:] 引用）
├── run.py               # CLI：对一个问题跑 hybrid 检索（可选生成），打印排名
│
│  ── 语料 + 评测 ──
├── corpus_multihop.py   # 加载 data/ 的 MultiHop-RAG（609 篇 → 6194 片段 + 2556 问）
├── eval_dataset.py      # Example 数据类（评测样例的共享类型）
├── eval_rag.py          # openevals 4 个 LLM-judge + 可 --upload 推 LangSmith
└── data/                # ← MultiHop-RAG 语料（已 gitignore；见「第三节」）
    ├── corpus.json          # 609 篇新闻全文
    └── MultiHopRAG.json     # 2556 个多跳问题 + gold 证据
```

---

## 三、`data/` 详解（这里是什么）

`data/` 放的是 **MultiHop-RAG** —— 一个**多跳 RAG 评测基准**，自带语料 + 问题 + gold 证据。

- **来源**：HuggingFace [`yixuantt/MultiHopRAG`](https://huggingface.co/datasets/yixuantt/MultiHopRAG)（论文 *MultiHop-RAG*, 2024）；许可 **ODC-BY**（公开、署名即可）。
- **为什么选它**：证据被**故意分散在 2~4 篇文章**里 → 对检索的召回是真实压力测试；自带 gold 证据；只有 609 篇，笔记本几分钟建完索引。
- **两个文件**：
  - `corpus.json`：609 篇新闻（`title` 标题 / `body` 正文 / `source` 媒体 / …）——**被检索的语料**。
  - `MultiHopRAG.json`：2556 个问题（`query` / `answer` 标准答案 / `question_type` / `evidence_list` gold 证据）——**评测集**。
- **我们怎么用**（`corpus_multihop.py`）：
  - `load_corpus()`：每篇 `body` 按 **1200 字符 / 重叠 150** 切块、拼上标题 → **6194 个 `Doc`**（`source = 文章标题`，与 gold 证据标题对齐）。
  - `load_examples()`：每个问题 → 一条 `Example`（`question` / `reference=answer` / `sources=gold 标题` / `kind`）。

### 如何重新下载

```bash
cd agentic-rag && mkdir -p data
curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json      -o data/corpus.json
curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json -o data/MultiHopRAG.json
# 离线自检（无需模型）：
.venv/bin/python -c "from corpus_multihop import load_corpus, load_examples; print(len(load_corpus()), '片段', len(load_examples()), '例')"  # 6194 片段 2556 例
```

---

## 四、快速开始

```bash
cd agentic-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 含 sentence-transformers（会拉 torch）；首次跑会自动下 bge 模型
cp .env.example .env                    # 填 OPENAI_API_KEY / OPENAI_BASE_URL / RAG_MODEL / LANGSMITH_*（真 key 放 .env）

# 1) 免费离线自检（不调模型）：语料加载 + 分块
python -c "from corpus_multihop import load_corpus, load_examples; print(len(load_corpus()), '片段', len(load_examples()), '例')"

# 2) 跑 hybrid 检索（首次会下 bge 模型 + 编码 6194 片段，之后走 .cache 缓存）
python run.py "Who is the individual associated with the cryptocurrency industry facing a criminal trial?"
python run.py --no-gen "..."           # 只看检索排名，不调生成模型

# 3) LLM-judge 评测（openevals，judge=模型；--upload 另推 LangSmith）
python eval_rag.py --n 8                # 本地打分表
python eval_rag.py --n 8 --upload       # 同上 + 推 dataset + experiment 到 LangSmith
```

> **网关提示**：某些中转网关 WAF 拦未知 UA → 设 `RAG_USER_AGENT`；偶发 5xx/超时 → `RAG_MAX_RETRIES` / `RAG_TIMEOUT`（`build_model` 已读取）。
> `eval_rag.py` 单题失败会跳过、不毁整轮。

---

## 五、架构与关键设计

**一句话**：`Retriever` 协议给我可替换后端；hybrid 把词法/向量的盲区互补起来；reranker 精排；生成只用检索到的内容并标引用。

```python
# retriever.py —— 后端只需实现这个协议，其余代码全不动
@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[Hit]: ...

# 三个后端，同一协议：
#   BM25Retriever     词法（rank_bm25）
#   DenseRetriever    bge-small-en-v1.5 向量，归一化余弦
#   HybridRetriever   ↑两者加权融合 → bge-reranker-base 重排
```

**HybridRetriever 的三步**（`retriever_hybrid.py`）：

1. **召回**：BM25、Dense 各取 top-`pool`(默认 20)。
2. **融合**：各自分数 **min-max 归一化后按权重相加**（默认 `w_bm25 = w_dense = 0.5`）；同一片段两边都命中则叠加。
3. **重排**：融合候选池丢给 `bge-reranker-base`（cross-encoder），按 (query, passage) 相关性取 top-k。

> 想调：`HybridRetriever(docs, w_bm25=.3, w_dense=.7, pool=30)`；升级模型改 `retriever_dense.py` 的 `bge-base-en-v1.5` / `retriever_hybrid.py` 的 `bge-reranker-v2-m3` 即可，上层不动。

---

## 六、评测设计（openevals · LLM-as-judge · LangSmith）

不再用自研启发式指标，而是 **openevals**（LangChain 官方开源评估器）的现成 **LLM-as-judge**——语义级、贴近生产。
裁判 LLM 走你配置的网关（如 grok），四个轴一一对应 RAG 的关键问题：

| 指标（openevals prompt） | 考什么 | 需要 |
|---|---|---|
| `correctness` | 答案对不对（vs 参考答案） | question + answer + reference |
| `groundedness` | 答案是否只由检索到的上下文支撑（**忠实度 / 不幻觉**） | answer + retrieved context |
| `retrieval_relevance` | 检索到的上下文与问题相关吗（**检索质量**） | question + retrieved context |
| `helpfulness` | 答案是否真正回应了问题 | question + answer |

- 每个指标是 `[0,1]` 连续分（`continuous=True`），裁判同时给出**打分理由**（comment）。
- `eval_rag.py --upload`：把抽样上传成 **LangSmith dataset**，用 `evaluate()` 跑 target（= 检索流水线）+ 上面四个评估器，
  结果作为 **experiment** 落在 LangSmith 里（可视化、可对比多次实验）。
- **成本**：LLM-judge 每题 = 1 次生成 + 4 次裁判调用，会花 token，先用小 `--n` 试。

> 对照：自研确定性指标（子串匹配、集合包含）**免费、可复现**但只是语义的粗代理；LLM-judge **语义级但花钱、不完全可复现**。
> 早期 agentic 版本用的是前者（见 tag `v1-agentic-comparison`），这一版换成后者。

---

## 七、实测结果

### 真实语料 MultiHop-RAG（hybrid + reranker · judge = grok-4.5 · n=6 试点）

| 指标 | 均分 | 说明 |
|---|---|---|
| **groundedness**（忠实度） | **1.00** ✅ | 6/6 答案都只由检索到的上下文支撑，**零幻觉** |
| retrieval_relevance（检索质量） | 0.72 | 多数题检到相关证据；难多跳掉链 |
| correctness（正确率） | 0.67 | 4/6 正确 |
| helpfulness | 0.67 | 跟随正确率（查不到就诚实认怂） |

**读法（有强有弱，都讲清）**：
- ✅ **忠实度满分（1.00）**——hybrid + reranker 检索质量够 + grounded 生成 + `[source:]` 纪律，让模型「只用检到的、查不到就认怂」，**实测零幻觉**。这是生产 RAG 最要命的一项。
- ⚠️ **correctness / retrieval = 0.67 / 0.72**——6 题里 4 题满分，2 题失败的**都是难多跳**（如 “Between the report … published at 23:02 …” 这种精确时序 + 跨文档比较），单次检索没把**分散在多篇**的证据凑齐（这两题 `retrieval_relevance` = 0.50 / 0.00）。
- 🔑 **收口**：**单次强检索能把幻觉压到 0，但对「证据分散在多篇」的多跳仍有天花板**——而这正是被冻结的 agentic 多跳版本（tag `v1-agentic-comparison`）要补的地方。**强检索与 agentic 多跳是正交、互补的两条路。**
- 局限：n=6 偏小只看方向；单模型（grok）当裁判、不完全可复现；`retrieval_relevance` 对时序/数值类多跳判得偏严。

> 复现：`python eval_rag.py --n 6`（加 `--upload` 把这张表变成 LangSmith 上的 experiment）。

---

## 八、路线图

- **混合检索（✅）**：BM25 + bge dense 加权融合 + bge-reranker 重排，同一 `Retriever` 协议。
- **LLM-judge 评测（✅）**：openevals 四轴（correctness / groundedness / retrieval_relevance / helpfulness），judge=网关模型，可 `--upload` 到 LangSmith。
- **调优（可选）**：融合权重 / `pool` 大小 / reranker 升级 `bge-reranker-v2-m3` / 向量升级 `bge-base` 的消融对比。
- **向量库（可选）**：`DenseRetriever` 现为内存 numpy 余弦；数据量大时可换 chroma / faiss（同协议、上层不动）。
- **历史归档**：agentic 过程层 + 「agentic vs 单次」对比在 tag `v1-agentic-comparison`。

---

## 九、一句话面试话术

> 「RAG 的上限在检索，不在生成。所以我搭了一条生产级检索栈：**BM25 词法 + bge 向量加权融合**把两种召回的盲区互补，
> 再用 **cross-encoder 重排**精排——召回和精度分两步拿。后端全在一个 `Retriever` 协议后面可替换。评测我用 **openevals 的
> LLM-as-judge**（correctness / 忠实度 / 检索相关性 / helpfulness），裁判走网关模型，一键推 **LangSmith** 做实验追踪。
> 这个项目还有个 agentic 前身（模型自己决定查不查、多跳、何时停），冻结在一个 tag 里——**我知道 agentic 和强检索是两条正交的路，也知道各自该怎么量化。**」
