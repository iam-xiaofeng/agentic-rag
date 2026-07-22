# 混合检索 RAG 流水线 · BM25 + bge 向量 + 交叉编码重排 · LLM-as-judge 评测

> 一个**面试可讲、可跑、可评测**的生产级 RAG **检索流水线**：
> **召回**（BM25 词法 + bge 稠密向量，加权融合）→ **重排**（bge-reranker 交叉编码）→ **生成**（grounded、带引用）
> → **评测**（openevals 的 LLM-as-judge，可一键推 LangSmith）。
>
> 重点是**检索质量**：单一检索方式都有盲区，词法看词面、向量看语义，**混合 + 重排**是业界最扎实的组合。
> 检索后端全部藏在一个 `Retriever` 协议后面、**可替换**，换后端时上层（生成 / 评测）一行不动。

> **两条路并存**：`master` 上既有**单次强检索流水线**（`run.py` / `eval_rag.py`），也保留了 **agentic RAG**
> （`run_agentic.py`：模型自己决定该不该查 / 查几次 / 何时停）——两者共用同一套 `Retriever` 协议，是正交、互补的做法。
> 最初的「agentic vs 单次」对比实验另冻结在 tag **`v1-agentic-comparison`**（`git checkout` 可复现）。

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
├── requirements.txt     # 依赖：sentence-transformers / rank-bm25 / openevals / langchain-openai / langgraph / langsmith
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
│  ── agentic RAG（与流水线并存，正交的另一条路）──
├── prompts.py           # 四条 agentic 策略（该不该查 / 查几次 / 何时停 / 别幻觉）
├── tools.py             # rag_search：暴露给模型的唯一工具
├── agent.py             # create_agent 的 agentic loop（复用 llm.py）
├── run_agentic.py       # CLI：跑 agentic 检索，逐步打印每次改写 / 停
│
│  ── 语料 + 评测 ──
├── corpus_multihop.py   # 加载 data/ 的 MultiHop-RAG（609 篇 → 6194 片段 + 2556 问）
├── eval_dataset.py      # Example 数据类（评测样例的共享类型）
├── eval_rag.py          # openevals 4 个 LLM-judge + 可 --upload 推 LangSmith
├── langsmith_upload.py  # 把全量 MultiHop-RAG（含 gold 证据 fact）传成 LangSmith 数据集
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

# 4) agentic RAG —— 另一条路：模型自己决定查不查 / 多跳 / 何时停
python run_agentic.py "Who is the individual ... facing a criminal trial?"

# 5) 把全量数据集（含 gold 证据）传到 LangSmith（免费，不调模型）
python langsmith_upload.py               # 2556 题 → 数据集 multihop-rag
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
#   DenseRetriever    bge-large-en-v1.5 向量，归一化余弦（有 GPU 自动用 CUDA）
#   HybridRetriever   ↑两者加权融合 → bge-reranker-v2-m3 重排
```

**HybridRetriever 的三步**（`retriever_hybrid.py`）：

1. **召回**：BM25、Dense 各取 top-`pool`(默认 20)。
2. **融合**：各自分数 **min-max 归一化后按权重相加**（默认 `w_bm25 = w_dense = 0.5`）；同一片段两边都命中则叠加。
3. **重排**：融合候选池丢给 `bge-reranker-v2-m3`（cross-encoder），按 (query, passage) 相关性取 top-k。

> 想调：`HybridRetriever(docs, w_bm25=.3, w_dense=.7, pool=30)`。默认已用 `bge-large-en-v1.5`（1024 维）+ `bge-reranker-v2-m3`（GPU 上跑）；无显卡的机器可在 `retriever_dense.py` / `retriever_hybrid.py` 改回 `bge-small-en-v1.5` / `bge-reranker-base`。

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

### 真实语料 MultiHop-RAG（hybrid + reranker · judge = grok · n=6 试点）

当前默认模型 **bge-large-en-v1.5 + bge-reranker-v2-m3**（GPU）：

| 指标 | 均分 | 说明 |
|---|---|---|
| **groundedness**（忠实度） | **0.98** ✅ | 几乎零幻觉——生产 RAG 最要命的一项，稳 |
| retrieval_relevance（检索质量） | 0.65 | 见下「小样本警告」 |
| correctness（正确率） | 0.49 | 6 题里 3 满分、3 失败 |
| helpfulness | 0.48 | 跟随正确率（查不到就诚实认怂） |

**⚠️ 小样本警告——别拿 6 个样本给模型选型下结论**：

| 同 n=6 对照 | correct | ground | retr | help |
|---|---|---|---|---|
| bge-small + reranker-base | 0.67 | 1.00 | 0.72 | 0.67 |
| **bge-large + reranker-v2-m3**（现默认） | 0.49 | 0.98 | 0.65 | 0.48 |

- **换更大的模型在这 6 题上没有可辨识的提升**——groundedness 两组都 ~1.0，其余差异**全在噪声内**（就一道题从满分翻成失败，均值被拉低 0.15+）。
- 原因：n=6 太小 + LLM-judge（grok）**不完全可复现**；而且「更大的模型 ≠ 每条 query 都更好」，检索质量是**逐 query** 的。
- **结论**：要真分辨模型 / 参数好坏，得把 `--n` 拉到 **20~30+** 才有统计意义——这本身是个诚实的工程教训。
- 唯一稳的信号：**不管大小模型，grounded 生成 + `[source:]` 纪律把幻觉压到近 0**；失败集中在「证据分散多篇」的难多跳——正是并存的 agentic 多跳（`run_agentic.py`）要补的地方。**强检索与 agentic 多跳正交、互补。**

### LangSmith 数据集实验（`multihop-rag` · n=12 · 含 context_recall）

在上传的全量数据集上跑一次（`eval_rag.py --dataset multihop-rag --n 12`），比本地多一个**确定性 `context_recall`**（gold 文章标题 ∩ 检索到的 source，不花 LLM）：

| correctness | groundedness | retrieval_relevance | helpfulness | **context_recall** |
|---|---|---|---|---|
| 0.42 | **1.00** | 0.64 | 0.52 | **0.64** |

- **瓶颈定位清楚**：`groundedness` 满分（零幻觉），但 `context_recall` 只有 **0.64**——难多跳的 gold 证据分散在 2~4 篇，单次检索只捞到约 2/3，`correctness`(0.42) 因此被**检索召回**卡住，**不是模型幻觉**。
- `retrieval_relevance`（LLM judge，0.64）和确定性 `context_recall`（0.64）**几乎一致**——两个独立口径互相印证。
- **提升方向**：补召回（更多跳 / 更大 `pool` / 更好融合），或直接上并存的 **agentic 多跳**（`run_agentic.py`）——正是它的用武之地。

> 复现：`python eval_rag.py --dataset multihop-rag --n 12`（结果落 LangSmith experiment `hybrid-on-multihop-rag-*`）；本地小样对照：`python eval_rag.py --n 6`。

---

## 八、路线图

- **混合检索（✅）**：BM25 + bge dense 加权融合 + bge-reranker 重排，同一 `Retriever` 协议。
- **LLM-judge 评测（✅）**：openevals 四轴（correctness / groundedness / retrieval_relevance / helpfulness），judge=网关模型，可 `--upload` 到 LangSmith。
- **调优（可选）**：融合权重（`w_bm25/w_dense`）/ `pool` 大小的消融对比（编码 / 重排已上 `bge-large-en-v1.5` + `bge-reranker-v2-m3`，GPU）。
- **向量库（可选）**：`DenseRetriever` 现为内存 numpy 余弦；数据量大时可换 chroma / faiss（同协议、上层不动）。
- **agentic RAG（并存 ✅）**：`run_agentic.py`（create_agent，四策略），与流水线共用 `Retriever` 协议；最初的「agentic vs 单次」对比在 tag `v1-agentic-comparison`。
- **历史归档**：自研确定性指标 + agentic-vs-单次对比脚本仍在 tag `v1-agentic-comparison`（未搬回 master）。

---

## 九、一句话面试话术

> 「RAG 的上限在检索，不在生成。所以我搭了一条生产级检索栈：**BM25 词法 + bge 向量加权融合**把两种召回的盲区互补，
> 再用 **cross-encoder 重排**精排——召回和精度分两步拿。后端全在一个 `Retriever` 协议后面可替换。评测我用 **openevals 的
> LLM-as-judge**（correctness / 忠实度 / 检索相关性 / helpfulness），裁判走网关模型，一键推 **LangSmith** 做实验追踪。
> 这个项目还**并存**着一个 agentic RAG（`run_agentic.py`：模型自己决定查不查、多跳、何时停）——**我知道 agentic 和强检索是两条正交、互补的路，也知道各自该怎么量化。**」
