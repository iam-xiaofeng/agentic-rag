# 混合检索 RAG 流水线 · BM25 + bge 向量 + 交叉编码重排 · LLM-as-judge 评测

> 一个**面试可讲、可跑、可评测**的生产级 RAG **检索流水线**：
> **召回**（BM25 词法 + bge 稠密向量[Chroma]，RRF 融合）→ **重排**（bge-reranker 交叉编码）→ **生成**（grounded、带引用）
> → **评测**（openevals 的 LLM-as-judge，可一键推 LangSmith）。
>
> 重点是**检索质量**：单一检索方式都有盲区，词法看词面、向量看语义，**混合 + 重排**是业界最扎实的组合。
> 检索后端全部藏在一个 `Retriever` 协议后面、**可替换**，换后端时上层（生成 / 评测）一行不动。

> **两条路并存**：`master` 上既有**单次强检索流水线**（`run.py` / `evals/eval_rag.py`），也保留了 **agentic RAG**
> （`run_agentic.py`：模型自己决定该不该查 / 查几次 / 何时停）——两者共用同一套 `Retriever` 协议，是正交、互补的做法。
> 最初的「agentic vs 单次」对比实验另冻结在 tag **`v1-agentic-comparison`**（`git checkout` 可复现）。

---

## 一、为什么这么设计（核心动机）

RAG = **检索(retrieve) → 拼上下文(augment) → 生成(generate)**。**答案质量的上限，几乎全由检索决定**——
检索不到，再强的模型也只能幻觉。所以本项目把功夫下在检索侧，用业界公认最稳的三段式：

1. 🔤 **词法召回（BM25）**——看**词面重叠**。命中精确实体 / 稀有词很强，但换个说法就抓瞎。
2. 🧠 **向量召回（bge dense）**——看**语义相近**。措辞不同也能召回，但对精确实体 / 数字不敏感。
3. 🎯 **交叉编码重排（bge-reranker）**——把 (query, passage) **一起**读，算一个比双塔向量更准的相关性分。贵，所以只重排少量候选。

> **为什么两路召回要融合、还要重排**：词法和向量的盲区**互补**——RRF 融合先把两种「看法」的候选按排名并起来（提召回），
> 再用 cross-encoder 精排（提精度）。这就是 “hybrid retrieval + reranking” 成为生产标配的原因。

检索后端解耦在 `Retriever` 协议后面：当前实现 `BM25Retriever` / `DenseRetriever` / `HybridRetriever` 三个后端，
**同一套接口**，`rag/pipeline.py`（生成）和 `evals/eval_rag.py`（评测）对换后端无感。

---

## 二、目录结构（每个文件干什么）

```
agentic-rag/
├── README.md            # 本文件
├── EXPERIMENTS.md       # 28 次实验的设置 / 命令 / 数据 / 结论（含被后续实验推翻的条目，原文保留 + 订正标注）
├── requirements.txt     # 依赖：sentence-transformers / chromadb / rank-bm25 / openevals / langchain-openai / langgraph / langsmith
├── .env.example         # 配置模板（复制成 .env 填真实值）
├── .env                 # 真实密钥（已 gitignore；模型 + LangSmith 凭据）
├── run.py               # CLI：对一个问题跑 hybrid 检索（可选生成），打印排名
├── run_agentic.py       # CLI：跑 agentic 检索，逐步打印每次改写 / 停
│
├── rag/                 # ★ 核心：检索栈 + agent。**只被依赖，不依赖 evals/**
│   │  ── 检索：一个协议 + 三个后端 ──
│   ├── retriever.py         # Retriever 协议 + Doc/Hit 数据类（后端地基）
│   ├── retriever_bm25.py    # BM25Retriever：词法检索（rank_bm25，零重依赖）
│   ├── retriever_dense.py   # DenseRetriever：bge 编码 → Chroma 持久化向量库（.cache/chroma/）
│   ├── retriever_hybrid.py  # HybridRetriever：BM25+向量 RRF 融合 → bge-reranker 重排（★ 主力）
│   ├── retriever_decompose.py  # 查询分解 → 逐子问句各自重排 → 轮流取名额（RAG_DECOMPOSE=1）
│   ├── reranker_qwen.py     # Qwen3-Reranker-4B（instruction-aware，评测/对照用，非默认）
│   │  ── 生成 + agentic ──
│   ├── llm.py               # build_model / build_judge：OpenAI 兼容网关 + .env 自动加载
│   ├── pipeline.py          # answer：检索 → 一次 grounded 生成（带 [source:] 引用）
│   ├── prompts.py           # agentic 六条策略（v3；改前先读 EXPERIMENTS 实验27④/28）
│   ├── tools.py             # rag_search：暴露给模型的唯一工具
│   ├── agent.py             # create_agent 的 agentic loop
│   │  ── 语料 ──
│   ├── corpus_multihop.py   # MultiHop-RAG（609 篇 → 15172 片段 + 2556 问）⚠️ 非真多跳，见第七节①
│   ├── corpus_musique.py    # MuSiQue（HuggingFace，21100 段 + 2417 问，2/3/4hop）★ 真桥接多跳
│   └── dataset.py           # Example 数据类（评测样例的共享类型）
│
├── evals/               # 评测脚本。每个都能 `python evals/xxx.py` 直接跑
│   ├── eval_judge.py        # ★ **头号指标**：裁判读 agent 自引的关键句 → correct/sufficient/faithful
│   ├── eval_agentic.py      # 跑 agent + 确定性指标；--local 落逐题 JSONL（eval_judge 的输入）
│   ├── eval_benchmark_probe.py  # ★ **判一个"多跳"数据集是不是真多跳**——换数据集前先跑它
│   ├── eval_answerability.py    # 地板/实测/天花板三条线（跑一次的常数，给 correct 减地板）
│   ├── eval_rebuild.py      # ★ 逐层重建验收：--layer 0 前置自检 / 1 chunk / 2 融合 / 3 重排（免费）
│   ├── eval_common.py       # 评测公共件（openevals 装配 + 确定性评估器）——库，无 CLI
│   ├── eval_rag.py          # 单次流水线的 openevals 评测，可 --upload 推 LangSmith
│   ├── eval_query.py / eval_chain.py / eval_diag.py / eval_ablation.py / eval_rerank_effect.py
│   ├── eval_rescore.py      # 用新裁判把存档 run 统一重打分（裁判下线时保旧实验可比）
│   └── langsmith_upload.py  # 把全量 MultiHop-RAG（含 gold 证据）传成 LangSmith 数据集
│
├── data/                # ← MultiHop-RAG 语料（已 gitignore；见「第三节」）
│   ├── corpus.json          # 609 篇新闻全文
│   └── MultiHopRAG.json     # 2556 个多跳问题 + gold 证据
└── runs/                # 后台实验的日志与逐题 dump（已 gitignore）
    └── dumps/               # eval_agentic.py --local / eval_judge.py 的 JSONL
```

> **依赖方向是单向的：`evals/ → rag/`。** 早先 `rag/corpus_multihop.py` 反过来依赖
> `eval_dataset.py`（核心依赖评测），重组时把那个共享数据类挪成了 `rag/dataset.py`。


---

## 三、`data/` 详解（这里是什么）

`data/` 放的是 **MultiHop-RAG** —— 一个**多跳 RAG 评测基准**，自带语料 + 问题 + gold 证据。

- **来源**：HuggingFace [`yixuantt/MultiHopRAG`](https://huggingface.co/datasets/yixuantt/MultiHopRAG)（论文 *MultiHop-RAG*, 2024）；许可 **ODC-BY**（公开、署名即可）。
- **为什么选它**：证据被**故意分散在 2~4 篇文章**里 → 对检索的召回是真实压力测试；自带 gold 证据；只有 609 篇，笔记本几分钟建完索引。
- **两个文件**：
  - `corpus.json`：609 篇新闻（`title` 标题 / `body` 正文 / `source` 媒体 / …）——**被检索的语料**。
  - `MultiHopRAG.json`：2556 个问题（`query` / `answer` 标准答案 / `question_type` / `evidence_list` gold 证据）——**评测集**。
- **我们怎么用**（`rag/corpus_multihop.py`）：
  - `load_corpus()`：每篇 `body` 用 **递归切分**（`RecursiveCharacterTextSplitter`，优先 `\n\n`→`\n`→句子边界，**600 字符 / 重叠 150**）切块、拼上标题 → **15172 个 `Doc`**（`source = 文章标题`，与 gold 证据标题对齐）。切分参数是整条栈最上游的旋钮，选型过程见 EXPERIMENTS 实验 17-19。
  - `load_examples()`：每个问题 → 一条 `Example`（`question` / `reference=answer` / `sources=gold 标题` / `kind`）。

### 如何重新下载

```bash
cd agentic-rag && mkdir -p data
curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json      -o data/corpus.json
curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json -o data/MultiHopRAG.json
# 离线自检（无需模型）：
.venv/bin/python -c "from rag.corpus_multihop import load_corpus, load_examples; print(len(load_corpus()), '片段', len(load_examples()), '例')"  # 15172 片段 2556 例
```

---

## 四、快速开始

```bash
cd agentic-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 含 sentence-transformers（会拉 torch）；首次跑会自动下 bge 模型
cp .env.example .env                    # 填 OPENAI_API_KEY / OPENAI_BASE_URL / RAG_MODEL / LANGSMITH_*（真 key 放 .env）

# 1) 免费离线自检（不调模型）：语料加载 + 分块
python -c "from rag.corpus_multihop import load_corpus, load_examples; print(len(load_corpus()), '片段', len(load_examples()), '例')"

# 2) 跑 hybrid 检索（首次会下 bge 模型 + 编码 15172 片段，之后走 .cache 缓存）
python run.py "Who is the individual associated with the cryptocurrency industry facing a criminal trial?"
python run.py --no-gen "..."           # 只看检索排名，不调生成模型

# 3) LLM-judge 评测（openevals，judge=模型；--upload 另推 LangSmith）
python evals/eval_rag.py --n 8                # 本地打分表
python evals/eval_rag.py --n 8 --upload       # 同上 + 推 dataset + experiment 到 LangSmith

# 4) agentic RAG —— 另一条路：模型自己决定查不查 / 多跳 / 何时停
python run_agentic.py "Who is the individual ... facing a criminal trial?"

# 4b) **推荐路径**：MuSiQue（真桥接多跳，见第七节②）+ 头号三指标，全程不碰 LangSmith
#     语料/题目走 HuggingFace 缓存（~/.cache/huggingface），仓库里只有加载器 rag/corpus_musique.py
python -c "from rag.corpus_musique import load_corpus, load_questions; print(len(load_corpus()),'段',len(load_questions()[0]),'题')"
RAG_TOPK=8 python evals/eval_agentic.py --benchmark musique --local runs/dumps/mq.jsonl --per-type 7
python evals/eval_judge.py runs/dumps/mq.jsonl --benchmark musique          # correct/sufficient/faithful
python evals/eval_judge.py runs/dumps/新.jsonl --benchmark musique --baseline runs/dumps/旧.jsonl  # 同题配对 + 95% CI

# 4c) 换评测集之前先跑这个：判它是不是**真**多跳（全确定性、无 LLM、几分钟）
python evals/eval_benchmark_probe.py                # MultiHop-RAG / MuSiQue / HotpotQA 同口径横比

# 5) 把全量数据集（含 gold 证据）传到 LangSmith（免费，不调模型）
python evals/langsmith_upload.py               # 2556 题 → 数据集 multihop-rag
```

> **网关提示**：某些中转网关 WAF 拦未知 UA → 设 `RAG_USER_AGENT`；偶发 5xx/超时 → `RAG_MAX_RETRIES` / `RAG_TIMEOUT`（`build_model` 已读取）。
> `evals/eval_rag.py` 单题失败会跳过、不毁整轮。

---

## 五、架构与关键设计

**一句话**：`Retriever` 协议给我可替换后端；hybrid 把词法/向量的盲区互补起来；reranker 精排；生成只用检索到的内容并标引用。

```python
# rag/retriever.py —— 后端只需实现这个协议，其余代码全不动
@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[Hit]: ...

# 三个后端，同一协议：
#   BM25Retriever     词法（rank_bm25）
#   DenseRetriever    bge-large-en-v1.5 向量 → Chroma 持久化库，余弦近邻（有 GPU 自动用 CUDA）
#   HybridRetriever   ↑两者 RRF 融合 → bge-reranker-v2-m3 重排
```

**HybridRetriever 的三步**（`rag/retriever_hybrid.py`）：

1. **召回**：BM25、Dense 各取 top-`pool`(默认 **200**)。pool 的**信息量**要与 chunk 匹配——`pool×chunk ≈ 恒定`；chunk 缩到 600 后 100→200 有收益、200→300 只涨池覆盖不涨交付（实验 19 ⑥）。
2. **融合**：**RRF 倒数排名融合**（默认，`score = Σ w/(rrf_k+rank)`）——只看排名、不看分数，对 BM25(无上界) 与余弦([-1,1]) 不可比的量纲免疫（业界标配）；另留 min-max 加权(`fusion="minmax"`)做对照。默认权重均衡 `w_bm25 = w_dense = 0.5`。
3. **重排**：融合候选池丢给 `bge-reranker-v2-m3`（cross-encoder），按 (query, passage) 相关性取 top-k。

> 想调：`HybridRetriever(docs, fusion="rrf", w_dense=.7, pool=200, rrf_k=60)`；参数消融见 **EXPERIMENTS 实验 6 / 19**（`python evals/eval_ablation.py`、`python evals/eval_rebuild.py`，免费、不调 LLM）。默认已用 `bge-large-en-v1.5`（1024 维）+ `bge-reranker-v2-m3`（GPU 上跑）；无显卡的机器可在 `rag/retriever_dense.py` / `rag/retriever_hybrid.py` 改回 `bge-small-en-v1.5` / `bge-reranker-base`。
> **整套配置可在 shell 层整体切换**做 A/B，不必改代码：`RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` / `RAG_POOL` / `RAG_TOPK`。

---

## 六、评测设计

### 6.1 头号指标（`evals/eval_judge.py`）—— 一次裁判调用，三个分

指标一度长到 15 个，**头号指标却一直不可信**，于是只能不断加诊断指标去补。2026-08 推倒重来：

| 指标 | 量什么 |
|---|---|
| `correct` | 答案与标准答案是否实质一致 |
| `sufficient` | **agent 自己引的那几句**够不够推出标准答案（= "召回"，**按需判、不要求全覆盖**） |
| `faithful` | 答案里的每个论断能否追回到它引的句子 |

为什么不再用「gold 证据句子串匹配」当头号指标（实验25/26 实测）：**过严**（gold 句只是作者挑的
某一句，同篇另一句同样能支撑却判 0）、**冗余同权**（MultiHop-RAG 的 inference 平均 3.36 条证据里
只有 1.47 条真含答案）、**没有成本项**（搜 10 次必然赢过搜 1 次）。所以改由裁判**逐题判哪些证据是必��的**。

### 6.2 守卫（确定性，零成本）—— 没有它，上面三个分会在最关键的地方骗人

| | |
|---|---|
| `cited_grounded` | agent 引的句子里，真能在它**实际检索到**的上下文中找到的比例 |

引用是 agent 的**自述**。不核对的话，「没检索到」和「检索到了但没引」长得一模一样 —— 会写引用的
模型分高、检索一样好但引用潦草的模型分低。低于 0.8 时 `evals/eval_judge.py` **拒绝给出结论**。

> **这个守卫救过一次场**（实验27）：某版提示词让 `faithful` 从 0.524 涨到 0.619，看着是改对了；
> `cited_grounded` 同时从 0.94 掉到 0.77 —— 真相是 agent 在**编引用**，而忠诚于自己编的句子当然容易。

### 6.3 诊断（`--diag` 才打印，只在头号指标动了、要归因时看）

`delivered` = 检索**实际交付**的证据链比例 / `fate_cited` 引用了 / `fate_uncited` **检索到却没引**
/ `fate_missing` 真缺口 —— **这三者必须分开**，混在一起会把提示词问题误判成检索问题、去调错的旋钮。
另有 `recall@B`（固定上下文预算）、逐跳边际召回、`n_search`、`n_unsupported`。

### 6.4 退役

`context_recall`(title 级，实验8-9 证明**方向会反**)、openevals 的 `groundedness` /
`retrieval_relevance` / `helpfulness`（从未提供过信息量，已被头号三分覆盖）。
`context_recall_fact` 降级为**过严的下界**，只在它与裁判分打架时看。

### 6.5 两条永远要记得的

- **裁判固定**（`RAG_JUDGE_MODEL`）。换裁判 = 换尺子，新旧分数不可比（实验13）。
- **`correct` 的绝对值要减地板**。MultiHop-RAG 的 comparison 96% 是 Yes/No 且 59.8% 答 "yes" ——
  一个恒答 "Yes" 的空壳就有 0.60。MuSiQue 无此问题（最高频答案仅占 1.4%）。

---

## 七、实测结果

> 完整的 28 次实验（设置 / 命令 / 数据 / 结论，含被后续实验推翻的条目）都记在
> **[`EXPERIMENTS.md`](EXPERIMENTS.md)**；这里只放最能打的几张表。

**① MultiHop-RAG 不是多跳评测集**（实验25 · 全确定性、无 LLM · `python evals/eval_benchmark_probe.py`）

同一把尺子横量三个数据集（例内 BM25、候选池统一 20 篇、top5）：

| 数据集 / 子集 | ⚑**捷径率** | 一次拿全 | 最高频答案占比 |
|---|---|---|---|
| MultiHop-RAG 全部 | **99.3%** | 76.5% | 34.5% |
| HotpotQA bridge | **96.5%** | 95.7% | 0.4% |
| MuSiQue 2hop | 79.6% | 78.6% | 0.8% |
| MuSiQue 3hop | 81.8% | 58.3% | 4.0% |
| **MuSiQue 4hop** | **69.0%** | **33.3%** | 12.1% |

⚑捷径率 = 光靠**原问句**就够到了**含答案那篇** → 中间的跳可以整个跳过。

- MultiHop-RAG 里 **99.3% / 92.2% / 83.9%** 的问句已把每篇 gold 文章的**出处点名**
  ——不存在"先查 A 才知道要查 B"。它是 **multi-document（跨文档聚合）**，不是 **multi-hop（顺序推理）**。
- **连"名字是 bridge"也不可信**：HotpotQA 的 bridge 子集桥接率 90.3%，捷径率仍有 96.5% ——
  **结构像桥接，行为不是**。所以换数据集前先跑这个探针，别看名字。
- 由此统一解释了此前三件一直没解释通的事：`n_search` 恒在 1.2、k 从 4 到 64 不改变迭代次数、
  强制查询分解端到端收益为零。**agentic RAG 在 MultiHop-RAG 上结构性地拿不到优势，不是实现问题。**

**② 换到 MuSiQue：一行代码没改，agent 自己就多跳了**（实验26 · 同模型同配置同规模 n=21）

| | MultiHop-RAG | MuSiQue |
|---|---|---|
| **n_search** | 1.24 | **1.81** |
| correct | 0.762 | 0.476 |
| 猜测下限 | ≈0.46 | **≈0.01** |
| **高出下限** | +0.30 | **+0.46** |

**分数低不等于更差**：MuSiQue 那个 0.476 里的真信号比 MultiHop-RAG 的 0.762 还多。

**③ 压 k：省三分之二上下文而覆盖不变**（实验27 · **三臂配对** n=21 · 同题同 `example_id`）

| 配对对照 | 检索次数 | 片段数 | 证据链交付 |
|---|---|---|---|
| 提示词（v1→v2, k 都=32） | +0.43 [−0.29,+1.10] 跨0 | +13.7 跨0 | +0.016 跨0 |
| **k（32→8, 提示词相同）** | **+0.67 [+0.24,+1.14] ✅** | **−48.4 [−64.8,−32.0] ✅** | +0.012 跨0 |

**同样的证据链覆盖，上下文只花三分之一。** 而同一个旋钮在 MultiHop-RAG 上完全无效（k 4→64
纹丝不动）——**差别全在题目结构**：那边一次就能拿全，压 k 没有压力传导。

**④ 一次失败的改动，留档**（实验27 ④）：给提示词加"N 跳至少 N 行引用"想修"引用不全"，
结果 `cited_grounded` **0.94→0.77** —— agent 为凑行数**编引用**。
**约束了输出形状，就会得到形状，代价是内容。** 提示词遂全文重写（v3），第一原则改成
「任何"必须产出 X"的要求，都必须同时给出一条不用编造的出路」（`unsupported: <哪一环>`），
**忠诚度优先于完整度**。

**⑤ 逐层重建检索栈**（实验 19-20 · 每型 60 题 · 确定性、免费、可复现 · `python evals/eval_rebuild.py`）：

| 层 | 改动 | 单点自检（给这层最有利的输入） | 结果 |
|---|---|---|---|
| BM25 | **不动** | 用证据句原文自查 top1 | 97~100%，全程健康 |
| embedding | chunk 1200→**600**/overlap 150 | 同上，dense | **56% → 73%**，三类齐涨 |
| 融合 | pool 100→**200** | 融合 ≥ 更好的那条单路腿？ | @4 **−0.026 → +0.022**（由负功转正） |
| 重排 | 不动（bge） | 重排 ≥ 不重排？ | comparison/inference 显著为正，temporal 无可检出效应 |

**等交付字符**下的 fact 级召回（重排后）：旧 `1200/pool100/@16` **0.749** → 新 `600/pool200/@32` **0.779**，**三个题型全部改善**（temporal +0.058）。若 `k` 不变，则**上下文减半而召回略高**（0.672 vs 0.650）。

**⑥ 端到端 A/B（诚实版）**（实验 21 · 同模型同题配对 · n=87 · `deepseek-v4-pro`）：`context_recall_fact` 配对差 **+0.022，95% CI [−0.066, +0.105]** —— **测不出差异**，不能宣称端到端变好。原因见实验21 ③：agent 自己已经在做查询改写、方差被 agent 的非确定性主导，要分辨 0.03 需每型约 700 题。
> **方法论：改动发生在哪一层，就在哪一层量。** 端到端该回答"这套系统整体行不行"，不该回答"我这个部件改好了没有"。

**⑦ agentic 多跳 · 100 题 · 按 question_type**（⚠️ **旧配置** RRF+Chroma+pool=100+k=8 + 答题模型 grok-4.5 · LangSmith `agentic-on-multihop-rag-d61482cc`；grok 已从网关下线、配置也已换代，此表仅作历史参考）：

| type | correctness | groundedness | retrieval_rel | helpfulness | context_recall | refused |
|---|---|---|---|---|---|---|
| comparison | 0.80 | 0.74 | 0.60 | 1.00 | 0.85 | 0.00 |
| inference | **1.00** | 0.75 | 0.87 | 1.00 | 0.72 | 0.00 |
| temporal | 0.67 | 0.83 | 0.54 | 0.97 | 0.83 | 0.16 |
| null | – | 0.96 | 0.03 | 0.64 | – | **0.92** |

- **多跳 correctness ~0.82**（vs 单次 0.42）：inference 满分、temporal 最难、null 92% 正确拒答；groundedness 0.77。（表中 `context_recall` 是**标题级**口径，仅为与早期实验可比而保留——它是漏水的代理，见下条；现行评测已改用 fact 级 `context_recall_fact`，honest 值 ~0.63。）
- **一条踩过、也修好的坑（含金量高）**：曾用 source 去重把 title 级 `context_recall` 刷到 0.89，correctness 却掉了——fact 级诊断发现去重在**游戏漏水的代理指标**、丢了真证据（Goodhart）。撤销去重、只留 `k=8` 后 correctness 反升到 **0.82**（实验 8-9）。**教训：盯 correctness / fact 级召回，别优化 title 级代理。**
- **第二条坑（评测框架本身）**：`target()` 里一句 `except: return 空结果` 把网关 502 吞成空答案，被裁判打 **0 分并计入均值**，整轮实验读起来像"模型变差了"（实测 temporal 8 题空了 7 题）。修法：长退避重试 → 仍失败就 `raise`，让 LangSmith 标 **errored 并排除出均值**。**教训：评测里失败必须可见，不能降级成 0 分**（实验 12）。
- **第二条坑的复发（实验 21 ①）**：`_run_once` 在流式事件里只读 `messages[-1]`，**同一轮的并行 `rag_search` 被吞掉**——实测 2 次检索只捕获 1 次，`context_recall_fact` 报 0.500 而真值 1.000。**这个 bug 只会让检索看起来更差**，"召回上不去"的表象里有一部分是评测自己造的。同时新增确定性指标 **`n_search`**：此前"**模型压根没查**"（实测 90 题里有 6 题）与"**检索器没捞到**"在指标上长得一模一样，归因会全错。
- **第三条坑：分子群分析救了我一次，又坑了我一次**（实验 15 → 实验 20）。实验15 分题型发现"bge 在 temporal 上净减益 **−0.100**、51% 的证据被往后推"，写成头条、进了 README、指导了后续三个实验的方向。**实验20 用同一配置把每型 15 题加到 60 题，这个数没能复现**：Δ@8 变成 **+0.031，95% CI [−0.042,+0.108]（不含 −0.100）**，"51% 被推后"变成 39% 被推后、中位名次 6→9 变成 8→7 **推前**。原因很朴素：**15 题的标准误差约 0.09，而结论依据的效应量是 0.10——信噪比约等于 1。** ⇒ **"按子群拆开验"这个动作是对的，但拆子群会让样本变小、更容易把噪声读成信号：分子群分析必须配置信区间。**
- **reranker 的真实价值（实验 20，每型 60 题 + 配对 bootstrap）**：comparison **+0.13 [+0.07,+0.20]**、inference **+0.11 [+0.04,+0.18]**（@8，区间不跨 0，真效应）；temporal 四个 k 的区间**全部跨 0**、点估计正负横跳（+0.022/−0.039/−0.025/+0.028）——**不是负增益，是没有可检出的效应**。默认保留 bge。
- **第四条坑：chunk 对比没控制住「交付信息量」**（实验 18）。实验11 用固定 `fact@8` 比不同 chunk——8 个 1200 字符的块 = 9600 字符、8 个 300 字符的块只有 2400 字符，**信息量差 4 倍**，小块必输。按**总字符对齐**重测后结论反转，且 **dense 精度只跟「块长」有关、跟「递归/按结构切」无关**（纯段落1000 与递归1200 自查同为 60%）。
- **第五条坑：尺子本身对被比的变量不中立**（实验 19 ①）。我们从实验10 起用的 `fact@k` 只检查证据句的**前 120 字符**——**证据被切断它完全看不见**：chunk=400 时全句留存已掉到 89~95%，而"前120留存"仍报 100%。**于是块越小越占便宜**，实验18 的"~500 甜点"是被这把偏心的尺子量出来的，计入截断后落点是 **600**。⇒ **控制变量要控制到「指标本身」：先怀疑尺子，再怀疑被测对象。**
- **最贵的一条：embedding 在最简单的任务上只有一半命中率**（实验 17 → 实验 19 修复）。**用证据句的原文去检索包含该原文的 chunk**，chunk=1200 时 dense 只有 **56%** 排到第 1，而 BM25 是 **98%**。排除了 Chroma 后定位到 **长块稀释句子级语义**。**这个硬伤从项目第一天就在，被所有端到端指标漏掉了。**
  > **教训：端到端指标必须配一组「单点能力自检」**——给每个部件最有利的输入，看它能否接近满分。否则某个部件长期半残，整体指标只会显示「就这样了」，不会告诉你是谁拖的。
- **融合层曾经是负资产，修好上游后才转正**（实验 16 → 实验 19 ④）。实验16 测得"融合从未提升交付、上游优化空间≈0"，**那是坏 dense 腿的下游症状**：chunk=1200 时融合相对更好单路腿在 @4 是 **−0.026（负功）**、@8 约等于 0；chunk 改成 600 后**每个 k 都是正的**（+0.022~+0.059）。⇒ **别把"坏输入下的表现"当成部件的固有属性。**
- **查询侧：真正起作用的是「重排跟着子问句走」，不是「分解」本身**（实验 14 → 实验 22-23，每型 100 题 + 配对 CI）。**同一批子问句、同一候选池**（池覆盖一模一样）下，"每个子问句各自重排再拼"比"合并后用原问句重排"高 **+0.059 [+0.026,+0.093]** —— 这个纯机制对照比"分解 vs 不分解"（+0.050 [+0.014,+0.085]）还强。而**只做分解不动重排（−0.009）、同义改写（−0.012）都跨 0，等于白做**。
  > 实验14 的**机制说对了、措辞和效应量说错了**：它的"唯一越过不重排天花板"依赖一个 15 题的伪基线——**不重排其实是显著更差的**（−0.040 [−0.073,−0.006]）。
  > **「扩大召回若不改变前排顺序，就不会变成交付」——这条规律在实验16 / 19 / 22 / 23 上重复了四次。**
  > ⚠️ 未落地：代价是每次检索多 1 次 LLM 调用 + 3 倍重排；而 agent 本身已在自发做事实层改写（实验21 ③），**这份收益可能与之重叠**，落地前得先测端到端还剩多少。


> **一句话：多跳不是万能钥匙，是"用忠实度和成本换覆盖与正确率"的可量化选择。** 全部实验在 LangSmith 数据集 `multihop-rag` 上可复现，细节见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

---

## 八、路线图

- **混合检索（✅）**：BM25 + bge dense 加权融合 + bge-reranker 重排，同一 `Retriever` 协议。
- **评测指标推倒重建（✅，实验25-27）**：openevals 四轴 → **头号三分 + 一个确定性守卫**
  （`correct` / `sufficient` / `faithful` + `cited_grounded`），见第六节。起因是指标长到 15 个而
  **头号指标一直不可信**，只能不断加诊断去补。`evals/eval_judge.py`，一次裁判调用出三个分。
- **检索调优（✅）**：RRF 融合 + 参数消融（实验 6）——`rrf_k` / 权重在合理区间不敏感。**pool 不是独立杠杆**：它的**信息量**要与 chunk 匹配（`pool×chunk ≈ 恒定`，实验 19 ⑥），且**只涨池覆盖不改前排顺序时，不会变成交付**。
- **逐层重建检索栈（✅，实验 19-21）**：BM25 不动 → embedding（chunk 1200→600）→ 融合（pool 100→200）→ 重排（保留 bge）。**每层两条验收缺一不可**：给这层最有利输入的**单点自检** + **按题型均衡**；从实验20 起再加**配对 bootstrap 置信区间**。脚本 `evals/eval_rebuild.py --layer 0/1/2/3`，全程确定性、免费。
- **更强 reranker（✅，评测级）**：`rag/reranker_qwen.py` 提供 instruction-aware 的 **Qwen3-Reranker-4B**（`evals/eval_agentic.py --reranker qwen`），fact 级交付显著优于 bge；代价 ~13s/次重排 + ~8GB 显存，故**默认仍是 bge**、4B 作评测/对照用（实验 11-12）。
- **下一步（待做）**：① **样本量** —— 目前 MuSiQue 上所有结论都卡在 n=21（区间宽 ±0.1~0.2），
  要判效应得 per-type 30（=90 题）；② **提示词的忠诚度问题**（见上）；③ 把「逐子问句重排」落到
  `rag/tools.py`（检索侧已证实 +0.059 [+0.026,+0.093]，但 agent 本身在自发改写，需先测端到端还剩多少，
  实验 23 ④）；④ **MuSiQue 4hop 是最该攻的靶子** —— 三个数据集里唯一"一次检索只有 1/3 能拿全"的子集。
- **答题模型可换、裁判要固定（✅）**：`evals/eval_agentic.py --model <名>` 只换答题端，裁判走 `llm.build_judge()`（`RAG_JUDGE_MODEL`）不动；裁判模型下线时用 `evals/eval_rescore.py` 拿新裁判把**存档 run** 统一重打分，旧实验照样可比（实验 13）。
- **配置可在 shell 层整体切换（✅）**：`RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` / `RAG_POOL` / `RAG_TOPK` —— 新旧配置 A/B 不必改代码（实验 21）。
- **向量库（✅）**：`DenseRetriever` 已从内存 numpy 换成 **Chroma** 持久化向量库（`.cache/chroma/`，近邻检索）——同 `Retriever` 协议、上层一行不动。⚠️ HNSW **只在 top1 上与精确余弦 100% 一致**；top100 的集合重合是 96%（`evals/eval_rebuild.py --layer 0`）。差异落在 k 截断的边界上，重排后基本被吸收（绝对值差 ≤0.005）。
- **agentic RAG（并存 ✅）**：`run_agentic.py`（create_agent，六条策略），与流水线共用 `Retriever` 协议；最初的「agentic vs 单次」对比在 tag `v1-agentic-comparison`。
- **换到真多跳评测集（✅，实验25-26）**：`evals/eval_benchmark_probe.py` 实测 MultiHop-RAG 捷径率 **99.3%**
  ——它是跨文档聚合、不是顺序推理，**agentic 在它上面结构性地拿不到优势**。改用 **MuSiQue**
  （`rag/corpus_musique.py`，21100 段 / 2417 题 / 2-3-4hop），换过去后 `n_search` 一行代码没改自己涨了 46%。
- **提示词（🟡 未达标，实验27-28）**：为修"引用不全"改过两版，**一次反向（`cited_grounded` 0.94→0.77，
  agent 凑行数编引用）、一次持平（v3 加 `unsupported:` 出路，回到 0.87）**。目前忠诚度最高的仍是
  什么都不要求的 v1。**这是当前最大的未解问题。**
- **历史归档**：自研确定性指标 + agentic-vs-单次对比脚本仍在 tag `v1-agentic-comparison`（未搬回 master）。

---

## 九、一句话面试话术

> 「RAG 的上限在检索，不在生成。所以我搭了一条生产级检索栈：**BM25 词法 + bge 向量（Chroma 库）RRF 融合**把两种召回的盲区互补，
> 再用 **cross-encoder 重排**精排——召回和精度分两步拿。后端全在一个 `Retriever` 协议后面可替换。评测我用 **openevals 的
> LLM-as-judge**（correctness / 忠实度 / 检索相关性 / helpfulness），裁判走网关模型，一键推 **LangSmith** 做实验追踪。
> 这个项目还**并存**着一个 agentic RAG（`run_agentic.py`：模型自己决定查不查、多跳、何时停）——**我知道 agentic 和强检索是两条正交、互补的路，也知道各自该怎么量化。**」

**如果只讲一件事，我会讲这个**：

> 「这条栈我推倒重来过一次，因为排查发现 **embedding 从项目第一天起就是半残的**——拿 gold 证据句的**原文**去检索包含它的片段（对检索器最有利的输入，query 就是答案本身），dense 只有 **56%** 排到第 1，而 BM25 是 98%。
> **所有端到端指标都没报警**，它们只会显示"就这样了"，不会告诉你是谁拖的。
> 更值钱的是后来发现：我此前几次方向性误判，**没有一次是模型不行，全是量错了**——网关 502 被 `except` 吞成 0 分计入均值；全类型平均掩盖了子群；**分子群之后样本太小，又把噪声（n=15 的 −0.100）读成了系统性结论并写进 README**；跨 chunk 比较时 `fact@k` 只查证据句前 120 字符，对小块系统性宽容。
> 所以我把重建做成了**逐层验收**：每层给它最有利的输入做**单点自检**、**按子群拆**、**配对 bootstrap 置信区间**、**对齐混杂变量**、还要**检查尺子对被比的变量是否中立**。
> 结果是 dense 自查 56%→73%、等上下文预算下 fact 召回 0.749→0.779（三个题型齐涨）；而**端到端 A/B 是 +0.022 [−0.066,+0.105]，测不出差异，我不会说端到端变好了**——要分辨 0.03 需每型约 700 题。
> **改动发生在哪一层就在哪一层量；被推翻的旧结论我原文保留、只加订正标注**（`EXPERIMENTS.md` 实验 14/15/16/18 都有）。」

**如果还有时间，我会讲这个**：

> 「最后我把**评测集本身**也证伪了。项目一直用 MultiHop-RAG，名字里带 MultiHop——但我写了个探针实测：
> **99.3% 的题，光靠原问句就能直接够到含答案的那篇文档**，中间的"跳"可以整个跳过；99.3% 的问句
> 已经把每篇 gold 文章的出处点名了，不存在"先查 A 才知道要查 B"。**它测的是跨文档聚合，不是顺序推理。**
> 而且**连"名字是 bridge"的也不可信**——HotpotQA 的 bridge 子集结构上 90.3% 是桥接，行为上捷径率仍有 96.5%。
> 这一条把此前三件一直没解释通的事统一解释了：`n_search` 恒在 1.2、k 从 4 调到 64 不改变迭代次数、
> 强制查询分解收益为零。**不是实现问题，是题目不需要多跳。**
> 换到 MuSiQue（捷径率随跳数单调下降、构造上保证桥接、猜测下限≈0）之后，**一行代码没改，
> `n_search` 自己涨了 46%**。同一个"压缩 k"的旋钮，在 MuSiQue 上省掉三分之二上下文而覆盖不变，
> 在 MultiHop-RAG 上完全无效——**差别全在题目结构**。
> 所以我现在报任何调优结论，都连同'它在什么结构的任务上成立'一起报。」
