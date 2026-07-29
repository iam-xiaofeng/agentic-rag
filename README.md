# 混合检索 RAG 流水线 · BM25 + bge 向量 + 交叉编码重排 · LLM-as-judge 评测

> 一个**面试可讲、可跑、可评测**的生产级 RAG **检索流水线**：
> **召回**（BM25 词法 + bge 稠密向量[Chroma]，RRF 融合）→ **重排**（bge-reranker 交叉编码）→ **生成**（grounded、带引用）
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

> **为什么两路召回要融合、还要重排**：词法和向量的盲区**互补**——RRF 融合先把两种「看法」的候选按排名并起来（提召回），
> 再用 cross-encoder 精排（提精度）。这就是 “hybrid retrieval + reranking” 成为生产标配的原因。

检索后端解耦在 `Retriever` 协议后面：当前实现 `BM25Retriever` / `DenseRetriever` / `HybridRetriever` 三个后端，
**同一套接口**，`rag.py`（生成）和 `eval_rag.py`（评测）对换后端无感。

---

## 二、目录结构（每个文件干什么）

```
agentic-rag/
├── README.md            # 本文件
├── EXPERIMENTS.md       # 18 次实验的设置 / 命令 / 数据 / 结论（LangSmith 可复现）
├── requirements.txt     # 依赖：sentence-transformers / chromadb / rank-bm25 / openevals / langchain-openai / langgraph / langsmith
├── .env.example         # 配置模板（复制成 .env 填真实值）
├── .env                 # 真实密钥（已 gitignore；模型 + LangSmith 凭据）
│
│  ── 检索栈（核心）：一个协议 + 三个后端 ──
├── retriever.py         # Retriever 协议 + Doc/Hit 数据类（后端地基）
├── retriever_bm25.py    # BM25Retriever：词法检索（rank_bm25，零重依赖）
├── retriever_dense.py   # DenseRetriever：bge 编码 → Chroma 持久化向量库（近邻检索；.cache/chroma/）
├── retriever_hybrid.py  # HybridRetriever：BM25+向量 RRF 融合 → bge-reranker 重排（★ 新核心）
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
├── corpus_multihop.py   # 加载 data/ 的 MultiHop-RAG（609 篇 → 6711 片段 + 2556 问）
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
  - `load_corpus()`：每篇 `body` 用 **递归切分**（`RecursiveCharacterTextSplitter`，优先 `\n\n`→`\n`→句子边界，1200 字符 / 重叠 150）切块、拼上标题 → **6711 个 `Doc`**（`source = 文章标题`，与 gold 证据标题对齐）。
  - `load_examples()`：每个问题 → 一条 `Example`（`question` / `reference=answer` / `sources=gold 标题` / `kind`）。

### 如何重新下载

```bash
cd agentic-rag && mkdir -p data
curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json      -o data/corpus.json
curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json -o data/MultiHopRAG.json
# 离线自检（无需模型）：
.venv/bin/python -c "from corpus_multihop import load_corpus, load_examples; print(len(load_corpus()), '片段', len(load_examples()), '例')"  # 6711 片段 2556 例
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

# 2) 跑 hybrid 检索（首次会下 bge 模型 + 编码 6711 片段，之后走 .cache 缓存）
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
#   DenseRetriever    bge-large-en-v1.5 向量 → Chroma 持久化库，余弦近邻（有 GPU 自动用 CUDA）
#   HybridRetriever   ↑两者 RRF 融合 → bge-reranker-v2-m3 重排
```

**HybridRetriever 的三步**（`retriever_hybrid.py`）：

1. **召回**：BM25、Dense 各取 top-`pool`(默认 100；见 EXPERIMENTS 实验 6，pool 是召回的主杠杆)。
2. **融合**：**RRF 倒数排名融合**（默认，`score = Σ w/(rrf_k+rank)`）——只看排名、不看分数，对 BM25(无上界) 与余弦([-1,1]) 不可比的量纲免疫（业界标配）；另留 min-max 加权(`fusion="minmax"`)做对照。默认权重均衡 `w_bm25 = w_dense = 0.5`。
3. **重排**：融合候选池丢给 `bge-reranker-v2-m3`（cross-encoder），按 (query, passage) 相关性取 top-k。

> 想调：`HybridRetriever(docs, fusion="rrf", w_dense=.7, pool=100, rrf_k=60)`；参数消融见 **EXPERIMENTS 实验 6**（`python eval_ablation.py`，免费、不调 LLM）。默认已用 `bge-large-en-v1.5`（1024 维）+ `bge-reranker-v2-m3`（GPU 上跑）；无显卡的机器可在 `retriever_dense.py` / `retriever_hybrid.py` 改回 `bge-small-en-v1.5` / `bge-reranker-base`。

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

> 完整的 18 次实验（设置 / 命令 / 数据 / 结论）都记在 **[`EXPERIMENTS.md`](EXPERIMENTS.md)**；这里只放两张最能打的表。

**① 瓶颈在检索召回、不在幻觉**（单次流水线 · `multihop-rag` n=12 · LangSmith `hybrid-on-multihop-rag-*`）：

| correctness | groundedness | retrieval_relevance | helpfulness | context_recall |
|---|---|---|---|---|
| 0.42 | **1.00** | 0.64 | 0.52 | **0.64** |

`groundedness` 满分 = 零幻觉；但 `correctness` 被 `context_recall`(0.64) 卡住——证据分散 2~4 篇，单次只捞 2/3，模型没证据就诚实拒答。（`topk` 4→8 可把召回 0.52→0.68，见 EXPERIMENTS 实验 3。）

**② agentic 多跳 · 100 题 · 按 question_type**（当前配置 RRF+Chroma+pool=100+k=8 · LangSmith `agentic-on-multihop-rag-d61482cc`）：

| type | correctness | groundedness | retrieval_rel | helpfulness | context_recall | refused |
|---|---|---|---|---|---|---|
| comparison | 0.80 | 0.74 | 0.60 | 1.00 | 0.85 | 0.00 |
| inference | **1.00** | 0.75 | 0.87 | 1.00 | 0.72 | 0.00 |
| temporal | 0.67 | 0.83 | 0.54 | 0.97 | 0.83 | 0.16 |
| null | – | 0.96 | 0.03 | 0.64 | – | **0.92** |

- **多跳 correctness ~0.82**（vs 单次 0.42）：inference 满分、temporal 最难、null 92% 正确拒答；groundedness 0.77。（表中 `context_recall` 是**标题级**口径，仅为与早期实验可比而保留——它是漏水的代理，见下条；现行评测已改用 fact 级 `context_recall_fact`，honest 值 ~0.63。）
- **一条踩过、也修好的坑（含金量高）**：曾用 source 去重把 title 级 `context_recall` 刷到 0.89，correctness 却掉了——fact 级诊断发现去重在**游戏漏水的代理指标**、丢了真证据（Goodhart）。撤销去重、只留 `k=8` 后 correctness 反升到 **0.82**（实验 8-9）。**教训：盯 correctness / fact 级召回，别优化 title 级代理。**
- **第二条坑（评测框架本身）**：`target()` 里一句 `except: return 空结果` 把网关 502 吞成空答案，被裁判打 **0 分并计入均值**，整轮实验读起来像"模型变差了"（实测 temporal 8 题空了 7 题）。修法：长退避重试 → 仍失败就 `raise`，让 LangSmith 标 **errored 并排除出均值**。**教训：评测里失败必须可见，不能降级成 0 分**（实验 12）。
- **第三条坑，也是最贵的一条：聚合指标把 reranker 的负增益藏了两个月**（实验 15）。全题型平均说"重排在帮忙 +0.088"，**分题型一看，bge 在 temporal 上是净减益 −0.100**——51% 的 gold 证据 chunk 被它**往后推**，中位名次从第 6 推到第 9，正好越过 k=8 的门槛。**教训：任何"平均是正的"都要按子群拆开验一遍，否则一个子群的系统性伤害会被另一个子群的收益掩盖。**
- **reranker 的价值必须分题型说**（实验 15，相对**不重排**的净值）：comparison **+0.244**、inference **+0.333**（Qwen3-4B 货真价实的大增益）；但 temporal 只有 **+0.011**（纯止损，天花板就是"什么都不做"的 0.522）。
- **第四条坑：chunk 对比没控制住「交付信息量」**（实验 18）。实验11 用固定 `fact@8` 比不同 chunk——8 个 1200 字符的块 = 9600 字符、8 个 300 字符的块只有 2400 字符，**信息量差 4 倍**，小块必输。按**总字符对齐**重测后结论反转：**~500 字符才是甜点**（重排后三类均值 0.670 vs 现状 1200 的 0.633），且 **dense 精度只跟「块长」有关、跟「递归/按结构切」无关**（纯段落1000 与递归1200 自查同为 60%）。**建议配置：chunk=600 + overlap=150 + pool=200 + k≈16 —— 上下文减半、召回 0.615→0.656。**
  > **300 更小反而更差**：证据句均长 157 字符，chunk=300 会**切断 11% 的证据句**（overlap 加到 150/200 也只救到 97%，块数却涨到 1200 的 5 倍）——**这是发生在检索之前的、不可恢复的损失**。由此得到一条可复用规则：**`chunk_overlap` 必须 ≥ 要检索的最小语义单元长度**。
- **最贵的一条：embedding 在最简单的任务上只有 45% 命中率**（实验 17）。**用证据句的原文去检索包含该原文的 chunk**，dense 只有 45~63% 排到第 1，而 BM25 是 94~100%。排除了 Chroma（精确余弦与 HNSW 结果 **100% 一致**）后定位到 **chunk 1200 字符稀释了句子级语义**：chunk 缩到 300，dense 自查 top1 **45%→79%**，而 BM25 全程 98~100% 不动。**这个硬伤从项目第一天就在，被所有端到端指标漏掉了**——对症解法是 **small-to-big**（小块建索引检索 → 映射回父块交付，@4 +0.046）。
  > **教训：端到端指标必须配一组「单点能力自检」**——给每个部件最有利的输入，看它能否接近满分。否则某个部件长期半残，整体指标只会显示「就这样了」，不会告诉你是谁拖的。
- **上游优化空间 ≈ 0，reranker 是唯一决定者**（实验 16）：**纯 BM25 单路 fact@8 0.522、纯 dense 0.344，相差 0.178——经过 reranker 后两者都变成 0.422**，`w_dense` 从 0 到 1 六个取值 temporal 交付一模一样。融合也从未提升交付（三类里持平或**低于 BM25 单路**），只提升覆盖；temporal 上它连覆盖都是负的（弱腿把强腿的好候选挤出 pool）。**pool/权重/融合方式只要不改覆盖，都会被下游抹平。**
- **真正的出路在查询侧，不在排序侧**（实验 14）：**查询分解到事实层 + 每个子问句各自重排**，temporal fact@8 **0.578**——**唯一越过"不重排"天花板的做法**，且只花 1 次 LLM 调用、仍用便宜的 bge。对照：同义改写（MultiQueryRetriever 式）把池覆盖从 0.867 抬到 0.978，**交付却一位小数都没动**——扩大召回的收益被重排 100% 吃掉。


> **一句话：多跳不是万能钥匙，是"用忠实度和成本换覆盖与正确率"的可量化选择。** 全部实验在 LangSmith 数据集 `multihop-rag` 上可复现，细节见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

---

## 八、路线图

- **混合检索（✅）**：BM25 + bge dense 加权融合 + bge-reranker 重排，同一 `Retriever` 协议。
- **LLM-judge 评测（✅）**：openevals 四轴（correctness / groundedness / retrieval_relevance / helpfulness），judge=网关模型，可 `--upload` 到 LangSmith。
- **检索调优（✅）**：RRF 融合 + 参数消融（实验 6）——**pool 20→100 是主杠杆**，`rrf_k` / 权重在合理区间不敏感；瓶颈转移到 reranker 精度 + top-k。
- **更强 reranker（✅，评测级）**：`reranker_qwen.py` 提供 instruction-aware 的 **Qwen3-Reranker-4B**（`eval_agentic.py --reranker qwen`），fact 级交付显著优于 bge；代价 ~13s/次重排 + ~8GB 显存，故**默认仍是 bge**、4B 作评测/对照用（实验 11-12）。
- **下一步不在检索侧**：池子加到 400（覆盖 100%）也换不来交付提升；temporal 的瓶颈是**查询-证据对不齐**，要做的是**查询分解 / 实体+日期结构化过滤**（实验 12）。
- **答题模型可换、裁判要固定（✅）**：`eval_agentic.py --model <名>` 只换答题端，裁判走 `llm.build_judge()`（`RAG_JUDGE_MODEL`）不动；裁判模型下线时用 `eval_rescore.py` 拿新裁判把**存档 run** 统一重打分，旧实验照样可比（实验 13）。
- **向量库（✅）**：`DenseRetriever` 已从内存 numpy 换成 **Chroma** 持久化向量库（`.cache/chroma/`，近邻检索）——同 `Retriever` 协议、上层一行不动；指标与 numpy 持平（HNSW 在此规模近似即精确）。
- **agentic RAG（并存 ✅）**：`run_agentic.py`（create_agent，四策略），与流水线共用 `Retriever` 协议；最初的「agentic vs 单次」对比在 tag `v1-agentic-comparison`。
- **历史归档**：自研确定性指标 + agentic-vs-单次对比脚本仍在 tag `v1-agentic-comparison`（未搬回 master）。

---

## 九、一句话面试话术

> 「RAG 的上限在检索，不在生成。所以我搭了一条生产级检索栈：**BM25 词法 + bge 向量（Chroma 库）RRF 融合**把两种召回的盲区互补，
> 再用 **cross-encoder 重排**精排——召回和精度分两步拿。后端全在一个 `Retriever` 协议后面可替换。评测我用 **openevals 的
> LLM-as-judge**（correctness / 忠实度 / 检索相关性 / helpfulness），裁判走网关模型，一键推 **LangSmith** 做实验追踪。
> 这个项目还**并存**着一个 agentic RAG（`run_agentic.py`：模型自己决定查不查、多跳、何时停）——**我知道 agentic 和强检索是两条正交、互补的路，也知道各自该怎么量化。**」
