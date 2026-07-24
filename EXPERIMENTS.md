# 实验记录（EXPERIMENTS）

所有实验的**设置 / 命令 / 数据 / 结论**。数据集统一是 **MultiHop-RAG**（2556 题；见 README 第三节），已整份传到 LangSmith 数据集 `multihop-rag`（含 gold 答案 + gold 文章标题 + evidence 摘录）。

评测口径统一：
- **LLM-as-judge**（openevals，judge = grok 网关）：`correctness` / `groundedness`(忠实度) / `retrieval_relevance` / `helpfulness`
- **确定性**：`context_recall` = gold 文章标题 ∩ 检索到的 source ÷ gold 标题数；`refused` = 答案是否命中拒答线索词

> ⚠️ 读数须知：LLM-judge **单裁判（grok）、不完全可复现**；小样本波动大。下面看**方向和量级**，别抠绝对值。所有实验对同一套指标口径，横向可比。

---

## 实验 1 · 模型选型：bge-small vs bge-large（n=6）

**问题**：升级到更大的向量/重排模型，指标会更好吗？
**命令**：`python eval_rag.py --n 6`（分别在两组模型下）

| 配置 | correctness | groundedness | retrieval_relevance | helpfulness |
|---|---|---|---|---|
| bge-small-en-v1.5 + reranker-base | 0.67 | 1.00 | 0.72 | 0.67 |
| **bge-large-en-v1.5 + reranker-v2-m3**（现默认，GPU） | 0.49 | 0.98 | 0.65 | 0.48 |

**结论**：n=6 上大模型**没有可辨识的提升**——差异全在 LLM-judge 噪声 + 小样本内（一道题翻转就拉动均值 0.15）。**6 个样本不能给模型选型下结论**（要 n≥20~30）。当前仍默认大模型（benchmark 上更强，且有 GPU）。

---

## 实验 2 · 单次强检索流水线 baseline（`multihop-rag`，n=12）

**问题**：hybrid(BM25+bge+reranker) + 单次生成，在真实数据集上什么水平？
**命令**：`python eval_rag.py --dataset multihop-rag --n 12`
**LangSmith experiment**：`hybrid-on-multihop-rag-be7e9a92`

| correctness | groundedness | retrieval_relevance | helpfulness | context_recall |
|---|---|---|---|---|
| 0.42 | **1.00** | 0.64 | 0.52 | **0.64** |

**结论**：`groundedness` 满分 = **零幻觉**；但 `correctness`(0.42) 被**检索召回**(`context_recall` 0.64)卡住——难多跳证据分散在 2~4 篇，单次只捞到约 2/3，模型**没证据就诚实拒答**（"I don't know"）。瓶颈在检索，不在生成。确定性 `context_recall`(0.64) ≈ LLM-judge `retrieval_relevance`(0.64)，两口径互印证。

---

## 实验 3 · topk 消融（context_recall，纯检索、**免费**不调 LLM）

**问题**：单次检索的召回瓶颈，靠调大 topk 能补多少？
**方法**：30 道多跳（gold 平均 **2.6 篇/题**），只算 context_recall。

| topk | context_recall |
|---|---|
| 4（原默认） | 0.52 |
| 8 | **0.68**（+0.16） |
| 12 | 0.70 |
| 20（= 候选池 pool 顶） | 0.75 |

**结论**：`topk=4` 对"平均要 2.6 篇"的多跳太小 → 元凶。→8 一步 +0.16；8 往上收益骤减，`pool=20` 是天花板。**这只是权宜之计**——见实验 4。

---

## 实验 4 · 单次 vs agentic 多跳（**受控**对比，context_recall）

**问题**：rerank 排的"最相关 top-k"经常漏掉多跳需要的另一篇 gold（它排到 5-8 甚至池外）。多跳改写检索能不能捞回单次**任何 topk 都够不着**的 gold？
**方法**：同 5 道多跳，**同一个 hybrid 检索器**，唯一变量 = 单查询 vs 多跳（把所有 hop 检索 source 取并集）。

| | sp@4 | sp@8 | sp@20（池顶） | **agentic** | 平均跳数 |
|---|---|---|---|---|---|
| 均值 | 0.45 | 0.60 | 0.65 | **0.73** | 4.0 |

- **agentic 0.73 > 单次池顶 0.65**：多跳**越过了单次的天花板**——改写后的 query 把不同片段拉进不同候选池。实锤:Will Smith 那道单次卡 0.50、**多跳 1.00**；Prime 那道单次卡 0.50、**多跳 0.75**（都捞回了池外 gold）。
- **但多跳不免费也不稳赢**：Engadget 那道单次 top-4 本就 1.00，agent 自作主张改写 query → **0.67（更差，query drift）**；Between 那道 **8 跳仍 0.50**（有些精确时序证据改写多少次都够不着）；平均 **4 跳 = ~4× 调用成本**。

**结论**：pointwise rerank 管"相关性"、**管不了多跳的证据覆盖**；多跳用**成本 + 忠实度**换覆盖。

---

## 实验 5 · agentic RAG × 100，按 question_type

**问题**：agentic 多跳在**不同题型**上分别什么表现？
**命令**：`python eval_agentic.py --per-type 25`（comparison / inference / temporal / null_query 各 25 = 100）
**LangSmith experiment**：`agentic-on-multihop-rag-99689380`（0 报错）

| type | correctness | groundedness | retrieval_relevance | helpfulness | **context_recall** | **refused** |
|---|---|---|---|---|---|---|
| comparison (25) | 0.81 | 0.74 | 0.61 | 1.00 | **0.75** | 0.08 |
| inference (25) | **1.00** | 0.71 | 0.86 | 1.00 | 0.58 | 0.00 |
| temporal (25) | 0.58 | 0.73 | 0.53 | 0.96 | 0.69 | 0.24 |
| null (25) | –（无 gold） | 0.95 | 0.03 | 0.35 | – | **0.84** |

- 🟢 **inference 强项**：correctness **1.00**。跨文档推断，agent 多跳凑证据 + 推理最吃得开。
- 🟢 **comparison 稳**：correctness 0.81、context_recall 0.75（四类最高）。
- 🔴 **temporal 最难**：correctness 0.58、retrieval_relevance 0.53、24% 被迫拒答——精确时序 + 跨文档比较，检索与推理双重吃力。
- 🟢 **null 拒答到位**：**84% 正确拒答**（认出"语料里没有"）；剩 16% 硬编 = 残余幻觉风险。

---

## 横向结论

1. **多跳把 correctness 拉起来**：三类可答题均值 correctness ≈ **0.80**（实验 5）vs 单次 **0.42**（实验 2）——方向性大赢（两次抽样非严格同批；严格受控见实验 4 的 context_recall 0.73 > 0.65）。
2. **代价：groundedness ~1.00（单次）→ ~0.73（多跳）**。多跳把多篇证据揉一起合成、断言更多 → 不如单次"贴着原文一句话"忠实。**收益与风险都被模型能力放大**（这条从项目开头到现在一以贯之）。
3. **context_recall 暴露了 rerank 的盲区**：相关性排序 ≠ 证据覆盖；`context_recall`(标题级)量得到覆盖缺口，但量不到"多跳把该跳的那块具体证据捞出来 + 在拼起来的上下文上推理"的增益——所以多跳 correctness 远高于 context_recall 的涨幅暗示的。
4. **该按题型选策略**：inference/comparison 上 agentic 值；temporal 是硬骨头（需更强时序/结构化检索）；null 上 agentic 已能 84% 诚实拒答。

> 一句话：**多跳不是万能钥匙,是"用忠实度和成本换覆盖与正确率"的一个可量化选择。** 每个结论都有同一套 LLM-judge + context_recall 的实测数字撑着，实验都在 LangSmith 数据集 `multihop-rag` 上可复现。
