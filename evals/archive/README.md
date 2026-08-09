# `evals/archive/` —— 冻结的评测脚本

这里的脚本**不再是现行路径**，但**原样保留**（本项目一贯的做法：被推翻的东西留档 + 加订正标注，
不删除）。它们撑起过 EXPERIMENTS.md 里的实验 1-23，那些数字仍然算数——只是它们量的东西
后来被证明不是我们要量的东西。

归档的两个理由：

**① 绑死在一个已被证伪的评测集上。** 实验25 用确定性探针实测 MultiHop-RAG 的**捷径率 99.3%**
——99.3% / 92.2% / 83.9% 的问句已经把每篇 gold 文章的出处点名，不存在"先查 A 才知道要查 B"。
它测的是**跨文档聚合**，不是**顺序推理**。在它上面得到的"agentic 没优势 / 查询分解收益为零 /
k 从 4 到 64 不改变迭代次数"三条结论，都是数据集结构的产物，不是实现问题。

**② 量的是已退役的指标。** openevals 四轴（groundedness / retrieval_relevance / helpfulness /
correctness）从未提供过增量信息量；title 级 `context_recall` 被实验8-9 证明**方向会反**
（去重把它刷到 0.89、correctness 反降）。

| 脚本 | 撑起过 | 为什么下线 |
|---|---|---|
| `eval_rag.py` | 实验 2-3 单次流水线基线 | openevals 四轴退役；LangSmith 路径现行评测不再依赖 |
| `eval_common.py` | 上面那个的装配件 | 同上（只被归档脚本引用） |
| `eval_rescore.py` | 实验13 换裁判后重打存档 run | 裁判 prompt 已整体重写，旧口径无法对齐 |
| `langsmith_upload.py` | 把 MultiHop-RAG 传成 LangSmith 数据集 | 数据集已换 MuSiQue（走 HuggingFace 缓存） |
| `eval_answerability.py` | 地板/实测/天花板 | → **`evals/eval_ceiling.py`**（换评测集通用 + 加了"可达空间利用率"） |
| `eval_diag.py` | 实验8-9 title vs fact 级召回 | title 级代理已退役 |
| `eval_chain.py` | 实验17 全链路 L0-L3 排查 | → **`evals/eval_rebuild.py --layer 0/1/2/3`**（同样的自检 + 配对 CI） |
| `eval_rerank_effect.py` | 实验15/20 重排前后名次 | → 同上 `--layer 3` |
| `eval_query.py` | 实验22-23 查询分解检索侧 | 结论已落进 `rag/retriever_decompose.py`；端到端由 `run_matrix.py` 的 `decompose=1` 臂量 |
| `eval_ablation.py` | 实验6 RRF 参数消融 | 结论（`rrf_k`/权重在合理区间不敏感）已定档，不需要复跑 |

要跑的话它们仍然能跑（`python evals/archive/eval_diag.py`），只是别拿它们的数当现行结论。
