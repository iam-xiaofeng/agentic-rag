"""评测公共件：openevals 的 LLM-as-judge 装配 + 确定性评估器。

eval_rag.py（单次流水线）和 eval_agentic.py（多跳）都从这里取同一套评测原语，
避免两处各写一份、口径漂移。

- LLM-as-judge（judge = 网关模型；prompt 来自 openevals，业界现成、非自研）：
    correctness / groundedness / retrieval_relevance / helpfulness
- 确定性评估器（不花 LLM，用上传到 LangSmith 的 gold）：
    context_recall  gold 文章标题 ∩ 检索到的 source ÷ gold 标题数
    refused         答案是否命中拒答线索词（null_query 该拒答）
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import re

from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

# 每个裁判的 prompt + 它需要的变量（决定 call_judge 传哪些 kwargs）。
SPECS = {
    "correctness": (CORRECTNESS_PROMPT, {"inputs", "outputs", "reference_outputs"}),
    "groundedness": (RAG_GROUNDEDNESS_PROMPT, {"outputs", "context"}),
    "retrieval_relevance": (RAG_RETRIEVAL_RELEVANCE_PROMPT, {"inputs", "context"}),
    "helpfulness": (RAG_HELPFULNESS_PROMPT, {"inputs", "outputs"}),
}

# 拒答线索词：答案命中任一即判为「拒答」。
# ⚠️ 必须**跨语言**：系统提示里混了中文轴标注（该不该查/查几次…），deepseek 等模型会据此整段用中文作答，
# 而 grok 用英文。只认英文线索词的话，中文拒答会被判成 refused=0 → null_query 看着像"全在幻觉"，
# 纯属度量假象（与实验12"失败必须可见"同类的坑：评估器本身不能对语言敏感）。
REFUSAL = ("don't know", "do not know", "not contain", "no information", "cannot find",
           "not available", "no relevant", "isn't specified", "is not specified",
           "does not specify", "not mention", "no answer",
           # 中文
           "不知道", "无法找到", "没有找到", "未找到", "没有提及", "未提及", "没有相关",
           "无相关", "未提供", "没有提供", "无法确定", "无法回答", "不包含", "没有包含",
           "检索不到", "语料中没有", "知识库中没有")


def make_judges(judge):
    """把 SPECS 里每个 prompt 装成一个 openevals LLM-judge（continuous=[0,1] + 打分理由）。"""
    return {
        name: create_llm_as_judge(prompt=prompt, feedback_key=name, judge=judge, continuous=True)
        for name, (prompt, _needs) in SPECS.items()
    }


def call_judge(name, fn, question, out, reference):
    """按该裁判所需变量装 kwargs 后调用，返回含 score/comment 的 dict。"""
    needs = SPECS[name][1]
    kw = {}
    if "inputs" in needs:
        kw["inputs"] = question
    if "outputs" in needs:
        kw["outputs"] = out["answer"]
    if "reference_outputs" in needs:
        kw["reference_outputs"] = reference
    if "context" in needs:
        kw["context"] = "\n\n".join(out["contexts"])
    return fn(**kw)


def context_recall(run, example) -> dict:
    """确定性 **title 级** context recall：gold 文章标题里有多少被检索到（不花 LLM）。
    ⚠️ 这是**粗代理**：只看「文章中没中」、不看「证据句到没到」，能被 source 去重刷高、方向还会反
    （EXPERIMENTS 实验8-9 实测：去重让 title 0.77→0.885 却把 fact 0.645→0.530）。要更准用 context_recall_fact。"""
    gold = set(example.outputs.get("gold_titles") or [])
    got = set(run.outputs.get("sources") or [])
    return {"key": "context_recall", "score": (len(gold & got) / len(gold)) if gold else None}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def context_recall_fact(run, example) -> dict:
    """确定性 **fact 级** context recall：gold 证据句（reference_contexts）真出现在检索文本里的比例。
    比 title 级准——量的是「证据到没到」，与 correctness 同向（实验8-9 的教训：盯它，别盯 title 代理）。"""
    facts = example.outputs.get("reference_contexts") or []
    blob = _norm(" ".join(run.outputs.get("contexts") or []))
    if not facts:
        return {"key": "context_recall_fact", "score": None}
    hit = sum(1 for f in facts if _norm(f)[:120] in blob)
    return {"key": "context_recall_fact", "score": hit / len(facts)}


def refused(run, example) -> dict:
    """确定性 refused：答案是否命中拒答线索词（null_query 该拒答，可答题里 refused=1 反而是漏答）。"""
    ans = (run.outputs.get("answer") or "").lower()
    return {"key": "refused", "score": 1.0 if any(c in ans for c in REFUSAL) else 0.0}
