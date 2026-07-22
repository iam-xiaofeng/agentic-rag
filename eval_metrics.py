"""确定性指标 —— 不额外花 LLM 成本，所以可复现、免费。

每个指标签名为 (example, output) -> [0,1] 的 float。`output` 是个 dict：
  {"answer": str, "sources": [检索到的来源 id], "n_search": int}

（生产环境会再加一个 LLM-judge 来判 faithfulness/answer-relevance；这里选启发式，
是为了让评测确定、可离线跑。）
"""

from __future__ import annotations

import re

from eval_dataset import Example

# 拒答线索词（匹配模型的英文输出，故保留英文）。
_REFUSAL_CUES = (
    "don't know", "do not know", "not contain", "no information", "doesn't specify",
    "does not specify", "cannot find", "not available", "not in the knowledge base",
    "no relevant", "not mention", "isn't specified", "is not specified",
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _cited(answer: str) -> set[str]:
    return set(re.findall(r"\[source:\s*([^\]]+)\]", answer or ""))


def refusal(ex: Example, out: dict) -> float:
    """答案是否明确拒答（而不是编造），是则 1.0。"""
    ans = _norm(out["answer"])
    return 1.0 if any(cue in ans for cue in _REFUSAL_CUES) else 0.0


def correctness(ex: Example, out: dict) -> float:
    """参考事实是否出现在答案里。对 negative，correct == 正确拒答。"""
    if ex.kind == "negative":
        return refusal(ex, out)
    return 1.0 if _norm(ex.reference) in _norm(out["answer"]) else 0.0


def faithfulness(ex: Example, out: dict) -> float:
    """引用合法性代理：答案里每个 [source: X] 都确实被检索到过。"""
    cited = _cited(out["answer"])
    if not cited:
        # 只有在「本就不该检索」时，没有引用才算合理
        return 1.0 if ex.kind in ("no_retrieve", "negative") else 0.0
    return 1.0 if cited <= set(out.get("sources", [])) else 0.0


def retrieval_hit(ex: Example, out: dict) -> float:
    """至少命中一个 gold 来源则 1.0（无 gold 则视为 n/a -> 1.0）。"""
    if not ex.sources:
        return 1.0
    return 1.0 if set(ex.sources) & set(out.get("sources", [])) else 0.0


def retrieval_discipline(ex: Example, out: dict) -> float:
    """过程层：该查时查、不该查时别查。
    no_retrieve -> 绝不能检索（过度检索 = 0）；其余 -> 必须检索。"""
    n = out.get("n_search", 0)
    if ex.kind == "no_retrieve":
        return 1.0 if n == 0 else 0.0
    return 1.0 if n >= 1 else 0.0


METRICS = {
    "correct": correctness,
    "faithful": faithfulness,
    "hit": retrieval_hit,
    "discipline": retrieval_discipline,
}
