"""MuSiQue 语料 + 题目 —— 换掉 MultiHop-RAG 的那个"真多跳"评测集。

    from rag.corpus_musique import load_corpus, load_questions

**为什么换**（实验25/26，`eval_benchmark_probe.py` 实测）：MultiHop-RAG 名字里的 "MultiHop"
指**证据跨多篇文章**，不指**推理要分步**。三边同口径探针的结果：

    数据集                  捷径率*    一次拿全    最高频答案占比
    MultiHop-RAG 全部       99.3%     76.5%     34.5%
    HotpotQA bridge         96.5%     95.7%      0.4%     ← 名字是 bridge，行为不是
    MuSiQue 2hop            79.6%     78.6%      0.8%
    MuSiQue 3hop            81.8%     58.3%      4.0%
    MuSiQue 4hop            69.0%     33.3%     12.1%     ← 唯一真需要多跳的子集

    * 捷径率 = 光靠原问句就够到了「含答案那篇」→ 中间的跳可以整个跳过。

MuSiQue 站得住的三条理由：① 捷径率随跳数**单调下降**（另两个数据集没有这个梯度）；
② 100% 的题子问句里字面写着 `#1`（前一跳答案的占位符）—— **构造上保证**的桥接；
③ 猜测下限几乎为 0（最高频答案 1.4%），而 MultiHop-RAG 是 34.5%。

⚠️ 语料是**所有题目的候选段落取并集**（含官方干扰项），不是只放 gold —— 只放 gold 等于
把检索题变成阅读题，分数会虚高。段落按 (title, text) 去重。
"""

from __future__ import annotations

import functools
import hashlib
import os

from rag.retriever import Doc

_SPLIT = os.environ.get("MUSIQUE_SPLIT", "validation")
_HF = os.environ.get("MUSIQUE_HF", "dgslibisey/MuSiQue")


@functools.lru_cache(maxsize=2)
def _raw(split: str = _SPLIT):
    from datasets import load_dataset
    return list(load_dataset(_HF, split=split))


@functools.lru_cache(maxsize=2)
def load_corpus(split: str = _SPLIT) -> list[Doc]:
    """所有题目的候选段落取并集，去重后当语料。

    段落本身就是自然的检索单元（维基一段），**不再二次切块** —— MultiHop-RAG 那边要切是因为
    源文档是整篇新闻。`source` 用段落标题（= 维基实体名），与 gold 的对齐单位一致。
    """
    seen: dict[str, Doc] = {}
    for r in _raw(split):
        for p in r["paragraphs"]:
            text = (p.get("paragraph_text") or "").strip()
            title = (p.get("title") or "").strip()
            if not text:
                continue
            key = hashlib.md5(f"{title}\n{text}".encode()).hexdigest()[:16]
            seen.setdefault(key, Doc(id=key, text=f"{title}. {text}", source=title))
    return list(seen.values())


class _Ex:
    """与 `eval_agentic.py` 的 LangSmith example 同形，好让评测那边一行都不用改。

    `reference_contexts` 放**每一跳的支撑段落原文**——确定性召回就是量"这些段落到没到"。
    """

    __slots__ = ("id", "inputs", "outputs")

    def __init__(self, r: dict):
        para = {p["idx"]: p for p in r["paragraphs"]}
        facts, titles = [], []
        for h in r["question_decomposition"]:
            p = para.get(h.get("paragraph_support_idx"))
            if p:
                facts.append((p.get("paragraph_text") or "").strip())
                titles.append((p.get("title") or "").strip())
        self.id = r["id"]
        self.inputs = {"question": r["question"]}
        self.outputs = {"reference": r["answer"], "gold_titles": sorted(set(titles)),
                        "reference_contexts": facts}


def load_questions(split: str = _SPLIT) -> tuple[list[_Ex], dict[str, str]]:
    """返回 (examples, {question: "2hop"|"3hop"|"4hop"})。只要 answerable 的题。

    题型直接用**跳数**——它就是这个数据集的难度轴，且 4hop 是唯一"一次检索只有 1/3 能拿全"的子集。
    """
    ex, qtype = [], {}
    for r in _raw(split):
        if not r.get("answerable"):
            continue
        e = _Ex(r)
        if not e.outputs["reference_contexts"]:
            continue
        ex.append(e)
        qtype[r["question"]] = f"{len(r['question_decomposition'])}hop"
    return ex, qtype


if __name__ == "__main__":
    c = load_corpus()
    ex, qt = load_questions()
    from collections import Counter
    print(f"语料 {len(c)} 段（去重后），题 {len(ex)} 道 {Counter(qt.values()).most_common()}")
    print(f"\n样例段落: {c[0].source!r}\n  {c[0].text[:150]}…")
    e = ex[0]
    print(f"\n样例题: {e.inputs['question']}")
    print(f"  标准答案: {e.outputs['reference']}")
    print(f"  gold 段落 {len(e.outputs['reference_contexts'])} 段，来自 {e.outputs['gold_titles']}")
