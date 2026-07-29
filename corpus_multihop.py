"""把 MultiHop-RAG（yixuantt/MultiHopRAG）加载成 (docs, examples) 供 P3 评测用。

为什么选这个语料：609 篇新闻 + 2556 个问题，其 gold 证据被**故意**分散在 2~4 篇文章里。
与 7 篇的玩具库不同，这里单次检索**无法**捞出完整证据链 —— 这正是 agentic 改写该挣到 delta 的地方。

数据文件（下载一次，已 gitignore）：
  data/corpus.json       list of {title, body, url, source, category, ...}
  data/MultiHopRAG.json  list of {query, answer, question_type, evidence_list:[{title, fact, ...}]}
下载：
  curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json      -o data/corpus.json
  curl -sL https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json -o data/MultiHopRAG.json

我们用文章**标题**（evidence_list[i]["title"]）作为 gold 来源 id，这样检索到的 sources 和
gold sources 能对上，便于算 hit@k / coverage。
"""

from __future__ import annotations

import json
import os
import pathlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from eval_dataset import Example
from retriever import Doc

DATA = pathlib.Path(__file__).resolve().parent / "data"

# 递归切分：优先按段落(\n\n)→行(\n)→句(". ")→词切，凑到 CHUNK_SIZE 字符、重叠 CHUNK_OVERLAP。
# 比旧版定长滑窗尊重语义边界（不把证据句拦腰切断），也不再先把 \n\n 压平。
#
# ⚠️ 这两个数是**整条检索栈最上游的旋钮**，选错了下游怎么调都补不回来（实验17-18）：
#   - chunk 越大 → 句子级语义被稀释，dense 自查 top1 从 79%(300) 掉到 45%(1200)；
#   - chunk 越小 → 证据句被切断的比例上升，而这是**发生在检索之前的、不可恢复的损失**。
#   - 由此得到规则：**overlap 必须 ≥ 要检索的最小语义单元长度**（本语料证据句均长 157 字符）。
#
# 定档 600/150（实验19 逐层重建，每型 30 题 / 241 条 gold 证据句实测）：
#   证据句留存 三类全 100%（唯一做到的配置；1200 在 temporal 上是 98.7%，400 掉到 89~95%）；
#   dense「用证据句原文自查」top1 **56.0% → 73.0%**；BM25 全程 97~100% 不动。
#   overlap 再加到 200 无收益（72.6%），150 已 ≥ 证据句中位长 145。
#
# 可用环境变量临时覆盖（**shell 级**，用于新旧配置 A/B，不必改代码）：
#   RAG_CHUNK_SIZE=1200 RAG_CHUNK_OVERLAP=150 python eval_agentic.py ...
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 150))
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=_SEPARATORS, length_function=len,
    )


_SPLITTER = _splitter(CHUNK_SIZE, CHUNK_OVERLAP)


def _chunks(text: str, splitter=None) -> list[str]:
    """递归切块：尊重 \\n\\n / \\n / 句子边界（见 _SPLITTER 上方的注释）。"""
    return (splitter or _SPLITTER).split_text((text or "").strip())


def load_corpus(
    path: str | pathlib.Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Doc]:
    """609 篇文章 -> 片段。source = 文章标题（与 gold 证据标题对齐）。

    `chunk_size` / `chunk_overlap` 显式给出时覆盖模块默认值——供切分参数的对照实验用
    （`eval_rebuild.py`），不给就走 CHUNK_SIZE / CHUNK_OVERLAP。
    """
    path = pathlib.Path(path or DATA / "corpus.json")
    arts = json.loads(path.read_text(encoding="utf-8"))
    sp = _splitter(chunk_size or CHUNK_SIZE, chunk_overlap or CHUNK_OVERLAP) \
        if (chunk_size or chunk_overlap) else _SPLITTER
    docs: list[Doc] = []
    for a in arts:
        title = (a.get("title") or a.get("url") or "untitled").strip()
        for j, ck in enumerate(_chunks(a.get("body") or "", sp)):
            # 每个片段都拼上标题，让标题里的实体词在任何片段中都可被检索到
            docs.append(Doc(id=f"{title}#{j}", text=f"{title}\n{ck}", source=title))
    return docs


def _kind(question_type: str) -> str:
    # null_query = 语料里信息不足 -> 我们的 "negative"（应拒答）
    return "negative" if question_type == "null_query" else "multihop"


def load_examples(path: str | pathlib.Path | None = None) -> list[Example]:
    path = pathlib.Path(path or DATA / "MultiHopRAG.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: list[Example] = []
    for r in rows:
        kind = _kind(r.get("question_type", ""))
        gold = sorted({e.get("title", "").strip()
                       for e in (r.get("evidence_list") or []) if e.get("title")})
        answer = (r.get("answer") or "").strip()
        out.append(Example(
            question=r["query"],
            reference="" if kind == "negative" else answer,
            sources=[] if kind == "negative" else gold,
            kind=kind,
        ))
    return out
