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
import pathlib

from eval_dataset import Example
from retriever import Doc

DATA = pathlib.Path(__file__).resolve().parent / "data"


def _chunks(text: str, size: int = 1200, overlap: int = 150):
    """按字符窗口切块、带重叠（散文）。"""
    text = " ".join((text or "").split())
    if not text:
        return
    i = 0
    while i < len(text):
        yield text[i:i + size]
        if i + size >= len(text):
            break
        i += size - overlap


def load_corpus(path: str | pathlib.Path | None = None) -> list[Doc]:
    """609 篇文章 -> 片段。source = 文章标题（与 gold 证据标题对齐）。"""
    path = pathlib.Path(path or DATA / "corpus.json")
    arts = json.loads(path.read_text(encoding="utf-8"))
    docs: list[Doc] = []
    for a in arts:
        title = (a.get("title") or a.get("url") or "untitled").strip()
        for j, ck in enumerate(_chunks(a.get("body") or "")):
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


if __name__ == "__main__":  # 离线冒烟测试，无需模型
    docs = load_corpus()
    exs = load_examples()
    import collections
    print(f"语料：  {len(docs)} 个片段（来自 data/corpus.json）")
    print(f"样例：  {len(exs)} 条（{collections.Counter(e.kind for e in exs)}）")
    print(f"示例文档: [{docs[0].source[:50]}] {docs[0].text[:70]!r}")
    mh = next(e for e in exs if e.kind == "multihop")
    print(f"示例多跳: gold_sources={len(mh.sources)}  ans={mh.reference[:40]!r}")
    print(f"  q: {mh.question[:90]}")
