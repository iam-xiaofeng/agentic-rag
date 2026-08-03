"""agentic RAG loop —— langchain.agents.create_agent，复用 llm.py 的 build_model。

模型自己决定 **该不该查 / 查几次 / 何时停**，由 AGENTIC_RAG_SYSTEM 的五条策略约束，而非写死控制流。
（第 5 条 REPORT EVIDENCE 让它把「最关键的那几句」原样吐在末尾 —— 评测端据此判「依据够不够」，
见 eval_judge.py；`run_agentic.py` 交互时这块会直接显示给用户，本来也是该有的引用。）
这是和 run.py（单次强检索流水线）**正交**的另一条路；检索后端仍是同一套 Retriever 协议。

（pivot 时本文件曾移入 tag v1-agentic-comparison，现按需恢复并存；build_model 统一走 llm.py，不再各搞一份。
langchain 1.x：`create_agent` 取代旧的 `langgraph.prebuilt.create_react_agent`，`prompt` 参数改名 `system_prompt`。）
"""

from __future__ import annotations

from langchain.agents import create_agent

from rag.llm import build_model
from rag.prompts import AGENTIC_RAG_SYSTEM
from rag.retriever import Retriever
from rag.tools import make_rag_search


def build_agent(retriever: Retriever, model: str | None = None):
    """model →(rag_search → model)* → stop。何时查 / 查几次 / 何时停由模型在策略下自定。

    `model` 显式指定答题模型（缺省读 `RAG_MODEL`）——做模型对比时只换这里，裁判端不动。"""
    return create_agent(build_model(model), [make_rag_search(retriever)], system_prompt=AGENTIC_RAG_SYSTEM)
