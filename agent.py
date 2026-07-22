"""agentic RAG loop —— create_react_agent，复用 llm.py 的 build_model。

模型自己决定 **该不该查 / 查几次 / 何时停**，由 AGENTIC_RAG_SYSTEM 的四条策略约束，而非写死控制流。
这是和 run.py（单次强检索流水线）**正交**的另一条路；检索后端仍是同一套 Retriever 协议。

（pivot 时本文件曾移入 tag v1-agentic-comparison，现按需恢复并存；build_model 统一走 llm.py，不再各搞一份。）
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from llm import build_model
from prompts import AGENTIC_RAG_SYSTEM
from retriever import Retriever
from tools import make_rag_search


def build_agent(retriever: Retriever):
    """model →(rag_search → model)* → stop。何时查 / 查几次 / 何时停由模型在策略下自定。"""
    return create_react_agent(build_model(), [make_rag_search(retriever)], prompt=AGENTIC_RAG_SYSTEM)
