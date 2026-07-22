"""构建 agentic RAG agent —— 一个 LangGraph ReAct 循环。

create_react_agent 提供了循环：model →(rag_search → model)* → stop。
关键在于：**何时检索、检索几次、何时停止**，都由模型在 AGENTIC_RAG_SYSTEM 策略约束下
自行决定，而不是写死的控制流。这正是「agentic RAG」区别于固定 retrieve-then-generate
流水线的地方。
"""

from __future__ import annotations

import os
from pathlib import Path

# 启动时把项目根 .env 读入 os.environ，好让下面 build_model() 读到真实密钥。
# 放在最前，早于任何读环境变量的代码；缺 python-dotenv / .env 都安静跳过
# （零依赖脚本 retriever.py 等不受影响）。
# 坑：某些执行环境（后台任务、部分 IDE 终端）会把变量导成空串 ""，而
# load_dotenv(override=False) 会把空串当“已设置”跳过 → 读不到 .env 真值。
# 所以先清掉空串变量，让 .env 能填进来；非空的（命令行临时覆盖）仍保留优先。
try:
    from dotenv import load_dotenv

    for _k in [_k for _k, _v in os.environ.items() if _v == ""]:
        del os.environ[_k]
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ModuleNotFoundError:
    pass

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from prompts import AGENTIC_RAG_SYSTEM
from retriever import InMemoryRetriever, Retriever
from tools import make_rag_search


def build_model() -> ChatOpenAI:
    """OpenAI 兼容 chat 模型 —— 适配 OpenAI / x.ai(grok)/ 智谱(glm) 及各类 OpenAI 兼容网关。

    - 某些网关会用 WAF 拦截未知 User-Agent：用 RAG_USER_AGENT 伪装一个（如 "claude-code/2.1.214"）。
    - RAG_MAX_RETRIES / RAG_TIMEOUT 抵抗网关偶发的 5xx / 超时（例如 Cloudflare 524：源站
      120s 内没返回）。默认多重试几次，让单次抖动不至于整轮失败。
    """
    headers: dict[str, str] = {}
    ua = os.environ.get("RAG_USER_AGENT")
    if ua:
        headers["User-Agent"] = ua
    return ChatOpenAI(
        model=os.environ.get("RAG_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        api_key=os.environ.get("OPENAI_API_KEY", "sk-noop"),
        temperature=0,
        max_retries=int(os.environ.get("RAG_MAX_RETRIES", "5")),
        timeout=float(os.environ.get("RAG_TIMEOUT", "120")),
        default_headers=headers or None,
    )


def build_agent(retriever: Retriever | None = None):
    retriever = retriever or InMemoryRetriever()
    tools = [make_rag_search(retriever)]
    return create_react_agent(build_model(), tools, prompt=AGENTIC_RAG_SYSTEM)
