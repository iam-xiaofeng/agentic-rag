"""OpenAI 兼容 chat 模型 + .env 自动加载。

从原 agent.py 抽出的「模型 / 密钥」基础设施——纯检索流水线的**生成端**和 RAGAS 的
**裁判 LLM** 都复用它。（agentic 循环已随 pivot 移除，见 tag v1-agentic-comparison。）
"""

from __future__ import annotations

import os
from pathlib import Path

# 启动时把项目根 .env 读入 os.environ，好让 build_model() 读到真实密钥。放最前，早于任何读环境
# 变量的代码；缺 python-dotenv / .env 都安静跳过。
# 坑：某些执行环境（后台任务、部分 IDE 终端）会把变量导成空串 ""，而 load_dotenv(override=False)
# 把空串当“已设置”跳过 → 读不到 .env 真值。所以先清掉空串变量，让 .env 能填进来；非空的（命令行
# 临时覆盖）仍保留优先。
try:
    from dotenv import load_dotenv

    for _k in [_k for _k, _v in os.environ.items() if _v == ""]:
        del os.environ[_k]
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ModuleNotFoundError:
    pass

from langchain_openai import ChatOpenAI


def build_model() -> ChatOpenAI:
    """OpenAI 兼容 chat 模型 —— 适配 OpenAI / x.ai(grok) / 智谱(glm) 及各类 OpenAI 兼容网关。

    - 某些网关用 WAF 拦未知 User-Agent：用 RAG_USER_AGENT 伪装一个（如 "claude-code/2.1.214"）。
    - RAG_MAX_RETRIES / RAG_TIMEOUT 抵抗网关偶发的 5xx / 超时（如 Cloudflare 524：源站 120s 没返回）。
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
