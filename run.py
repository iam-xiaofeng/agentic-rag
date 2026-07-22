"""CLI：提个问题，观察 agentic 检索过程。

    python run.py "What language is the database used by Nimbus written in?"

打印的 "rag_search #N" 行 + 末尾计数，让你直接**看见过程层**：检索了几次、怎么改写的、
什么时候决定停。

完整逐步 trace（每次工具调用、延迟、token）在设了 LANGSMITH_TRACING=true 和
LANGSMITH_API_KEY 时会出现在 LangSmith（见 .env.example）。
"""

from __future__ import annotations

import sys

from agent import build_agent


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Q: ").strip()
    agent = build_agent()

    n_search = 0
    for chunk in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
        msg = chunk["messages"][-1]
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] == "rag_search":
                n_search += 1
                print(f"  ↳ rag_search #{n_search}: {tc['args'].get('query')!r}")
        msg.pretty_print()

    print(f"\n[过程] 本次 rag_search 调用次数: {n_search}")


if __name__ == "__main__":
    main()
