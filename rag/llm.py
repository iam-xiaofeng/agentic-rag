"""OpenAI 兼容 chat 模型 —— **模型名 → 端点注册表**（`endpoints.json`）。

    from rag.llm import build_model, build_judge, resolve
    build_model("gpt-5.6-luna")     # 自动查到它属于哪个网关、用哪把 key

━━━ 为什么从「一个 base_url + 一把 key」改成注册表（2026-08 重构）━━━

旧版把 `OPENAI_BASE_URL` / `OPENAI_API_KEY` 当全局单例。做**多模型对照**时这个假设直接塌：
实验24 的两个臂（deepseek / gpt-5.6-sol）分属两个网关、两把 key，当时只能在命令行临时
导环境变量跑 —— 而 dump 里又**不记模型名**，事后无法判定 `runs/dumps/mq*.jsonl` 到底是谁跑的。
（实测：`.env` 写着 `RAG_MODEL=grok-4.5`，而 EXPERIMENTS 记的是 deepseek-v4-pro，对不上。）

注册表把「模型名」变成唯一入口：base_url / key 由它查，`rag/runctx.py` 记进每个 run 产物的
`__meta__` 头。**凡是撑起统计主张的 run，必须能从产物本身读出它是谁跑的。**

`endpoints.json` 已 gitignore；模板见 `endpoints.example.json`。仍支持退回 `OPENAI_*` 环境变量
（注册表里查不到的模型走这条老路），所以旧命令不会突然跑不起来。
"""

from __future__ import annotations

import functools
import json
import os
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# 启动时把项目根 .env 读入 os.environ。放最前，早于任何读环境变量的代码；缺 python-dotenv / .env 都安静跳过。
# 坑：某些执行环境（后台任务、部分 IDE 终端）会把变量导成空串 ""，而 load_dotenv(override=False)
# 把空串当"已设置"跳过 → 读不到 .env 真值。所以先清掉空串变量，让 .env 能填进来；非空的（命令行
# 临时覆盖）仍保留优先。
try:
    from dotenv import load_dotenv

    for _k in [_k for _k, _v in os.environ.items() if _v == ""]:
        del os.environ[_k]
    load_dotenv(_ROOT / ".env")
except ModuleNotFoundError:
    pass

from langchain_openai import ChatOpenAI


@lru_cache(maxsize=1)
def _registry() -> dict:
    p = _ROOT / "endpoints.json"
    if not p.exists():
        return {"_roles": {}, "models": {}}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"_roles": d.get("_roles", {}), "models": d.get("models", {})}


def known_models() -> list[str]:
    """注册表里声明过端点的模型名（`--model` 的候选、`run_matrix.py` 的臂来源）。"""
    return sorted(_registry()["models"])


def resolve(model: str | None, role: str = "answer") -> tuple[str, str, str]:
    """(模型名, base_url, api_key) —— 单一解析入口，**记 meta 和建模型走的是同一条路**。

    优先级：显式 model → 环境变量（answer 读 `RAG_MODEL`，judge 读 `RAG_JUDGE_MODEL`）
    → `endpoints.json` 的 `_roles[role]`。端点先查注册表，查不到退回 `OPENAI_BASE_URL/KEY`。
    """
    reg = _registry()
    env = "RAG_JUDGE_MODEL" if role == "judge" else "RAG_MODEL"
    name = model or os.environ.get(env) or reg["_roles"].get(role) or os.environ.get("RAG_MODEL")
    if not name:
        raise SystemExit(
            f"没有指定{role}模型：给 --model，或设 {env}，或在 endpoints.json 的 _roles.{role} 里填一个。")
    ep = reg["models"].get(name, {})
    base = ep.get("base_url") or os.environ.get("OPENAI_BASE_URL") or ""
    key = ep.get("api_key") or os.environ.get("OPENAI_API_KEY", "sk-noop")
    if not ep and not os.environ.get("OPENAI_BASE_URL"):
        raise SystemExit(
            f"模型 {name!r} 不在 endpoints.json 里，也没有 OPENAI_BASE_URL 兜底。\n"
            f"已注册：{known_models() or '(空)'}")
    return name, base, key


def _chat(cls, model: str | None, role: str) -> ChatOpenAI:
    """网关适配：RAG_USER_AGENT 绕 WAF（某些中转拦未知 UA）；
    RAG_MAX_RETRIES / RAG_TIMEOUT 抵抗偶发 5xx / 超时（如 Cloudflare 524：源站 120s 没返回）。

    **RAG_RPS：进程级限流（每秒请求数），这是治 429 的正确工具。**
    429 的含义是"你发太快了"，所以解法是**发慢点**，不是重试更狠 ——
    实测中转网关在并发 3 下仍频繁 429，而任务级重试会把整题（含已成功的几次检索）
    **全部重发**，反而制造更多负载、把限流推向恶性循环。
    `InMemoryRateLimiter` 在**所有线程之间**共享一个令牌桶，从源头把速率压住。
    缺省不开（`RAG_RPS` 未设 = 不限流），只有被网关限流时才需要。
    """
    name, base, key = resolve(model, role)
    headers: dict[str, str] = {}
    ua = os.environ.get("RAG_USER_AGENT")
    if ua:
        headers["User-Agent"] = ua
    return cls(
        model=name,
        base_url=base or None,
        api_key=key,
        temperature=0,
        max_retries=int(os.environ.get("RAG_MAX_RETRIES", "5")),
        timeout=float(os.environ.get("RAG_TIMEOUT", "180")),
        default_headers=headers or None,
        rate_limiter=_rate_limiter(),
    )


@functools.lru_cache(maxsize=1)
def _rate_limiter():
    """**全进程共享一个**令牌桶——必须 lru_cache，否则每个 ChatOpenAI 各限各的，等于没限。"""
    rps = os.environ.get("RAG_RPS", "").strip()
    if not rps:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter
    r = float(rps)
    print(f"[限流] 全进程 {r} 请求/秒（RAG_RPS）", flush=True)
    # max_bucket_size=1：不允许攒额度后突发——突发正是触发 429 的那一下。
    return InMemoryRateLimiter(requests_per_second=r, check_every_n_seconds=0.1, max_bucket_size=1)


def build_model(model: str | None = None) -> ChatOpenAI:
    """**答题**模型。做模型对比时只换这里，裁判端不动 —— 否则分数变化分不清是"答得不同"还是"判得不同"。"""
    return _chat(ChatOpenAI, model, "answer")


class _FunctionCallingChat(ChatOpenAI):
    """把结构化输出强制走 **function_calling**，而不是 langchain 默认的 `response_format: json_schema`。

    为什么需要：openevals 的裁判内部调 `judge.with_structured_output(schema)`；ChatOpenAI 默认用
    `json_schema` 响应格式，**本项目的网关不支持**——模型于是用**散文**给结论（"该回答正确。依据…"），
    openevals 解析失败，**四个 LLM-judge 全军覆没、只剩确定性指标**（实测 deepseek 全系如此）。
    而 function_calling 在同一网关上是好的（agent 的 rag_search 就靠它），改走这条路即可兑现结构化。
    """

    def with_structured_output(self, schema, **kwargs):  # type: ignore[override]
        kwargs["method"] = "function_calling"
        kwargs.pop("strict", None)                        # 网关不认 strict schema
        return super().with_structured_output(schema, **kwargs)


def build_judge(model: str | None = None) -> ChatOpenAI:
    """**裁判**模型（LLM-as-judge 专用）：同 build_model，但结构化输出走 function_calling。

    **做模型对比时把裁判固定成同一个**，否则新旧分数不可比；裁判模型下线导致旧实验无法复算时，
    用 `evals/eval_rescore.py` 拿新裁判把存档 run 统一重打一遍（见 EXPERIMENTS 实验13）。
    """
    return _chat(_FunctionCallingChat, model, "judge")


if __name__ == "__main__":   # 自检：`python -m rag.llm` 逐个 ping 注册表里的模型
    import sys
    for m in known_models():
        n, b, k = resolve(m)
        try:
            r = build_model(m).invoke("Reply with exactly: OK")
            print(f"  ✅ {m:32s} {b:40s} → {(r.content or '')[:40]!r}")
        except Exception as e:                                        # noqa: BLE001
            print(f"  ❌ {m:32s} {b:40s} → {type(e).__name__}: {str(e)[:90]}", file=sys.stderr)
