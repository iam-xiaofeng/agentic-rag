"""**每个 run 产物必须能自证它是谁跑的** —— 配置快照 + 带 `__meta__` 头的 JSONL 读写。

    from rag.runctx import snapshot, DumpWriter, read_dump

━━━ 为什么有这个模块（2026-08 重构，根因之一）━━━

旧的 `runs/dumps/*.jsonl` 只有逐题结果，**不记模型 / 裁判 / topk / chunk / pool / 提示词版本**。
而这个项目有 8 个 env 旋钮（`RAG_MODEL` `RAG_JUDGE_MODEL` `RAG_TOPK` `RAG_POOL`
`RAG_CHUNK_SIZE` `RAG_CHUNK_OVERLAP` `RAG_DECOMPOSE` `MUSIQUE_SPLIT`）+ 提示词随 git 变。
于是"两个 dump 能不能配对比较"完全靠笔记和记忆。

实测这已经出过事：`.env` 里写着 `RAG_MODEL=grok-4.5`，而 EXPERIMENTS 实验26 记的是
`deepseek-v4-pro` —— 光看 `mq.jsonl` **无法判定它到底是谁跑的**，那一轮的三个结论因此不可复核。

所以：`__meta__` 是每个 dump 的**第一行**（不是 sidecar，sidecar 会和数据走散）。
`read_dump()` 把它和数据行分开返回；`eval_judge.py --baseline` 在配对前**自动比对两边的 meta**，
把「唯一变量」打印出来 —— 如果连裁判都不一样，直接拒绝出结论。

**绝不写入 api_key**：`snapshot()` 只记模型名，端点/密钥留在 gitignore 的 `endpoints.json` 里。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import threading
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).resolve().parents[1]
META_KEY = "__meta__"


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(("git", *a), cwd=_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:                                        # noqa: BLE001
            return ""
    return {"sha": run("rev-parse", "--short", "HEAD"),
            # dirty=true 意味着这次 run 用的代码**不在任何 commit 里** —— 复现要靠下面的 prompt_sha
            "dirty": bool(run("status", "--porcelain"))}


def prompt_fingerprint() -> dict:
    """系统提示词的指纹。**换提示词 = 换了被测对象**，跨版本分数不可比（prompts.py 文件头有警告）。

    react 记 `AGENTIC_RAG_SYSTEM`，planner 记它自己那三段的合并 sha —— 否则改了 planner 的
    extract 提示词，两个 run 在 meta 里会长得**一模一样**，配对时看不出自变量。
    """
    from rag import prompts
    txt = prompts.AGENTIC_RAG_SYSTEM
    plan = prompts.PLANNER_PLAN + prompts.PLANNER_EXTRACT + prompts.PLANNER_SYNTH
    return {"prompt_sha": hashlib.sha256(txt.encode()).hexdigest()[:12],
            "planner_prompt_sha": hashlib.sha256(plan.encode()).hexdigest()[:12],
            "prompt_chars": len(txt)}


def snapshot(**extra) -> dict:
    """当前进程的完整配置快照。`extra` 放调用方才知道的东西（benchmark / per_type / arm 名…）。"""
    from rag.llm import resolve
    from rag.retriever_dense import _MODEL as DENSE_MODEL
    from rag.retriever_hybrid import _POOL, _RERANKER
    from rag.tools import topk

    def role(r):
        try:
            return resolve(None, r)[0]
        except SystemExit:
            return None

    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git(),
        "answer_model": extra.pop("answer_model", None) or role("answer"),
        "judge_model": role("judge"),
        "agent": os.environ.get("RAG_AGENT", "react"),
        **prompt_fingerprint(),
        "retrieval": {
            # ⚠️ **必须调 topk() / 读 _POOL，不许在这里重写一遍默认值。**
            # 这里曾经写死 `os.environ.get("RAG_TOPK", 32)`，而 tools.topk() 的默认已改成 8 ——
            # 于是实际用 k=8 跑的 run，`__meta__` 里记的是 k=32。
            # **污染溯源信息比普通 bug 更坏**：它让"这个 dump 是什么配置跑的"这件事本身不可信，
            # 而整个配对比较装置都建立在它之上。同一个默认值只准有一处定义。
            "topk": topk(),
            "pool": _POOL,
            "chunk_size": int(os.environ.get("RAG_CHUNK_SIZE", 600)),
            "chunk_overlap": int(os.environ.get("RAG_CHUNK_OVERLAP", 150)),
            "dense": DENSE_MODEL,
            "reranker": _RERANKER,
            "decompose": os.environ.get("RAG_DECOMPOSE", "") in ("1", "true", "yes"),
        },
        **extra,
    }


# ── 决定「两个 dump 能不能配对比较」的字段 ──────────────────────────────
# 分两级：FATAL 不同 = 比的不是同一件事，拒绝出结论；VARIED 不同 = 那就是本次实验的自变量，打印出来。
_FATAL = ("benchmark", "judge_model")
_VARIED = ("answer_model", "agent", "prompt_sha", "planner_prompt_sha", "oracle",
           "benchmark",                       # 名字不同但语料相同时，降级为「自变量」打印出来
           "retrieval.topk", "retrieval.pool", "retrieval.chunk_size", "retrieval.decompose")


def _get(d: dict, path: str):
    for p in path.split("."):
        d = (d or {}).get(p) if isinstance(d, dict) else None
    return d


def _corpus_of(bench) -> str | None:
    """把 benchmark 名归一到**决定可比性的东西**：语料 + gold 来源。

    `musique` 与 `musique+1hop` 用**同一份 21100 段语料、同一个 gold 字段**，
    后者只是额外挂了由第 1 步派生的单跳题（题目集合是超集，按 example_id 配对时
    多出来的那些自然匹配不上）。**拿它们配对是合法的**，所以致命判据比对的应当是
    「语料是谁」而不是「benchmark 叫什么」。

    ⚠️ 这不是在放宽守卫：`musique` vs `multihoprag` 仍然致命。
    降级掉的那一档会作为**自变量**打印出来，读的人看得见。
    """
    return bench.split("+")[0] if isinstance(bench, str) else bench


def compare_meta(a: dict, b: dict) -> tuple[list[str], list[str]]:
    """→ (致命差异, 自变量差异)。给 `eval_judge.py --baseline` 在配对前自动体检用。

    `benchmark` 走 `_corpus_of()` 归一后再比：名字不同但语料相同（musique / musique+1hop）
    不算致命，会落到 varied 里被打印出来；语料真不同（musique vs multihoprag）仍然致命。
    """
    norm = {"benchmark": _corpus_of}
    def val(d, k):                                      # noqa: E306
        v = _get(d, k)
        return norm[k](v) if k in norm else v
    fatal = [f"{k}: {_get(a, k)!r} vs {_get(b, k)!r}"
             for k in _FATAL if val(a, k) != val(b, k)]
    varied = [f"{k}: {_get(b, k)!r} → {_get(a, k)!r}" for k in _VARIED if _get(a, k) != _get(b, k)]
    return fatal, varied


class DumpWriter:
    """**边跑边落盘**的 JSONL：第 1 行 `__meta__`，之后每题一行、立即 flush。

    旧版是全部跑完再一次性 `write_text` —— 90 题跑到第 80 题网关抽风就**全丢**。
    配 `resume_ids()` 就能接着跑，这是把 n 从 21 抬到 90+ 的前提（重构 Step 2）。
    线程安全：`eval_agentic.py` 用 ThreadPoolExecutor 并发跑题。
    """

    def __init__(self, path: str | pathlib.Path, meta: dict, resume: bool = False) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if resume and self.path.exists():
            self.done = resume_ids(self.path)
            self._fh = self.path.open("a", encoding="utf-8")
            return
        self.done = set()
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(json.dumps({META_KEY: meta}, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write(self, row: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_dump(path: str | pathlib.Path) -> tuple[dict, list[dict]]:
    """→ (meta, rows)。**兼容无 meta 的旧 dump**（meta 返回 {}，并在表头标注"旧格式"）。

    **同一 example_id 只保留最后一次出现（last-wins）。** `--resume` 是**追加**写：
    第一次失败的题会先落一行 errored，补跑成功后再追加一行完整结果，**两行都在文件里**。
    任何不去重的读取都会把那道题算两次，而 errored 那行没有 answer ——
    下游一律会把它兜底成 0 分。**那就是"网关失败被记成 0 分"，本项目栽过三次的同一个坑。**
    """
    meta: dict = {}
    seen: dict[str, dict] = {}
    rows: list[dict] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if META_KEY in d and len(d) == 1:
            meta = d[META_KEY]
        elif (eid := d.get("example_id")) is not None:
            if eid in seen:
                rows[seen[eid]] = d                  # 后来者覆盖，保持原始顺序
            else:
                seen[eid] = len(rows)
                rows.append(d)
        else:
            rows.append(d)
    return meta, rows


def resume_ids(path: str | pathlib.Path) -> set[str]:
    """已经跑成功过的 example_id（errored 的**不算**，会重跑）。"""
    _, rows = read_dump(path)
    return {r["example_id"] for r in rows if "errored" not in r and r.get("example_id")}


def fmt_meta(meta: dict) -> str:
    if not meta:
        return "（⚠️ 旧格式 dump，无 __meta__ 头：无法自证配置，结论请谨慎）"
    r = meta.get("retrieval", {})
    return (f"model={meta.get('answer_model')} judge={meta.get('judge_model')} "
            f"agent={meta.get('agent')} prompt={meta.get('prompt_sha')} "
            f"bench={meta.get('benchmark')} k={r.get('topk')} pool={r.get('pool')} "
            f"chunk={r.get('chunk_size')} decompose={r.get('decompose')} "
            f"git={meta.get('git', {}).get('sha')}{'+dirty' if meta.get('git', {}).get('dirty') else ''}")
