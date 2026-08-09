"""agentic RAG 的**两条控制流**，共用一个 runner 接口。

    from rag.agent import build_runner
    run = build_runner(retriever, model="gpt-5.6-luna", kind="react")   # 或 "planner"
    out = run("Who is the spouse of the Green performer?")
    # → {answer, cited, n_unsupported, contexts, n_search, sources, n_llm_calls}

━━━ 为什么有 `kind`（2026-08 重构 Step 5）━━━

react 臂（`create_agent`，模型自己决定查不查/几次/何时停）在 MuSiQue 上的头号失败模式是
**第一跳拿到桥接实体后直接停**：实测 21 题里 `n_search=1` 的有 6 题，其中典型一题已经从
"Rory Green → Ewan McGregor" 拿到了桥，却没有再搜一次 "Ewan McGregor spouse"，预算 6 只用了 1。

提示词改过两版去修它（v2 加"N 跳至少 N 行引用" → 反向：`cited_grounded` 0.94→0.77，agent 编引用；
v3 加 `unsupported:` 出路 → `n_search` 反而**显著下降** 2.90→2.14）。**两次都证明措辞推不动它。**
⇒ 把「缺哪一环 → 就搜哪一环」从提示词搬进**代码**：planner 臂显式地
`plan → per-hop search → per-hop extract → synthesize`，"第二跳会被发出去"变成控制流保证。

⚠️ planner **不是**默认，它是一条**待检验的臂**。两臂同题配对跑，让数据决定
（`evals/run_matrix.py`）。代价：每跳多 1 次 LLM 调用（3 跳约 5 次 vs react 的约 3 次），
`n_llm_calls` 逐题记录，成本必须和收益一起报 —— 实验24⑤ 的教训是"没有成本项的指标会骗人"。
"""

from __future__ import annotations

import json
import os
import re

from langchain.agents import create_agent

from rag.llm import build_model
from rag.prompts import AGENTIC_RAG_SYSTEM, PLANNER_EXTRACT, PLANNER_PLAN, PLANNER_SYNTH
from rag.retriever import Retriever
from rag.tools import format_hits, make_rag_search, topk

_SRC = re.compile(r"\[source:\s*([^\]]+)\]")
# agent 末尾的 `KEY EVIDENCE:` 块（见 prompts.py 策略6）：它自认为最关键的那几句原文。
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$", re.M)
_SRCTAG = re.compile(r"\[source:[^\]]*\]")
_UNSUP = re.compile(r"^\**\s*unsupported\s*[:：]", re.I)   # prompts.py v3 的 escape hatch
_JSON = re.compile(r"[\[{].*[\]}]", re.S)

# 「拒答」的判据 —— **全项目唯一一份**，确定性、不经过裁判。
#
# ⚠️ 这里曾经是两份：eval_agentic.py 一份、eval_judge.py 一份，词表不同。
# 后果是同一批答案在两个脚本里得到互相矛盾的拒答率 —— 而且恰好在**最需要它的地方**炸：
# MultiHop-RAG 的 null_query（题目本身不可答，**拒答才是正确行为**）上，模型答的是
# "I could not determine …"，只有其中一份词表收了 "could not determine"，
# 于是一个脚本报"30 题全部正确拒答"、另一个报"一次都没拒答"。
# **同一个概念只准有一处定义**；要加措辞就加在这里。
#
# ⚠️ 关键词法有两个**实测踩过的坑**，别再踩：
#   ① 弯引号：模型写的是 "couldn’t"（U+2019），匹配不上 "couldn't"，更匹配不上 "could not"。
#      → `_norm_ans()` 先把 ’ 归一成 '，再把 "n't" 展开成 " not"。
#   ② 词表永远漏：实测还漏过 "could not verify" / "could not find" / "could not reliably identify"。
#      **所以这个信号只用于确定性诊断，不作为唯一判据**——真正要下结论的地方看三分类。
_APOS = str.maketrans({"’": "'", "‘": "'"})

REFUSAL: tuple[str, ...] = (
    "not in passages", "i do not know", "i don't know",
    "could not determine", "could not find", "could not identify", "could not verify",
    "could not reliably", "cannot determine", "can not determine", "cannot find",
    "insufficient information", "not enough information", "unable to determine",
    "unable to find", "unable to verify", "no information", "no relevant",
    "无法确定", "无法回答", "没有找到", "未能找到", "无法从", "不确定",
)

# 「给了候选答案，但自己明确标注证据不支持」—— 这是**第三种行为**，既不是拒答也不是编造。
# 二值的 refused 把它和「睁眼编」归成一类，会把一个诚实的行为记成失败。
HEDGE: tuple[str, ...] = (
    "tentative", "not verify", "does not verify", "do not verify", "not fully support",
    "does not support", "do not support", "not directly verify", "was not found",
    "does not state", "not established", "but the retrieved", "likely",
    "无法核实", "未能证实", "证据不足",
)


def _norm_ans(answer: str | None) -> str:
    return (answer or "").translate(_APOS).replace("n't", " not").strip().lower()


def refused(answer: str | None) -> bool:
    """答案是否是一次**拒答**（不给候选）。空答案也算——它给不出任何断言。"""
    a = _norm_ans(answer)
    return (not a) or any(x in a for x in REFUSAL)


def hedged(answer: str | None) -> bool:
    """给了候选，但**自己声明证据不支持**。⚠️ 只在 `refused()` 为假时才有意义。"""
    a = _norm_ans(answer)
    return bool(a) and any(x in a for x in HEDGE)


def answer_stance(answer: str | None) -> str:
    """三分类：`refused`（不给候选）/ `hedged`（给候选但自曝无依据）/ `asserted`（无免责断言）。

    **在不可答题（null_query）上，只有 `asserted` 才是失败。**`hedged` 是诚实的行为：
    它把不确定性显式交给了用户。实测 30 道不可答题上 asserted = 0。
    """
    if refused(answer):
        return "refused"
    return "hedged" if hedged(answer) else "asserted"



def split_answer(raw: str) -> tuple[str, list[str], int]:
    """把 agent 的输出切成（答案, 引的证据句, 它自认没找到依据的环数）。

    ⚠️ `unsupported: …` 行**必须排除在引用之外**（prompts.py v3 的 escape hatch）。
    不排除的话它会被当成"编造的引用"去核对 `cited_grounded`，把这条改动本身砸掉 ——
    而这条改动的**全部意义**就是让「如实说没有」比「编一个」更省事。
    """
    i = raw.upper().rfind("KEY EVIDENCE")
    if i < 0:
        return raw, [], 0
    body = raw[i:]
    body = body[body.find("\n") + 1:] if "\n" in body else ""
    cited, unsup = [], 0
    for m in _BULLET.finditer(body):
        line = m.group(1).strip()
        if _UNSUP.match(line):                # 自认没依据的环 —— 是诚实，不是引用
            unsup += 1
            continue
        s = _SRCTAG.sub("", line).strip().strip("\"“”'‘’").strip()
        if len(s) >= 20:                      # 太短的多半是小标题不是句子
            cited.append(s)
    return raw[:i].rstrip(), cited, unsup


def build_agent(retriever: Retriever, model: str | None = None):
    """react 臂的底层 LangGraph agent。何时查 / 查几次 / 何时停由模型在 `AGENTIC_RAG_SYSTEM` 下自定。"""
    return create_agent(build_model(model), [make_rag_search(retriever)],
                        system_prompt=AGENTIC_RAG_SYSTEM)


class ReactRunner:
    """`create_agent` 循环 → 统一 runner 输出。"""

    kind = "react"

    def __init__(self, retriever: Retriever, model: str | None = None) -> None:
        self.agent = build_agent(retriever, model)

    def __call__(self, question: str) -> dict:
        """从**终态的完整消息列表**里取工具返回。

        ⚠️ 曾经的写法是在 `stream(stream_mode="values")` 的每个事件里只读 `messages[-1]`——
        **同一轮里的并行 tool 调用会被吞掉**（实测 2 次检索只捕到 1 次，`context_recall_fact`
        被报成 0.500 而真值是 1.000）。**这个 bug 只会让检索看起来更差**，于是"召回上不去"的
        表象里有一部分其实是评测自己造成的。终态里 messages 是全的，直接遍历它。
        """
        state = None
        for chunk in self.agent.stream({"messages": [("user", question)]}, stream_mode="values"):
            state = chunk
        msgs = (state or {}).get("messages", [])
        contexts = [m.content or "" for m in msgs
                    if getattr(m, "type", "") == "tool" and getattr(m, "name", None) == "rag_search"]
        # agent **实际发出的检索 query** —— 只有这个能看出它有没有真的在多跳：
        # 若第 2 次查询里出现了第 1 次才拿到的桥接实体，那才是真的跳了一步。
        # 从 tool_calls 取而不是从 tool 返回取，理由同上：同一轮的并行调用不能漏。
        queries = [(c.get("args") or {}).get("query", "")
                   for m in msgs if getattr(m, "type", "") == "ai"
                   for c in (getattr(m, "tool_calls", None) or [])
                   if (c.get("name") or "") == "rag_search"]
        answer = ""
        for m in msgs:
            if getattr(m, "type", "") == "ai" and (m.content or "").strip() and not getattr(m, "tool_calls", None):
                answer = m.content
        # 末尾的 KEY EVIDENCE 块切出来单独存：裁判判「依据够不够」要用它，
        # 而 answer 本身要**去掉**这块再交给裁判判对错，否则等于把证据当答案一起打分。
        answer, cited, unsup = split_answer(answer)
        got: set[str] = set()
        for c in contexts:
            got |= set(_SRC.findall(c))
        return {"answer": answer, "cited": cited, "n_unsupported": unsup,
                "sources": sorted(got), "contexts": contexts, "n_search": len(contexts),
                "queries": queries,
                "n_llm_calls": sum(1 for m in msgs if getattr(m, "type", "") == "ai"),
                "plan": None}


class PlannerRunner:
    """`plan → (search → extract)*  → synthesize`：**多跳是控制流，不是请求**。

    与 react 的唯一区别就是"第二跳一定会被发出去"。其余（同一个检索器、同一个 k、同一个
    KEY EVIDENCE 输出契约）全部保持一致，好让两臂的差值只归因于控制流。
    """

    kind = "planner"
    MAX_HOPS = int(os.environ.get("RAG_MAX_HOPS", 5))

    def __init__(self, retriever: Retriever, model: str | None = None) -> None:
        self.retr, self.llm = retriever, build_model(model)

    def _ask(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content or ""

    def _plan(self, question: str) -> list[dict]:
        """→ [{goal, query}]，query 里可含 `#1`/`#2` 指代前面某跳的答案。

        解析失败**不静默降级**：退回单跳（等价于原问句直查）并打印，
        静默降级正是实验12/23 栽过的坑（失败长得像成功）。
        """
        raw = self._ask(PLANNER_PLAN.format(q=question, max_hops=self.MAX_HOPS))
        m = _JSON.search(raw)
        try:
            steps = json.loads(m.group(0)) if m else None
            steps = [{"goal": str(s["goal"])[:300], "query": str(s["query"])[:300]}
                     for s in steps if isinstance(s, dict) and s.get("query")][:self.MAX_HOPS]
        except Exception:                                              # noqa: BLE001
            steps = None
        if not steps:
            print(f"  ⚠️ 规划解析失败，退回单跳直查：{raw[:80]!r}", flush=True)
            return [{"goal": question, "query": question}]
        return steps

    @staticmethod
    def _fill(query: str, found: list[str | None]) -> str:
        """把 `#1` 换成第 1 跳查到的答案。**这一步就是"桥"**：react 臂正是在这里停住的。"""
        for i, a in enumerate(found, start=1):
            if a:
                query = query.replace(f"#{i}", a)
        return re.sub(r"#\d+", "", query).strip()

    def _extract(self, goal: str, snippets: str) -> dict:
        """→ {answer, basis, quote, how, source}。`basis` 三档：stated / inferred / none。

        为什么要有 `inferred` 这一档（2026-08-04）：二元的「抄到原句 / NOT FOUND」是 planner
        拒答率 47% 的直接原因。实测把 gold 段直接喂给模型、只把"允许拒答"换成"必须作答"，
        correct 就 **+0.133 [+0.022,+0.244] ✅** —— 那些题它**推得出来，只是不肯断言**。
        根因是 MuSiQue 的关系常常只被**旁证条目**顺带提到（问"某县的县治"，答案藏在一所高中的
        条目里），单段读不出、两段一拼就出来了。
        中间档让它敢跨这一步，同时**要求把跨的那一步写出来**（`how`），所以可审计性不丢：
        `quote` 里的每一句仍要能在检索上下文里核对到（`cited_grounded`）。
        """
        raw = self._ask(PLANNER_EXTRACT.format(goal=goal, snippets=snippets[:24000]))
        m = _JSON.search(raw)
        try:
            d = json.loads(m.group(0))
            basis = str(d.get("basis") or "").strip().lower()
            if basis not in ("stated", "inferred"):
                return {"answer": None, "basis": "none", "quote": "", "how": "", "source": ""}
            return {"answer": (d.get("answer") or "").strip() or None,
                    "basis": basis,
                    "quote": (d.get("quote") or "").strip(),
                    "how": (d.get("how") or "").strip()[:200],
                    "source": (d.get("source") or "").strip()}
        except Exception:                                              # noqa: BLE001
            return {"answer": None, "basis": "none", "quote": "", "how": "", "source": ""}

    def __call__(self, question: str) -> dict:
        steps = self._plan(question)
        contexts, chain, found, calls = [], [], [], 1
        for step in steps:
            q = self._fill(step["query"], found) or step["goal"]
            snips = format_hits(self.retr.search(q, k=topk()))
            contexts.append(snips)
            got = self._extract(step["goal"], snips)
            calls += 1
            if got["answer"] is None:
                # 一次重试：只用 goal 本身当 query（换个说法再捞一遍），仍空则如实记为未支撑。
                snips2 = format_hits(self.retr.search(step["goal"], k=topk()))
                contexts.append(snips2)
                got = self._extract(step["goal"], snips2)
                calls += 1
            found.append(got["answer"])
            chain.append({"goal": step["goal"], "query": q, **got})
        rendered = "\n".join(
            f"{i}. {c['goal']}\n   → {c['answer'] or 'NOT FOUND'}  [{c.get('basis', 'none')}]"
            + (f"\n   quote: \"{c['quote']}\" [source: {c['source']}]" if c["quote"] else "")
            + (f"\n   how: {c['how']}" if c.get("how") else "")
            for i, c in enumerate(chain, start=1))
        answer = self._ask(PLANNER_SYNTH.format(q=question, chain=rendered))
        calls += 1
        # KEY EVIDENCE **不问模型要**，直接由链条拼出来：每一跳贡献它 extract 出的那句原文。
        # 引用于是成为控制流的**副产品**，而不是 agent 的自述 —— 这正是 react 臂两次改提示词
        # 都没修好的那件事（v2 逼它凑行数→编引用 0.94→0.77；v3 给出路→变懒 n_search 2.90→2.14）。
        # 是否真的更可靠，仍由确定性的 `cited_grounded` 核对（extract 也可能编 quote）。
        answer, _stray, _ = split_answer(answer)      # 模型若仍自带 KEY EVIDENCE 块，剥掉
        # `inferred` 档的 quote 可能是用 " | " 拼起来的多句 —— 拆开，好让每一句都能被
        # `cited_grounded` 单独核对。不拆的话整串在上下文里找不到，会被误判成编造。
        cited = [q.strip() for c in chain for q in c["quote"].split(" | ") if q.strip()]
        unsup = sum(1 for c in chain if not c["quote"])
        got_src: set[str] = set()
        for c in contexts:
            got_src |= set(_SRC.findall(c))
        return {"answer": answer, "cited": cited, "n_unsupported": unsup,
                "sources": sorted(got_src), "contexts": contexts, "n_search": len(contexts),
                "queries": [c["query"] for c in chain],      # 与 react 同名，报表/展示可共用一条路径
                "n_llm_calls": calls, "plan": chain,
                # 成本/行为项：跨段推了几环。它涨而 correct 不涨 = 推错了；两个一起涨才是买到了东西。
                "n_inferred": sum(1 for c in chain if c.get("basis") == "inferred")}


def build_runner(retriever: Retriever, model: str | None = None, kind: str | None = None):
    """统一入口。`kind` 缺省读 `RAG_AGENT`（react / planner），再缺省 react。"""
    kind = (kind or os.environ.get("RAG_AGENT") or "react").lower()
    if kind == "planner":
        return PlannerRunner(retriever, model)
    if kind != "react":
        raise SystemExit(f"未知 agent 类型 {kind!r}（可选：react / planner）")
    return ReactRunner(retriever, model)
