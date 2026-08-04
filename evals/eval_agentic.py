"""跑 agentic RAG 并算**确定性**指标，逐题落盘（`eval_judge.py` 的输入）。

    python evals/eval_agentic.py --out runs/dumps/a.jsonl --per-type 30
    python evals/eval_agentic.py --out runs/dumps/a.jsonl --resume        # 网关抽风后接着跑
    python evals/eval_agentic.py --out runs/dumps/b.jsonl --model gpt-5.6-luna --agent planner

━━━ 2026-08 重构：这个文件只做**确定性**的量 ━━━

这里出的每个数都是字符串匹配算出来的，零裁判噪声。头号裁判分在 `eval_judge.py`。

  检索栈  `delivered` = gold 证据段里检索**实际给到**的比例（= 引用了 + 检索到没引）。
          **这才是评检索栈的数**，且它完全不经过 agent 的嘴。
  引用行为 `fate_cited` 引出来了 / `cited_grounded` 引的句子真在检索上下文里。
          **是诊断，不是目标** —— 把它当目标就会得到 v2 那种"凑行数编引用"（0.94→0.77）。
  成本    `n_chunks` 交付片段数 / `n_search` 检索次数 / `n_llm_calls` LLM 调用次数。
          实验24⑤ 的教训：没有成本项的指标必然奖励"塞得更多"。

三条改掉旧版毛病的：
  ① **边跑边落盘 + `--resume`**：旧版跑完才一次性写，90 题跑到 80 题网关挂就全丢 ——
     这是此前所有结论都卡在 n=21 的直接原因之一。
  ② **`__meta__` 头**：记模型/裁判/提示词 sha/k/pool/chunk/git sha。旧 dump 不记，
     实测已经出过"`.env` 写 grok-4.5、实验记录写 deepseek"对不上的事。
  ③ **默认 `--per-type 30`**：n=21 时区间宽 ±0.2，而在追的效应是 0.05 —— 那种实验
     在设计上就不可能有结论（实验26/27/28 共 14 个区间，13 个跨 0）。低于 20 会告警。

⚠️ 失败必须可见：网关连续失败的题记 `errored` 并**排除出均值，绝不当 0 分**（实验12）。
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import pathlib
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from rag.agent import build_runner
from rag.llm import build_model
from rag.retriever_decompose import maybe_wrap
from rag.retriever_hybrid import HybridRetriever
from rag.runctx import DumpWriter, fmt_meta, read_dump, resume_ids, snapshot

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
REFUSAL = ("insufficient information", "cannot determine", "not enough information",
           "无法确定", "没有找到", "未能找到", "无法回答")

# 工具返回里每个片段都以 "[source: ...]" 开头（见 tools.py），据此切回单个片段。
_CHUNK = re.compile(r"(?=\[source: )")
_BUDGETS = (8, 16, 32, 64)


def _seen_types(rows) -> list[str]:
    """表格里要打哪几行题型 —— 从数据里取，别写死。写死的话换个评测集（MuSiQue 是 2/3/4hop）
    整张表会**静默变空**，看着像"跑了但没结果"。"""
    order = {t: i for i, t in enumerate(
        ["2hop", "3hop", "4hop", "comparison", "inference", "temporal", "null"])}
    return sorted({r["type"] for r in rows}, key=lambda t: (order.get(t, 99), t))


# ── 评测集：examples + question→type ──────────────────────────────────────
class _Ex:
    """离线版 example（MultiHop-RAG 用）。MuSiQue 那边 corpus_musique 自带同形的 _Ex。"""

    __slots__ = ("id", "inputs", "outputs")

    def __init__(self, i: int, r: dict):
        ev = r.get("evidence_list") or []
        self.id = f"local-{i}"
        self.inputs = {"question": r["query"]}
        self.outputs = {
            "reference": "" if r.get("question_type") == "null_query" else (r.get("answer") or "").strip(),
            "gold_titles": sorted({e.get("title", "").strip() for e in ev if e.get("title")}),
            "reference_contexts": [e.get("fact", "").strip() for e in ev if e.get("fact")],
        }


def load_benchmark(benchmark: str):
    """→ (examples, {question: type}, corpus)。**唯一**一处知道两个评测集差别的地方。"""
    if benchmark == "musique":
        from rag.corpus_musique import load_corpus, load_questions
        ex, qtype = load_questions()
        return ex, qtype, load_corpus()
    from rag.corpus_multihop import load_corpus
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return ([_Ex(i, r) for i, r in enumerate(raw)],
            {r["query"]: r.get("question_type", "?") for r in raw},
            load_corpus())


def sample(examples, qtype, types: list[str], per_type: int) -> list:
    """按题型均衡、**确定性**抽样（等距取样，不用随机种子）—— 同 per_type 必抽到同一批题，
    这是"同题配对"的前提。配对做不成，所有差值都会被题目难度的方差淹掉。"""
    by_type: dict[str, list] = defaultdict(list)
    for e in examples:
        by_type[qtype.get(e.inputs.get("question"), "?")].append(e)
    picked = []
    for t in types:
        pool = by_type.get(t, [])
        step = max(1, len(pool) // per_type)
        picked += pool[::step][:per_type]
    return picked


# ── 确定性指标 ────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _recall(facts: list[str], text: str) -> float:
    blob = _norm(text)
    return sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts) if facts else float("nan")


def _fact_fate(facts: list[str], contexts: list[str], cited: list[str]) -> dict:
    """每条 gold 证据段的**去向**——把"检索没给"和"agent 没引"分开。

    起因（MuSiQue 21 题实测）：`sufficient` 只认 agent **引出来**的句子，于是"检索到了却没引"
    被记成依据不足。实测这一类占 33~49%，而真检索缺口只有 23~34% —— 不拆开的话会把
    **提示词问题误判成检索问题**，然后去调错的旋钮（本项目栽过的同一类坑）。

    `delivered` = cited + retrieved_not_cited = **检索实际交付了多少链条**，它才是评检索栈的数。
    """
    if not facts:
        return {"fate_cited": None, "fate_uncited": None, "fate_missing": None, "delivered": None}
    ctx, cit = _norm(" ".join(contexts)), [_norm(c) for c in cited]
    a = b = c = 0
    for f in facts:
        nf = _norm(f)
        if nf[:80] in " ".join(cit) or any(x[:80] in nf for x in cit if len(x) >= 80):
            a += 1
        elif nf[:120] in ctx:
            b += 1
        else:
            c += 1
    n = len(facts)
    return {"fate_cited": a / n, "fate_uncited": b / n, "fate_missing": c / n, "delivered": (a + b) / n}


def _in_ctx(s: str, blob: str) -> bool:
    """这句引用是不是真的来自 agent 检索到的上下文（宽松：允许首尾标点被改）。"""
    n = _norm(s)
    if not n:
        return False
    if n[:120] in blob:
        return True
    t = n.split()                             # 退一步：有连续 10 个词原样出现就算抄自原文
    return any(" ".join(t[i:i + 10]) in blob for i in range(max(0, len(t) - 9)))


def _budget_metrics(facts: list[str], contexts: list[str]) -> dict:
    """`recall@B`：只取**交付顺序上的前 B 个片段**算召回 —— 不管搜几次、每次 k 多大，
    所有配置在**同一个上下文预算**下才可比（修「搜得多必然赢」）。
    `curve`：逐跳**累计**召回，相邻两项之差 = 那一跳的边际贡献；第 2 跳边际≈0 说明多跳是假的。"""
    per_hop = [[c for c in _CHUNK.split(ctx) if c.strip()] for ctx in contexts]
    delivered = [c for hop in per_hop for c in hop]          # 交付顺序（按跳、跳内按名次）
    curve, seen = [], []
    for hop in per_hop:
        seen += hop
        curve.append(_recall(facts, " ".join(seen)))
    return {
        "context_recall_fact": _recall(facts, " ".join(delivered)),   # 历史口径：全部并集
        "n_chunks": len(delivered),
        "n_ctx_chars": sum(len(c) for c in delivered),
        "recall_at": {str(b): _recall(facts, " ".join(delivered[:b])) for b in _BUDGETS},
        "curve": curve,
    }


def score_row(ex, qtype: dict, out: dict) -> dict:
    """一题的全部确定性指标。`out` = runner 的返回。"""
    facts = ex.outputs.get("reference_contexts") or []
    cited = out["cited"]
    blob = _norm(" ".join(out["contexts"]))
    return {
        "example_id": str(ex.id), "question": ex.inputs["question"],
        "type": qtype.get(ex.inputs["question"], "?").split("_")[0],
        **_budget_metrics(facts, out["contexts"]),
        **_fact_fate(facts, out["contexts"], cited),
        "n_search": out["n_search"], "n_llm_calls": out.get("n_llm_calls"),
        # planner 专有：跨段推断了几环 / 认输了几环。**必须显式抄进行记录** ——
        # 初版漏了这一行，于是 n_inferred 恒为 0，差点被我读成"中间档一次没被用上"
        # （真值 11/252 步，从 plan 字段才挖出来）。指标缺一行 = 一个不存在的结论。
        "n_inferred": out.get("n_inferred"),
        "refused": 1.0 if any(c in (out["answer"] or "").lower() for c in REFUSAL) else 0.0,
        "answer_len": len(out["answer"] or ""), "answer": out["answer"],
        "cited": cited, "n_cited": len(cited), "n_unsupported": out["n_unsupported"],
        # 守卫：引用是 agent 的**自述**，不核对的话「没检索到」和「检索到了但没引」分不开。
        "cited_grounded": (sum(_in_ctx(s, blob) for s in cited) / len(cited)) if cited else float("nan"),
        # 存 agent **实际检索到**的上下文 —— eval_judge 判 `grounded` 要拿它当参照系，
        # 不存就只能另起炉灶重检一次，那量到的是检索栈、不是这个 agent 的检索。
        "contexts": out["contexts"],
        "plan": out.get("plan"),
    }


# ── 汇总表 ────────────────────────────────────────────────────────────────
def report(rows: list[dict], meta: dict, diag: bool = False) -> None:
    ok = [r for r in rows if "errored" not in r]
    n_err = len(rows) - len(ok)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")   # noqa: E731
    ok_ = lambda xs: [x for x in xs if x == x and x is not None]    # noqa: E731 —— 去 NaN/None

    print(f"\n== 确定性指标 ==\n   {fmt_meta(meta)}")
    print(f"{'type':8s}{'n':>4}{'交付†':>8}{'└引用了':>9}{'└没引':>8}{'缺口':>7}"
          f"{'片段数':>8}{'万字符':>8}{'n_search':>9}{'LLM次':>7}{'引用句':>8}{'认怂环':>8}"
          f"{'引用属实':>9}{'refused':>8}")
    for t in [*_seen_types(ok), "全部"]:
        d = [r for r in ok if t == "全部" or r["type"] == t]
        if not d:
            continue
        f = lambda k: mean(ok_([r.get(k) for r in d]))               # noqa: E731
        print(f"{t:8s}{len(d):>4}{f('delivered'):>8.3f}{f('fate_cited'):>9.3f}"
              f"{f('fate_uncited'):>8.3f}{f('fate_missing'):>7.3f}{f('n_chunks'):>8.1f}"
              f"{f('n_ctx_chars') / 1e4:>8.2f}{f('n_search'):>9.2f}{f('n_llm_calls'):>7.1f}"
              f"{f('n_cited'):>8.1f}{f('n_unsupported'):>8.1f}"
              f"{f('cited_grounded'):>9.2f}{f('refused'):>8.2f}")
    print("  † 交付 = gold 证据段里检索**实际给到**的比例（= 引用了 + 检索到没引）。**这才是评检索栈的数。**\n"
          "    「检索到没引」高 → agent 的引用行为问题；「缺口」高 → 真检索问题。两者混在一起会去调错的旋钮。\n"
          "  * 引用属实 <0.8 = agent 在编引用，`eval_judge.py` 的 sufficient 会被判为不可信。\n"
          "  * 万字符 / LLM次 是**成本项**：任何「分更高」的结论都要和它一起读（实验24⑤）。")

    if diag:
        print(f"\n== [--diag] 固定预算召回 recall@B（只算交付顺序前 B 个片段）==")
        print(f"{'type':8s}{'n':>4}" + "".join(f"{'@' + str(b):>8}" for b in _BUDGETS))
        for t in _seen_types(ok):
            d = [r for r in ok if r["type"] == t and r["context_recall_fact"] == r["context_recall_fact"]]
            if d:
                print(f"{t:8s}{len(d):>4}"
                      + "".join(f"{mean([r['recall_at'][str(b)] for r in d]):>8.3f}" for b in _BUDGETS))
        print(f"\n== [--diag] 逐跳**边际**召回（第 i 跳新增了多少 gold 证据；≈0 = 那一跳只在重复捞）==")
        print(f"{'type':8s}" + "".join(f"{'第' + str(i + 1) + '跳':>10}" for i in range(4)))
        for t in _seen_types(ok):
            d = [r for r in ok if r["type"] == t and r.get("curve")]
            if not d:
                continue
            cells = []
            for i in range(4):
                g = [r["curve"][i] - (r["curve"][i - 1] if i else 0.0) for r in d if len(r["curve"]) > i]
                cells.append(f"{mean(g):>+10.3f}" if g else f"{'—':>10}")
            print(f"{t:8s}" + "".join(cells))

    if n_err:
        print(f"\n⚠️ {n_err} 题网关连续失败，已记 errored 并排除（**不当 0 分**）。`--resume` 可只补这些题。")
    zero = [r for r in ok if r["n_search"] == 0]
    if zero:
        print(f"⚠️ {len(zero)}/{len(ok)} 题 agent **一次都没检索** —— 召回=0 是**模型没查**，不是检索器没捞到。")
    nocite = [r for r in ok if r["n_search"] > 0 and r["n_cited"] == 0]
    if nocite:
        print(f"⚠️ {len(nocite)}/{len(ok)} 题查了却**没吐 KEY EVIDENCE 块** —— 是**格式问题不是检索问题**。")
    if len(ok) < 60:
        print(f"\n⚠️ n={len(ok)}：配对差值的 95% 区间宽度约 ±{1.96 * 0.35 / max(len(ok), 1) ** 0.5:.2f}。"
              f"\n   要分辨 0.05 量级的效应需 n≳200；分辨 0.10 需 n≳50。**别在这个样本量上宣布方向**（实验15/20 的教训）。")


def execute(picked, qtype, runner, out: str, meta: dict, concurrency: int = 4,
            resume: bool = False, make_runner=None) -> list[dict]:
    """跑一批题、边跑边落盘。**run_matrix.py 也走这条路** —— 一个臂 = 一次 execute。

    网关 502/524 会整题打空 → **绝不能吞掉**：吞了就变成 0 分计入均值，把实验读成
    「模型变差了」（实测 qwen 轮 temporal 8 题空了 7 题）。长退避重试，仍失败就记 errored。
    """
    done = resume_ids(out) if resume and pathlib.Path(out).exists() else set()
    todo = [e for e in picked if str(e.id) not in done]
    if not todo:
        return read_dump(out)[1]

    def one(ex) -> dict:
        row = {"example_id": str(ex.id), "question": ex.inputs["question"],
               "type": qtype.get(ex.inputs["question"], "?").split("_")[0]}
        last: Exception | None = None
        for wait in (0, 30, 90):                      # 3 次尝试，总退避 ~2 分钟
            if wait:
                time.sleep(wait)
            try:
                # oracle 模式：每题现建一个绑着**这道题 gold 段**的 runner（构造很便宜，
                # 只是包一层 create_agent；模型/提示词/循环全都不变）。
                r = make_runner(ex) if make_runner else runner
                return score_row(ex, qtype, r(ex.inputs["question"]))
            except Exception as e:                    # noqa: BLE001 —— 网关各类 5xx/超时
                last = e
                print(f"  ⚠️ {ex.id} 网关失败({type(e).__name__})，重试…", flush=True)
        return row | {"errored": f"{type(last).__name__}: {str(last)[:120]}"}

    t0 = time.time()
    with DumpWriter(out, meta, resume=resume) as w, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i, row in enumerate(pool.map(one, todo), start=1):
            w.write(row)
            if i % 10 == 0:
                print(f"  … {i}/{len(todo)}  ({(time.time() - t0) / i:.1f}s/题)", flush=True)
    return read_dump(out)[1]


class GoldRetriever:
    """**永远返回这道题的 gold 段**（忽略 query）—— 用来量「把证据全喂到嘴边，agent 能答多少」。

    与 `eval_ceiling.py` 的 oracle 线的区别，正是这个脚本存在的理由：
    那边用的是一句话提示词（"用这些段落回答"），这边走的是**agent 的完整链路** ——
    同一套系统提示词（第一原则是"绝不写没抄到的句子"）、同一个多跳循环、同一套 KEY EVIDENCE 契约。
    两个数不一样的部分，就是**agent 这层外壳自己的代价**（多半体现在拒答率上）。

    ⚠️ 它使 `delivered` 恒为 1.0（gold 必然在上下文里），那一列在 oracle 模式下没有信息量。
    """

    def __init__(self, docs) -> None:
        self.docs = docs

    def search(self, query: str, k: int = 4):
        from rag.retriever import Hit
        return [Hit(doc=d, score=0.0) for d in self.docs[:k]]


def gold_docs(ex):
    from rag.retriever import Doc
    titles = ex.outputs.get("gold_titles") or []
    return [Doc(id=f"gold-{i}", text=t, source=(titles[i] if i < len(titles) else "gold"))
            for i, t in enumerate(ex.outputs.get("reference_contexts") or [])]


def build_retriever(corpus, reranker: str = "bge", model: str | None = None):
    """检索器。**建一次给多个臂复用** —— 重排器加载 + 向量库连接是每轮的固定开销。"""
    if reranker == "qwen":
        from rag.reranker_qwen import QwenReranker
        retr = HybridRetriever(corpus, reranker=QwenReranker())
    else:
        retr = HybridRetriever(corpus)
    # RAG_DECOMPOSE=1 时套上「分解 → 逐子问句各自重排 → 轮流取名额」（默认关闭）
    return maybe_wrap(retr, llm=build_model(model))


def main() -> None:
    ap = argparse.ArgumentParser(description="跑 agentic RAG + 确定性指标，逐题落盘")
    ap.add_argument("--out", "--local", dest="out", required=True, metavar="OUT.jsonl",
                    help="逐题 JSONL 输出（第 1 行是 __meta__ 配置快照）。这是 eval_judge.py 的输入")
    ap.add_argument("--benchmark", choices=["musique", "multihoprag"], default="musique",
                    help="默认 musique —— **MultiHop-RAG 已被实验25 证伪**（捷径率 99.3%%，测的是跨文档聚合不是多跳）")
    ap.add_argument("--per-type", type=int, default=30,
                    help="每种题型抽几题（默认 30 = 90 题）。n=21 时区间宽 ±0.2，追 0.05 的效应等于白跑")
    ap.add_argument("--types", default="", help="只评这些题型（逗号分隔）。默认按评测集全取")
    ap.add_argument("--model", default=None, help="**答题**模型名（查 endpoints.json）。裁判端不受影响")
    ap.add_argument("--agent", default=None, choices=["react", "planner"],
                    help="控制流：react（模型自定何时查/几次/何时停）| planner（plan→逐跳搜→合成，多跳是代码保证）")
    ap.add_argument("--reranker", default="bge", choices=["bge", "qwen"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="接着已有 --out 跑：跳过其中已成功的题（errored 的会重跑）。网关抽风时的救命稻草")
    ap.add_argument("--diag", action="store_true", help="额外打印 recall@B + 逐跳边际召回")
    ap.add_argument("--report-only", action="store_true", help="不跑，只把已有 --out 重新汇总一遍")
    ap.add_argument("--oracle", action="store_true",
                    help="**把 gold 段直接当检索结果喂给 agent**（不做真检索）——量「证据全给到，"
                         "这套 agent 能答多少」。与 eval_ceiling.py 的 oracle 线的差别是：那边是一句话"
                         "提示词，这边走 agent 的完整链路，差值 = agent 外壳自己的代价")
    args = ap.parse_args()

    if args.report_only:
        meta, rows = read_dump(args.out)
        return report(rows, meta, args.diag)

    default_types = {"musique": ["2hop", "3hop", "4hop"],
                     "multihoprag": ["comparison_query", "inference_query", "temporal_query", "null_query"]}
    types = [t.strip() for t in args.types.split(",") if t.strip()] or default_types[args.benchmark]

    examples, qtype, corpus = load_benchmark(args.benchmark)
    picked = sample(examples, qtype, types, args.per_type)
    print(f"抽样 {len(picked)} 题（{', '.join(types)} 各 {args.per_type}）")

    if args.agent:
        os.environ["RAG_AGENT"] = args.agent
    make_runner = None
    if args.oracle:
        print("  [oracle] 检索被旁路：每题直接把它的 gold 段当检索结果喂给 agent。"
              "delivered 恒=1.0，那一列无信息量。")
        make_runner = lambda ex: build_runner(GoldRetriever(gold_docs(ex)), args.model, args.agent)  # noqa: E731
        runner = build_runner(GoldRetriever([]), args.model, args.agent)
    else:
        runner = build_runner(build_retriever(corpus, args.reranker, args.model), args.model, args.agent)
    meta = snapshot(benchmark=args.benchmark, per_type=args.per_type, types=types,
                    n_sampled=len(picked), reranker=args.reranker, oracle=bool(args.oracle),
                    answer_model=args.model, agent=runner.kind)
    print(f"   {fmt_meta(meta)}")

    execute(picked, qtype, runner, args.out, meta, args.concurrency, args.resume, make_runner)
    meta, rows = read_dump(args.out)
    report(rows, meta, args.diag)


if __name__ == "__main__":
    main()
