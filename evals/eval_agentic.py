"""在 LangSmith 数据集（multihop-rag）上跑 **agentic RAG（多跳）**，按 question_type 均衡抽样。

    python eval_agentic.py --local ab.jsonl --per-type 30   # 推荐：跑 agent + 确定性指标，写逐题 JSONL
    python eval_judge.py ab.jsonl                           # 再用裁判打头号三分（对/依据够/忠诚）
    python eval_agentic.py --per-type 8                     # 老路：直接推 LangSmith（4 个 openevals 裁判）

target = agentic agent（多跳，检索后端 hybrid）。

**指标分工**（2026-07 重整，起因：指标从 4 个长到 15 个，头号指标却一直不可信）：
  头号 —— `eval_judge.py`：裁判读 agent **自己引的关键句**，判 对/依据够不够/忠诚。
           不要求每条 gold 证据都召回（inference 类平均 3.36 条里只有 1.47 条真含答案）。
  守卫 —— `cited_grounded`（本文件，确定性）：那些引用真来自检索上下文的比例。
           **它不接近 1.0，头号指标就作废** —— 引用是自述，不核对的话
           「没检索到」和「检索到了但没引」分不开。
  诊断 —— `context_recall_fact`（过严下界）/ `recall@B` / 逐跳边际召回 / `n_search`。
           **只在头号指标动了、要归因时看**（`--diag` 展开），不作结论。
  退役 —— `context_recall`(title 级，实验8-9 证明方向会反) / groundedness / retrieval_relevance
           / helpfulness（三者从未提供过信息量，已被头号三分覆盖）。

⚠️ 成本：每题多跳 ~5-9 次生成；`--local` 每题便宜 5 倍（不叫裁判）。
   网关偶发 5xx 会让个别题 errored —— **排除出均值，绝不当 0 分**。
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import json
import pathlib
import re
import time
from collections import defaultdict

from langsmith import Client, evaluate

from rag.agent import build_agent
from rag.corpus_multihop import load_corpus
from evals.eval_common import REFUSAL, SPECS, call_judge, context_recall_fact, make_judges, refused
from rag.llm import build_judge, build_model
from rag.retriever_decompose import maybe_wrap
from rag.retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
_TYPES = ["comparison_query", "inference_query", "temporal_query", "null_query"]
_SRC = re.compile(r"\[source:\s*([^\]]+)\]")


def _seen_types(rows) -> list[str]:
    """表格里要打哪几行题型 —— 从数据里取，别写死。写死的话换个评测集（MuSiQue 是 2/3/4hop）
    整张表会**静默变空**，看着像"跑了但没结果"。"""
    order, out = {t: i for i, t in enumerate(
        ["comparison", "inference", "temporal", "null", "2hop", "3hop", "4hop"])}, set()
    for r in rows:
        out.add(r["type"])
    return sorted(out, key=lambda t: (order.get(t, 99), t))


def _corpus(benchmark: str):
    """按评测集取语料。MuSiQue 的段落**不再二次切块**——维基一段本身就是自然检索单元。"""
    if benchmark == "musique":
        from rag.corpus_musique import load_corpus as _lc
        return _lc()
    return load_corpus()


def _type_map() -> dict[str, str]:
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {r["query"]: r.get("question_type", "?") for r in raw}


class _Ex:
    """离线版的 LangSmith example —— 字段与 `langsmith_upload.py` 造的完全一致。

    有它，`--local` 就**完全不碰 LangSmith**（连读数据集都不用）。原因：本月追踪配额已满，
    且多一个网络依赖就多一种「跑一半挂掉」。⚠️ 抽到的题**不保证**与 LangSmith 路径一一对应
    （那边的返回顺序由服务端定），所以新旧 dump 不要混着配对；离线路径自己是确定性可复现的。
    """

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


def _offline_examples() -> list[_Ex]:
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return [_Ex(i, r) for i, r in enumerate(raw)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


# 工具返回里每个片段都以 "[source: ...]" 开头（见 tools.py），据此切回单个片段。
_CHUNK = re.compile(r"(?=\[source: )")
_BUDGETS = (8, 16, 32, 64)

# agent 末尾的 `KEY EVIDENCE:` 块（见 prompts.py 策略5）：它自认为最关键的那几句原文。
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$", re.M)
_SRCTAG = re.compile(r"\[source:[^\]]*\]")
_UNSUP = re.compile(r"^\**\s*unsupported\s*[:：]", re.I)   # prompts.py v3 的 escape hatch


def _split_answer(raw: str) -> tuple[str, list[str], int]:
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


def _fact_fate(facts: list[str], contexts: list[str], cited: list[str]) -> dict:
    """每条 gold 证据段的**去向**——把"检索没给"和"agent 没引"分开。

    起因（MuSiQue 21 题实测）：`sufficient` 只认 agent **引出来**的句子，于是"检索到了却没引"
    被记成依据不足。实测这一类占 33.3%，而真检索缺口只有 28.6% —— 不拆开的话会把
    **提示词问题误判成检索问题**，然后去调错的旋钮（本项目栽过的同一类坑）。

    返回 cited / retrieved_not_cited / missing 三者的比例，三者之和为 1。
    `delivered` = cited + retrieved_not_cited = **检索实际交付了多少链条**，
    它才是评检索栈的那个数；`cited` 才是评 agent 会不会引的那个数。
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
    return {"fate_cited": a / n, "fate_uncited": b / n, "fate_missing": c / n,
            "delivered": (a + b) / n}


def _in_ctx(s: str, blob: str) -> bool:
    """这句引用是不是真的来自 agent 检索到的上下文（宽松：允许首尾标点被改）。"""
    n = _norm(s)
    if not n:
        return False
    if n[:120] in blob:
        return True
    t = n.split()                             # 退一步：有连续 10 个词原样出现就算抄自原文
    return any(" ".join(t[i:i + 10]) in blob for i in range(max(0, len(t) - 9)))


def _recall(facts: list[str], text: str) -> float:
    blob = _norm(text)
    return sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts) if facts else float("nan")


def _agentic_metrics(facts: list[str], contexts: list[str]) -> dict:
    """给 agentic RAG 用的一组指标 —— 因为「所有轮次取并集的召回」这把尺子对它是错的。

    ⚠️ **这些是诊断，不是头号指标**（默认表里不打印，`--diag` 才展开）。头号指标见 `eval_judge.py`：
    让裁判读 agent **自己引的证据句**，判「这些依据够不够支撑答案」。原因（用户提出，实测支持）：
    子串匹配 gold 句**过严**（gold 句只是作者挑的某一句，同篇另一句同样能支撑却判 0），
    且**冗余证据同等计分**（inference 类平均 3.36 条证据里只有 1.47 条真含答案）。

    留着它们是因为它们**确定性、零成本、且是个下界** —— 当它和裁判分打架时，是查 bug 的入口。

    毛病三处：
      1. **没有成本项**：搜 10 次 × 32 片段必然赢过搜 1 次 × 32，哪怕后 9 次全是重复；
      2. **分不清「广撒网撞到」和「顺着线索找到」**：一次大 k 的粗检索与一条真多跳链，在并集上一模一样。
         而多跳的定义恰恰是**后一跳用到了前一跳的信息**；
      3. 它只是 correctness 的代理，而实验8-9 已经栽过一次「刷代理、correctness 反降」。

    所以这里补两把尺子：
      `recall@B` —— 只取**交付顺序上的前 B 个片段**算召回。不管搜几次、每次 k 多大，
                    所有配置在**同一个上下文预算**下才可比。这是修掉毛病 1 的那把。
      `curve`    —— 逐跳的**累计**召回。相邻两项之差 = 那一跳的**边际贡献**；
                    若第 2 跳边际≈0，说明它只是在重复捞同样的东西，多跳是假的。这是修掉毛病 2 的那把。
    """
    per_hop = [[c for c in _CHUNK.split(ctx) if c.strip()] for ctx in contexts]
    delivered = [c for hop in per_hop for c in hop]          # 交付顺序（按跳、跳内按名次）
    curve, seen = [], []
    for hop in per_hop:
        seen += hop
        curve.append(_recall(facts, " ".join(seen)))
    return {
        "context_recall_fact": _recall(facts, " ".join(delivered)),   # 历史口径：全部并集
        "n_chunks": len(delivered),
        "recall_at": {str(b): _recall(facts, " ".join(delivered[:b])) for b in _BUDGETS},
        "curve": curve,                                               # 逐跳累计召回
    }


def _run_local(picked, qtype, target, out: pathlib.Path, concurrency: int, diag: bool = False) -> None:
    """本地跑 agent + 确定性指标，逐题写 JSONL（`example_id` 用于**同题配对**）。

    为什么要有这条路：LLM-judge 的分在每型 8 题上标准误差就有 0.1~0.15——**把预算堆在裁判上，
    量到的全是噪声**。去掉裁判调用后每题便宜 5 倍，同样的钱能把 n 抬到能区分效应的量级。
    另：LangSmith 本月追踪配额已满（`evaluate()` 的 run 存不进去），本地路径顺带绕开它。

    产出的 JSONL 是 `eval_judge.py` 的输入 —— 那边才是头号指标（裁判读 agent 自引的证据句，
    判依据够不够）。这边只出**确定性**的量，其中最要紧的是 `cited_grounded`：
    引用是 agent 的**自述**，不核对的话「没检索到」和「检索到了但没引出来」分不开。
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(ex):
        facts = ex.outputs.get("reference_contexts") or []
        row = {"example_id": str(ex.id), "question": ex.inputs["question"],
               "type": qtype.get(ex.inputs["question"], "?").split("_")[0]}
        try:
            o = target(ex.inputs)
        except Exception as e:                            # noqa: BLE001 —— 网关连续失败
            return row | {"errored": f"{type(e).__name__}: {str(e)[:100]}"}
        blob = _norm(" ".join(o["contexts"]))
        cited = o["cited"]
        return row | _agentic_metrics(facts, o["contexts"]) | _fact_fate(facts, o["contexts"], cited) | {
            "n_search": o["n_search"],
            "refused": 1.0 if any(c in (o["answer"] or "").lower() for c in REFUSAL) else 0.0,
            "answer_len": len(o["answer"] or ""),
            "answer": o["answer"],
            # agent 自己挑的「最关键的几句」（prompts.py 策略5）+ 其中真来自检索上下文的比例。
            # 后者是**守卫**：它低就说明 agent 在编引用，`eval_judge.py` 的依据分随之作废。
            "cited": cited,
            "n_cited": len(cited),
            # agent 自认「这一环我没找到依据」的次数。高不是坏事 —— 它是**诚实的替代品**，
            # 用来换掉编引用。要跟 cited_grounded 一起看：unsup 高 + grounded 高 = 改对了。
            "n_unsupported": o["n_unsupported"],
            "cited_grounded": (sum(_in_ctx(s, blob) for s in cited) / len(cited)) if cited else float("nan"),
            # 存下 agent **实际检索到**的上下文 —— 裁判/可答性脚本要用它。不存的话就只能另起炉灶
            # 重检一次，那量到的是检索栈、不是这个 agent 的检索。
            "contexts": o["contexts"],
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, picked))
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    ok = [r for r in rows if "errored" not in r]
    n_err = len(rows) - len(ok)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    ok_ = lambda xs: [x for x in xs if x == x]                     # noqa: E731 —— 去 NaN
    print(f"\n== 本地确定性指标（逐题结果已写入 {out}）==")
    print(f"{'type':11s}{'n':>4}{'交付†':>8}{'└引用了':>9}{'└没引':>8}{'缺口':>7}"
          f"{'片段数':>8}{'n_search':>9}{'引用句':>8}{'认怂环':>8}{'引用属实':>9}{'refused':>8}")
    for t in _seen_types(ok):
        d = [r for r in ok if r["type"] == t]
        if not d:
            continue
        f = lambda k: mean(ok_([r[k] for r in d if r.get(k) is not None]))   # noqa: E731
        print(f"{t:11s}{len(d):>4}{f('delivered'):>8.3f}{f('fate_cited'):>9.3f}"
              f"{f('fate_uncited'):>8.3f}{f('fate_missing'):>7.3f}"
              f"{mean([r['n_chunks'] for r in d]):>8.1f}"
              f"{mean([r['n_search'] for r in d]):>9.2f}{mean([r['n_cited'] for r in d]):>8.1f}"
              f"{mean([r.get('n_unsupported', 0) for r in d]):>8.1f}"
              f"{mean(ok_([r['cited_grounded'] for r in d])):>9.2f}"
              f"{mean([r['refused'] for r in d]):>8.2f}")
    print("  † 交付 = gold 证据段里检索**实际给到**的比例（= 引用了 + 检索到没引）。"
          "**这才是评检索栈的数。**\n"
          "    「检索到没引」高 → 提示词问题（agent 没把链条引全）；「缺口」高 → 真检索问题。"
          "两者混在一起会去调错的旋钮。\n"
          "  * 引用属实 = agent 自引的句子里，真能在它检索到的上下文中找到的比例。"
          "**它不接近 1.0 的话，任何基于引用的分都不可信**。")

    if diag:
        print(f"\n== [--diag] 固定预算召回 recall@B（只算交付顺序前 B 个片段；修「搜得多必然赢」）==")
        print(f"{'type':11s}{'n':>4}" + "".join(f"{'@' + str(b):>8}" for b in _BUDGETS))
        for t in _seen_types(ok):
            d = [r for r in ok if r["type"] == t and r["context_recall_fact"] == r["context_recall_fact"]]
            if d:
                print(f"{t:11s}{len(d):>4}"
                      + "".join(f"{mean([r['recall_at'][str(b)] for r in d]):>8.3f}" for b in _BUDGETS))
        print(f"\n== [--diag] 逐跳**边际**召回（第 i 跳新增了多少 gold 证据；≈0 = 那一跳只是在重复捞）==")
        print(f"{'type':11s}" + "".join(f"{'第' + str(i + 1) + '跳':>10}" for i in range(4)) + f"{'  样本(题数)':>16}")
        for t in _seen_types(ok):
            d = [r for r in ok if r["type"] == t and r.get("curve")]
            if not d:
                continue
            cells, ns = [], []
            for i in range(4):
                g = [r["curve"][i] - (r["curve"][i - 1] if i else 0.0) for r in d if len(r["curve"]) > i]
                cells.append(f"{mean(g):>+10.3f}" if g else f"{'—':>10}")
                ns.append(len(g))
            print(f"{t:11s}" + "".join(cells) + f"{str(ns):>16}")

    if n_err:
        print(f"\n⚠️ {n_err} 题网关连续失败，已记 errored 并排除（**不当 0 分**）。")
    zero = [r for r in ok if r["n_search"] == 0]
    if zero:
        print(f"⚠️ {len(zero)}/{len(ok)} 题 agent **一次都没检索** —— 这些题的召回=0 "
              f"是**模型没查**，不是检索器没捞到，别混为一谈。")
    nocite = [r for r in ok if r["n_search"] > 0 and r["n_cited"] == 0]
    if nocite:
        print(f"⚠️ {len(nocite)}/{len(ok)} 题查了却**没吐出 KEY EVIDENCE 块** —— 模型没遵守 prompts.py 策略5，"
              f"这些题在 eval_judge.py 里依据分会偏低，是**格式问题不是检索问题**。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="multihop-rag")
    ap.add_argument("--per-type", type=int, default=8, help="每种 question_type 抽几题（默认 8=32 题，快；25=100 更稳）")
    ap.add_argument("--reranker", default="bge", choices=["bge", "qwen"],
                    help="重排器：bge(默认,快) 或 qwen(Qwen3-Reranker-4B,instruction-aware,慢但 fact 召回更高)")
    ap.add_argument("--concurrency", type=int, default=4, help="evaluate 并发数（Qwen3-4B 慢+网关限流时降到 1）")
    ap.add_argument("--types", default="", help="只评这些题型（逗号分隔，如 temporal）。默认全四类。"
                                               "用于把样本预算压在**有区分度**的题型上——comparison/inference 已双双 1.00，多抽也没信息量")
    ap.add_argument("--model", default=None,
                    help="**答题**模型（缺省读 .env 的 RAG_MODEL）。裁判模型固定为 RAG_MODEL 不受影响——"
                         "换裁判等于换尺子，分数就不可比了")
    ap.add_argument("--local", metavar="OUT.jsonl", default=None,
                    help="只跑 agent + **确定性**指标，跳过 LLM 裁判与 LangSmith evaluate，每题**便宜 5 倍**，"
                         "并把逐题结果（含 answer / cited / contexts）写成 JSONL —— 这是 "
                         "`eval_judge.py` 的输入，也支持**同题配对**比较。")
    ap.add_argument("--benchmark", choices=["multihoprag", "musique"], default="multihoprag",
                    help="评测集。**musique 才是真多跳**（见 eval_benchmark_probe.py 实测：MultiHop-RAG "
                         "捷径率 99.3%%、MuSiQue 4hop 仅 69.0%%）。musique 只走 --local")
    ap.add_argument("--offline", action="store_true",
                    help="题目直接从 data/MultiHopRAG.json 抽，**完全不连 LangSmith**（只能配 --local）")
    ap.add_argument("--diag", action="store_true",
                    help="额外打印诊断表（recall@B 固定预算召回 + 逐跳边际召回）。默认不打："
                         "它们是**查 bug 的入口，不是头号指标**，头号指标在 eval_judge.py")
    args = ap.parse_args()

    if args.benchmark == "musique":
        types = [t.strip() for t in args.types.split(",") if t.strip()] or ["2hop", "3hop", "4hop"]
    else:
        types = [f"{t.strip()}_query" if not t.strip().endswith("_query") else t.strip()
                 for t in args.types.split(",") if t.strip()] or _TYPES

    qtype = _type_map()
    if args.benchmark == "musique":
        from rag.corpus_musique import load_questions
        if not args.local:
            raise SystemExit("--benchmark musique 只走 --local（题目不在 LangSmith 上）")
        examples, qtype = load_questions()
    elif args.offline:
        if not args.local:
            raise SystemExit("--offline 只对 --local 有意义（推 LangSmith 本来就要连 LangSmith）")
        examples = _offline_examples()
    else:
        examples = list(Client().list_examples(dataset_name=args.dataset))
    by_type: dict[str, list] = defaultdict(list)
    for e in examples:
        by_type[qtype.get(e.inputs.get("question"), "?")].append(e)

    picked = []
    for t in types:
        pool = by_type.get(t, [])
        step = max(1, len(pool) // args.per_type)
        picked += pool[::step][:args.per_type]
    print("抽样：" + " | ".join(f"{t.split('_')[0]} {min(len(by_type.get(t, [])), args.per_type)}"
                                for t in types) + f"  = {len(picked)} 题")

    if args.reranker == "qwen":
        from rag.reranker_qwen import QwenReranker
        retr = HybridRetriever(_corpus(args.benchmark), reranker=QwenReranker())
    else:
        retr = HybridRetriever(_corpus(args.benchmark))
    # RAG_DECOMPOSE=1 时套上「分解 → 逐子问句各自重排 → 轮流取名额」（默认关闭，见 retriever_decompose.py）
    retr = maybe_wrap(retr, llm=build_model(args.model))
    agent = build_agent(retr, args.model)
    judges = make_judges(build_judge()) if not args.local else {}   # 裁判恒为 RAG_JUDGE_MODEL，跨实验可比

    def _run_once(question: str) -> dict:
        """跑完一题，从**终态的完整消息列表**里取工具返回。

        ⚠️ 曾经的写法是在 `stream(stream_mode="values")` 的每个事件里只读 `messages[-1]`——
        **同一轮里的并行 tool 调用会被吞掉**（实测 2 次检索只捕到 1 次，`context_recall_fact`
        被报成 0.500 而真值是 1.000）。**这个 bug 只会让检索看起来更差**，于是"召回上不去"的
        表象里有一部分其实是评测自己造成的。终态里 messages 是全的，直接遍历它。
        """
        state = None
        for chunk in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
            state = chunk
        msgs = (state or {}).get("messages", [])
        contexts = [m.content or "" for m in msgs
                    if getattr(m, "type", "") == "tool" and getattr(m, "name", None) == "rag_search"]
        got: set[str] = set()
        for c in contexts:
            got |= set(_SRC.findall(c))
        answer = ""
        for m in msgs:
            if getattr(m, "type", "") == "ai" and (m.content or "").strip() and not getattr(m, "tool_calls", None):
                answer = m.content
        # 末尾的 KEY EVIDENCE 块切出来单独存：裁判判「依据够不够」要用它，
        # 而 answer 本身要**去掉**这块再交给裁判判对错，否则等于把证据当答案一起打分。
        answer, cited, unsup = _split_answer(answer)
        return {"answer": answer, "cited": cited, "n_unsupported": unsup,
                "sources": sorted(got), "contexts": contexts, "n_search": len(contexts)}

    def target(inputs: dict) -> dict:
        """网关 502/524 会整题打空 → **绝不能吞掉**：吞了就变成 correctness=0 计入均值，
        把两轮实验读成「模型变差了」（实测 qwen 轮 temporal 8 题空了 7 题）。这里长退避重试
        （Cloudflare 502 建议 retry_after=60，SDK 内置退避只有秒级、扛不住源站分钟级抽风），
        仍失败就 **raise** —— LangSmith 把该题标 errored 并**排除出均值**，而不是当 0 分。"""
        last: Exception | None = None
        for wait in (0, 30, 90):                      # 3 次尝试，总退避 ~2 分钟
            if wait:
                time.sleep(wait)
            try:
                return _run_once(inputs["question"])
            except Exception as e:                    # noqa: BLE001 —— 网关各类 5xx/超时
                last = e
                print(f"  ⚠️ 网关失败({type(e).__name__})，{wait or 0}s 后已重试…", flush=True)
        raise RuntimeError(f"网关连续 3 次失败，该题记 errored（不计入均值）：{type(last).__name__}: {str(last)[:120]}")

    def _adapt(name, fn):
        def scorer(run, example):
            ref = example.outputs.get("reference", "")
            if name == "correctness" and not ref:   # null_query 无 gold 答案 → 跳过 correctness
                return {"key": "correctness", "score": None}
            r = call_judge(name, fn, example.inputs["question"], run.outputs, ref)
            return {"key": name, "score": r["score"], "comment": r.get("comment")}
        return scorer

    def n_search(run, example) -> dict:
        """这一题 agent 到底检索了几次。

        没有它，"**模型压根没查**"与"**检索器没捞到**"在 `context_recall_fact` 上长得一模一样
        （都是 0.000），归因就会全错——实测 deepseek 有的题一次都不查（实验13 已记 它平均只查
        1.1~1.5 次）。**失败必须可见**，这是本项目第 N 次栽在同一件事上。"""
        return {"key": "n_search", "score": float((run.outputs or {}).get("n_search", 0))}

    if args.local:
        _run_local(picked, qtype, target, pathlib.Path(args.local), args.concurrency, args.diag)
        return

    evaluators = [_adapt(k, judges[k]) for k in SPECS] + [context_recall_fact, refused, n_search]
    results = evaluate(target, data=picked, evaluators=evaluators,
                       experiment_prefix=f"agentic-{args.reranker}-{(args.model or 'default').split('/')[-1]}-on-{args.dataset}",
                       max_concurrency=args.concurrency, client=Client())

    # 本地按 question_type 聚合（去 LangSmith 也能看，这里省得点）
    try:
        agg: dict = defaultdict(lambda: defaultdict(list))
        n_err = 0
        for row in results:
            t = qtype.get(row["example"].inputs.get("question"), "?").split("_")[0]
            if not (row["run"].outputs or {}).get("answer"):   # 网关打空/errored：不计入均值
                n_err += 1
                continue
            for er in (row["evaluation_results"]["results"] or []):
                if er.score is not None:
                    agg[t][er.key].append(er.score)
        metrics = ["correctness", "groundedness", "retrieval_relevance", "helpfulness",
                   "context_recall_fact", "refused", "n_search"]
        print("\n== 按 question_type 均值 ==")
        print(f"{'type':11s} " + " ".join(f"{m[:9]:>10}" for m in metrics) + "     n有效")
        for t in ["comparison", "inference", "temporal", "null"]:
            d = agg.get(t, {})
            n = max((len(v) for v in d.values()), default=0)
            print(f"{t:11s} " + " ".join(
                f"{sum(d[m]) / len(d[m]):>10.2f}" if d.get(m) else f"{'-':>10}" for m in metrics) + f"   {n:>5}")
        if n_err:
            print(f"\n⚠️ {n_err} 题网关连续失败被判 errored，已**排除**出均值（不当 0 分算）——"
                  f"若 n有效 明显小于抽样数，这轮的对比不可信，重跑。")
    except Exception as e:
        print(f"（本地聚合失败，去 LangSmith 看即可：{type(e).__name__}: {str(e)[:60]}）")
    print(f"\n[langsmith] agentic 实验已推到数据集 {args.dataset!r}（前缀 agentic-{args.reranker}-on-{args.dataset}）。")


if __name__ == "__main__":
    main()
