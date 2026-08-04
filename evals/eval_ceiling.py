"""**地板 / 实测 / 天花板** —— 三条线，回答"这套系统还剩多少可优化空间"。

    python evals/eval_ceiling.py --per-type 30 --model deepseek-v4-pro
    python evals/eval_ceiling.py --per-type 30 --dump runs/dumps/a.jsonl   # 把实测那条线一起画上
    # ★ 推荐：天花板改用「同一套 agent + gold」，比值就成了纯检索指标
    python evals/eval_ceiling.py --per-type 30 --dump runs/dumps/a_judged.jsonl \
                                 --oracle-dump runs/dumps/oracle_judged.jsonl

━━━ 为什么这是重构里最该先跑的一个脚本 ━━━

"指标提不上去、不知道哪里出了问题"这种感觉，很大一部分来自**没有参照系**：
MuSiQue 上只知道 correct=0.476，不知道 0.476 是好是坏、上面还剩多少。

    地板 closed   不给任何检索，模型凭参数化知识答 → **不是检索的功劳的那部分**
    实测 rag      现行系统（从 --dump 读，同一批题、同一个裁判）
    天花板 oracle 直接把 **gold 证据段**塞进上下文 → 检索若完美能到哪
                  （它同时也是"生成/推理端的上限"：oracle 都答不对的题，再强的检索也救不了）

**可达空间利用率 = (实测 − 地板) / (天花板 − 地板)。**
利用率已经 80%+ → 继续调检索是在榨最后一点，该换靶子；
利用率 30% → 检索确实是瓶颈，值得投；
天花板本身就低 → 瓶颈在推理/答题端或题目本身，调检索毫无意义。

━━━ `--oracle-dump`：让这个比值变成**纯检索指标** ━━━

内置的天花板用的是**一句话提示词**读 gold，而实测那条线走的是**完整 agent**。
两条线的系统不同 ⇒ 利用率里混了两样东西：**检索的缺口 + agent 外壳自己的代价**。

`--oracle-dump` 接一个 `eval_agentic.py --oracle` 跑出并判过分的 dump：那条线用的是
**同一个模型、同一套提示词、同一个多跳循环、同一套指标**，与实测的**唯一差别就是"证据从哪来"**。
于是比值退化成一个干净的问题：**这套检索栈交付了理论可得的百分之多少**（→「检索兑现率」）。

    deepseek：实测 0.478 / agent+gold 0.644 = **74%**

⚠️ 两条必须一起写的边界：
  · **它把答题模型的能力除掉了** —— 这正是它作为检索指标的优点，但也意味着**不能单独当头条**：
    换模型是独立的大杠杆（本项目实测 +0.148），会被这个比值整个藏起来。**绝对值与比值成对报。**
  · 分母**可以被超过**，见下。

⚠️ **oracle 不是真上限，两个方向都会偏**（2026-08-04 实测）：
  · **高估**：它给的是 gold 段并集、没有干扰项，检索永远做不到这么干净。
  · **低估**：gold 段有时**不含关系陈述**（问"演 Jarvis 的演员的配偶"，gold 只说两人合演过一部电影），
    而真实语料里**别的段落**说了。实测 10 题 RAG 答对而 oracle 答错，其中 6 题 oracle 直接回
    "NOT IN PASSAGES" —— **2hop 上实测 0.714 > 天花板 0.667，利用率算出来 118%。**
  ⇒ 把它读成「**gold-only 参照线**」，不是"检索完美能到哪"。
  **利用率 >100% 是个明确信号：那一档的 gold 标注不完整。**
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from evals.eval_agentic import load_benchmark, sample
from evals.eval_judge import _ci
from rag.llm import build_judge, build_model
from rag.runctx import fmt_meta, read_dump, snapshot

_JSON = re.compile(r"\{.*\}", re.S)

_CLOSED = """Answer this question from your own knowledge. You have no documents to consult.

QUESTION: {q}

Answer in one short sentence. If you do not know, say exactly "I do not know" — a wrong guess \
is worse than admitting the gap."""

_ORACLE = """Answer the question using ONLY the passages below.

PASSAGES:
{ctx}

QUESTION: {q}

Answer in one short sentence. If the passages do not contain the answer, say exactly \
"NOT IN PASSAGES"."""

# 第三条线：**同样的 gold 段，但不许拒答**。
# 起因：oracle 大量回 "NOT IN PASSAGES"，逐条查是因为 MuSiQue 的 gold 段常常**含有答案实体、
# 却没有陈述那条关系**（典型：问"演 Jarvis 的演员的配偶"，gold 段只说 Bettany 和 Connelly
# 一起演了《美丽心灵》，从没说他们是夫妻）。这种题上**"答对"和"有依据"是互斥的** ——
# 拒答是忠诚且正确的行为，却被 correct 记 0。
# oracle_forced − oracle 就是这一块的大小：它把「推不出来」和「推得出来但不肯断言」分开。
_FORCED = """Answer the question using the passages below as your main source. You may also use \
your own knowledge to connect them.

PASSAGES:
{ctx}

QUESTION: {q}

You MUST give a concrete answer in one short sentence. Do not refuse and do not say the passages \
are insufficient — if they do not state it outright, give your single best guess."""

_GRADE = """Does the SYSTEM ANSWER match the GOLD ANSWER in substance? Wording, extra explanation \
and formatting do not matter. A refusal / "I do not know" / "NOT IN PASSAGES" scores 0.0.

QUESTION: {q}
GOLD ANSWER: {gold}
SYSTEM ANSWER: {a}

Reply with ONLY a JSON object: {{"correct": 0.0}}"""


def _retry(fn, *a):
    for wait in (0, 5, 20, 60):
        if wait:
            time.sleep(wait)
        try:
            return fn(*a)
        except Exception:                                          # noqa: BLE001
            pass
    return None


def _grade(judge, q: str, gold: str, ans: str | None) -> float | None:
    """→ correct，**拿不到答案一律返回 None（排除出均值），绝不当 0 分**。

    ⚠️ 这里踩过一次：初版只判 `ans is None`，而网关偶尔返回**空字符串**——空串被送去判分、
    裁判打 0、计入均值，于是"网关抽风"被读成"模型答不出来"。实测 90 题里有 1 题如此
    （`3hop1__622145_42197_18397`，oracle/forced 都被记 0）。这是本项目第三次栽在同一件事上
    （实验12 的 `except: return 空`、实验21 的吞并行 tool 调用）——**失败必须可见，且不能长得像失败之外的东西。**
    """
    if not (ans or "").strip():
        return None
    txt = _retry(lambda: judge.invoke(_GRADE.format(q=q, gold=gold, a=ans[:2000])).content)
    m = _JSON.search(txt or "")
    if not m:
        return None
    try:
        v = json.loads(m.group(0)).get("correct")
        return min(1.0, max(0.0, float(v))) if isinstance(v, (int, float)) else None
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="地板(无检索) / 实测(--dump) / 天花板(gold 上下文) 三条线")
    ap.add_argument("--benchmark", choices=["musique", "multihoprag"], default="musique")
    ap.add_argument("--per-type", type=int, default=30, help="必须与 eval_agentic.py 同值，否则不是同一批题")
    ap.add_argument("--types", default="")
    ap.add_argument("--model", default=None, help="答题模型（地板/天花板两条线用它；应与被评系统同一个）")
    ap.add_argument("--dump", default=None, help="现行系统的 dump —— 实测那条线从它的 judge 结果读；"
                                                 "没有就只画地板和天花板")
    ap.add_argument("--oracle-dump", default=None, metavar="JUDGED.jsonl",
                    help="★ **推荐**：用 `eval_agentic.py --oracle` 跑出并判过分的 dump 当天花板，"
                         "替代内置的一句话提示词。这样天花板与实测**走的是同一套 agent**，"
                         "唯一差别只剩「证据从哪来」⇒ 那个比值成为**纯检索指标**（见文件头）。"
                         "给了它就不再调用 oracle / 强制作答两条线，本轮只算地板。")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None, help="逐题结果 JSONL")
    args = ap.parse_args()

    default_types = {"musique": ["2hop", "3hop", "4hop"],
                     "multihoprag": ["comparison_query", "inference_query", "temporal_query"]}
    types = [t.strip() for t in args.types.split(",") if t.strip()] or default_types[args.benchmark]
    examples, qtype, _ = load_benchmark(args.benchmark)
    picked = sample(examples, qtype, types, args.per_type)
    model, judge = build_model(args.model), build_judge()

    # ── 天花板从哪来：外部 agent+gold dump（推荐）还是内置的一句话提示词 ──────────
    ext_oracle: dict[str, float] = {}
    if args.oracle_dump:
        ometa, orows = read_dump(args.oracle_dump)
        if not ometa.get("oracle"):
            raise SystemExit(f"⛔ {args.oracle_dump} 的 __meta__ 里 oracle 不为真 —— "
                             f"它不是 `eval_agentic.py --oracle` 跑出来的，不能当天花板。")
        for r in orows:
            j = r.get("judge")
            if isinstance(j, dict) and j.get("correct") is not None:
                ext_oracle[r["example_id"]] = 1.0 if j["correct"] else 0.0
        if not ext_oracle:
            raise SystemExit(f"⛔ {args.oracle_dump} 里没有 judge 结果 —— 先跑 eval_judge.py。")
        print(f"天花板那条线来自 {args.oracle_dump}（{fmt_meta(ometa)}）\n"
              f"   ⇒ 天花板与实测**同一套 agent**，比值是**纯检索指标**（检索兑现率）。")

    n_lines = 1 if ext_oracle else 3
    meta = snapshot(benchmark=args.benchmark, per_type=args.per_type, answer_model=args.model,
                    oracle_dump=args.oracle_dump,
                    lines=["closed"] + ([] if ext_oracle else ["oracle", "oracle_forced"]))
    print(f"地板/天花板：{len(picked)} 题 × {n_lines} 条线 = {len(picked) * n_lines} 次答题 + 同样多次判分")
    print(f"   {fmt_meta(meta)}")

    def one(ex) -> dict:
        q, gold = ex.inputs["question"], ex.outputs["reference"]
        ctx = "\n\n".join(ex.outputs.get("reference_contexts") or [])
        closed = _retry(lambda: model.invoke(_CLOSED.format(q=q)).content)
        row = {"example_id": str(ex.id), "question": q, "type": qtype.get(q, "?").split("_")[0],
               "closed": _grade(judge, q, gold, closed), "closed_answer": (closed or "")[:300]}
        if ext_oracle:                       # 天花板由外部 dump 提供，这里不再花钱重跑
            return row | {"oracle": ext_oracle.get(str(ex.id)), "forced": None,
                          "oracle_answer": "(来自 --oracle-dump)", "forced_answer": ""}
        oracle = _retry(lambda: model.invoke(_ORACLE.format(q=q, ctx=ctx[:40000])).content)
        forced = _retry(lambda: model.invoke(_FORCED.format(q=q, ctx=ctx[:40000])).content)
        return row | {"oracle": _grade(judge, q, gold, oracle), "forced": _grade(judge, q, gold, forced),
                      "oracle_answer": (oracle or "")[:300], "forced_answer": (forced or "")[:300]}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = []
        for i, r in enumerate(pool.map(one, picked), start=1):
            rows.append(r)
            if i % 10 == 0:
                print(f"  … {i}/{len(picked)} ({(time.time() - t0) / i:.1f}s/题)", flush=True)

    actual: dict[str, float] = {}
    if args.dump:
        dmeta, drows = read_dump(args.dump)
        for r in drows:
            j = r.get("judge")
            if j and j.get("correct") is not None:
                actual[r["example_id"]] = j["correct"]
        if not actual:
            print(f"\n⚠️ {args.dump} 里没有 judge 结果 —— 先跑 `eval_judge.py {args.dump} --out <判分.jsonl>`，"
                  f"再把**那个**判分文件传给 --dump。本轮只画地板/天花板。")
        else:
            print(f"\n实测那条线来自 {args.dump}（{fmt_meta(dmeta)}）")

    if args.out:
        _pl.Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                      encoding="utf-8")

    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")            # noqa: E731
    ok = lambda xs: [x for x in xs if x is not None]                          # noqa: E731
    order = ["2hop", "3hop", "4hop", "comparison", "inference", "temporal"]
    tps = sorted({r["type"] for r in rows}, key=lambda t: (order.index(t) if t in order else 99, t))
    has_forced = any(r.get("forced") is not None for r in rows)
    # 天花板走的是同一套 agent 时，(实测−地板)/(天花板−地板) 的唯一自变量就是"证据从哪来"
    # ⇒ 它是**纯检索指标**，改叫「检索兑现率」；否则它还混着 agent 外壳的代价。
    util_label, ceil_label = (("检索兑现率", "天花板(agent+gold)") if ext_oracle
                              else ("可达空间利用率", "天花板(gold)"))
    print(f"\n{'=' * 78}\ncorrect 的{'两' if ext_oracle else '三'}条线（同一批题、同一个裁判）\n{'=' * 78}")
    print(f"{'type':10s}{'n':>4}{'地板(无检索)':>14}{'实测(现行)':>13}{ceil_label:>{14 + len(ceil_label) - 10}}"
          + (f"{'└强制作答':>13}" if has_forced else "") + f"{'★检索还能买到':>15}{util_label:>16}")
    for t in [*tps, "全部"]:
        d = [r for r in rows if t == "全部" or r["type"] == t]
        if not d:
            continue
        lo, hi = mean(ok([r["closed"] for r in d])), mean(ok([r["oracle"] for r in d]))
        mid = mean([actual[r["example_id"]] for r in d if r["example_id"] in actual]) if actual else float("nan")
        util = (mid - lo) / (hi - lo) if actual and hi > lo and mid == mid else float("nan")
        fc = mean(ok([r.get("forced") for r in d]))
        print(f"{t:10s}{len(d):>4}{lo:>14.3f}"
              + (f"{mid:>13.3f}" if mid == mid else f"{'—':>13}")
              + f"{hi:>14.3f}" + (f"{fc:>13.3f}" if has_forced else "")
              + (f"{hi - mid:>+15.3f}" if mid == mid else f"{'—':>15}")
              + (f"{util:>15.0%}" if util == util else f"{'—':>16}"))

    print(f"\n  ★ **「检索还能买到」= 天花板 − 实测**，这是决定「要不要继续投检索」的那个数。\n"
          f"    它是**两个直接测量的差**，不除以任何东西 —— 优先看它，不要优先看{util_label}。")
    # 地板线每次都要重新生成（模型现答一遍），除裁判噪声外还有**生成噪声**；
    # 而 util 的分母是 (天花板−地板)，把这个噪声整个放进了除数。实测：同配置重跑一次，
    # 3hop 地板 0.400→0.214（逐题一致率 66.7%），同一档的利用率就从 0% 变成 44%。
    small = [t for t in tps if len([r for r in rows if r["type"] == t]) < 50]
    if small and actual:
        print(f"  ⚠️ **分档的{util_label}在 n<50 时不可信**：它的分母是（天花板−地板）——"
              f"**两个噪声量的差**。\n     地板线每轮都要重新生成，除裁判噪声外还有生成噪声；"
              f"实测同配置重跑一次，3hop 地板\n     0.400→0.214，那一档的{util_label}就从 0% 变成 44%。"
              f"**分档只读「检索还能买到」那一列。**")

    lo_a, hi_a = mean(ok([r["closed"] for r in rows])), mean(ok([r["oracle"] for r in rows]))
    m, l, h = _ci([r["oracle"] - r["closed"] for r in rows
                   if r["oracle"] is not None and r["closed"] is not None])
    print(f"\n  可优化区间 = 天花板 − 地板 = {hi_a - lo_a:+.3f}（配对 {m:+.3f} [{l:+.3f}, {h:+.3f}]）")
    for t in tps:
        d=[r for r in rows if r["type"]==t]
        lo2,hi2=mean(ok([r["closed"] for r in d])),mean(ok([r["oracle"] for r in d]))
        mid2=mean([actual[r["example_id"]] for r in d if r["example_id"] in actual]) if actual else float("nan")
        if mid2==mid2 and hi2>lo2 and (mid2-lo2)/(hi2-lo2)>1.0:
            print(f"\n  ⛔ {t} 的{util_label} >100%(实测 {mid2:.3f} > 天花板 {hi2:.3f}) —— "
                  f"**不是系统超神，是那一档的 gold 标注不完整**：\n"
                  f"     gold 段没写那条关系，而真实语料里别的段落写了，检索反而补上了。"
                  f"这一档的天花板要当**下界**读。")
    print(f"  ▸ 区间窄 → 这个数据集在这个模型上**本来就没多少可做**，再调检索也涨不动；\n"
          f"  ▸ 天花板本身低 → 瓶颈在**答题/推理端或题目**，不在检索（gold 都喂到嘴边了还答不对）；\n"
          f"  ▸ {util_label}高 → 检索已接近它能给的上限，该换靶子而不是继续拧旋钮。")
    if ext_oracle:
        print(f"  ▸ ⚠️ **绝对值和{util_label}必须成对报。**{util_label}把答题模型的能力除掉了 —— "
              f"这正是它作为\n     检索指标的优点，但也意味着它**不能单独当项目头条**"
              f"（换模型是独立的大杠杆，会被这个比值藏起来）。")
    if has_forced:
        fm, fl, fh = _ci([r["forced"] - r["oracle"] for r in rows
                          if r.get("forced") is not None and r["oracle"] is not None])
        print(f"\n  ▸ **强制作答 − 允许拒答 = {fm:+.3f} [{fl:+.3f}, {fh:+.3f}]** —— 同样的 gold 段，"
              f"唯一差别是准不准拒答。\n"
              f"    这一块是「**推得出来但不肯断言**」的题；它之所以存在，是因为 MuSiQue 的 gold 段"
              f"常常\n    **含有答案实体、却没有陈述那条关系**。在这些题上「答对」和「有依据」是"
              f"**互斥**的 ——\n    拒答是忠诚且正确的行为，却被 correct 记 0。"
              f"⇒ 报 correct 时必须连同这条一起报。")
    nk = sum(1 for r in rows if r["oracle"] == 0)
    if nk:
        print(f"\n  ▸ {nk}/{len(rows)} 题**连 gold 上下文都答不对** —— 这些题检索再完美也拿不到分，"
              f"是推理端/题目本身的上限。抽一条看看：")
        for r in rows:
            if r["oracle"] == 0:
                print(f"      Q: {r['question'][:90]}\n      oracle 答: {r['oracle_answer'][:120]}")
                break


if __name__ == "__main__":
    main()
