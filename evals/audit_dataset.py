"""**题目体检**：把评测集里的坏题标出来，产出可复核、可入库的 blocklist。

    python evals/audit_dataset.py --per-type 30 --auditor gpt-5.6-luna     # 生成 blocklist
    python evals/audit_dataset.py --report                                  # 只看已有 blocklist

━━━ 为什么需要它 ━━━

MuSiQue 的**结构**是干净的（实测 88/88 含 `#N` 占位符、266/266 跳的答案在支撑段有明文），
但**复合出来的问句**会坏。人工核对 10 道"gold 喂到嘴边 + 强制作答仍答错"的题，7 道是题本身的毛病：

  · **答案类型对不上问句**：问 "what **year**" 而 gold 是 `November 5`；问 "which **country**"
    而 gold 是 `Richland County`（county 被写成 country）
  · **占位符替换后语义不成立**：`How long had Nanjing been the capital city of Yangzhou?`
    —— 南京从来不是扬州的首府；gold「约 400 年」出自"南京作为中国首都"的表述
  · **gold 事实错误**：末跳问"波斯湾**以北**的 region"，gold 答 Kingdom of Saudi Arabia，
    而支撑段原文写的是 `north of **Yemen**` —— 沙特在波斯湾以南
  · **问句预设不成立**：问"另一个孩子"，而那位父亲有 5 个孩子，答案不唯一

这类题上「答对」是运气，模型答得越守规矩分越低。**不标出来，所有分数都掺着它们。**

━━━ 三条不许违反的设计约束 ━━━

1. **绝不静默剔除。** 评测表必须同时报「剔除前」和「剔除后」两个数 —— 否则这个脚本本身
   就变成刷分工具（本项目在实验8-9 上栽过一次 Goodhart，不能用一个新工具再栽一次）。
2. **审计模型 ≠ 裁判模型。** 同一个模型既判题好坏又判答案对错，两边的错误会相关：
   它看不懂的题会同时被标成"坏题"和"答错"，于是剔除坏题必然让分变好 —— 那是循环论证。
3. **审计自己要被验收。** `--seed-check` 拿人工核对过的种子标注算一致率；一致率不报，
   blocklist 就只是"另一个模型的意见"。

产物 `evals/blocklist_musique.json` **入库**：每条带判定理由，可独立复核、可推翻。
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import collections
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from evals.eval_agentic import load_benchmark, sample
from rag.llm import build_model

BLOCKLIST = _pl.Path(__file__).resolve().parent / "blocklist_musique.json"
_JSON = re.compile(r"\{.*\}", re.S)

# ── 人工核对过的种子（2026-08-04，我逐题读了问句 / gold / 支撑段原文）──────────
# 用来验收自动审计：它抓不抓得到这些、会不会误伤下面的 good 组。
SEED_BAD = {
    "2hop__278022_71701": "问 what year，gold 是 'November 5'（日期不是年份）",
    "3hop2__72083_92991_76291": "1946 年赢得国会的党何时控制众议院 —— 1947/2015 都成立，gold 只取 2015",
    "3hop2__93066_88342_47738": "gold 'the 2009 season' 与 2017 AL East 冠军的世界大赛记录对不上",
    "4hop1__60000_59747_211319_557671": "Birth of a Nation 有 1915/2016 两部，拍摄州不唯一，gold 链条指向第三个州",
    "4hop1__76111_624859_355213_203322": "Hungry Eyes 演唱者出生县与 gold 的 Cabarrus County 对不上",
    "4hop1__89632_32392_823060_610794": "问 which country，gold 是 'Richland County'（county 误写成 country）",
    "4hop3__316459_41402_146223_13584": "复合后问句不可读（'Besides areas of the country gaining control of Florida...'）",
    "3hop2__326964_811351_7713": "替换后成 'Nanjing 作为 Yangzhou 首府多久' —— 语义不成立",
    "4hop1__638988_17130_70784_61381": "gold 末跳称沙特在波斯湾以北，支撑段原文是 north of Yemen",
}
SEED_GOOD = {
    "3hop1__161433_33952_33939", "3hop1__354480_834494_33939",
    "4hop1__342858_131850_33952_33939", "3hop1__818422_160545_34751",
    "3hop2__127483_60649_10557", "4hop1__436202_765799_282674_759393",
}

# ── 第一层：确定性判据（免费，可复现，不需要模型）────────────────────────────
_TYPE_WORDS = {
    "year": (r"\bwhat year\b|\bwhich year\b|\bin what year\b",
             lambda a: bool(re.fullmatch(r"\D*\b(1\d{3}|20\d{2})\b\D*", a))),
    "country": (r"\bwhich country\b|\bwhat country\b",
                lambda a: not re.search(r"\b(county|city|state|province)\b", a, re.I)),
}


def deterministic_flags(q: str, ans: str) -> list[str]:
    """规则能抓到的坏题。**只报高把握的**，模糊的留给 LLM 审计。"""
    out = []
    for name, (pat, ok) in _TYPE_WORDS.items():
        if re.search(pat, q, re.I) and not ok(ans):
            out.append(f"答案类型不匹配：问句要 {name}，gold 是 {ans!r}")
    if ans.strip().lower() in q.lower():
        out.append("gold 答案字面出现在问句里（题目退化）")
    return out


_PROMPT = """You are auditing one question from a multi-hop QA benchmark. The question was \
built automatically by chaining single-hop questions, so it can come out broken. Your job is \
ONLY to judge the question and its gold answer — not to answer it.

QUESTION: {q}
GOLD ANSWER: {gold}

THE CHAIN THE DATASET USED (each step, its own answer, and the passage that supports it):
{chain}

First, the distinction that matters most. Two different things look similar and you must NOT \
confuse them:

  (a) BROKEN — the question cannot be answered as asked. The composed phrasing is incoherent or \
      self-contradictory, or it presupposes something false, or the gold answer contradicts the \
      passages, or the gold answer is the wrong kind of thing for what was asked.
  (b) MULTI-ANSWER — the question is perfectly coherent and the gold answer is right, but other \
      answers would be equally right ("which county borders X" when X borders four counties; \
      "the other child" when there are several). The question is fine; it just has more than one \
      correct answer.

(b) is NOT broken. Report it separately. Only (a) counts against the question.

Judge:

"well_posed" — 0 only if the composed question is incoherent, self-contradictory, or presupposes \
something false. A question that is long, obscure, hard, or has several valid answers is 1.

"gold_derivable" — 0 only if the passages contradict the gold answer, or say nothing that bears \
on the relation asked about. If a passage states the relation only in passing, inside an entry \
about some other entity, that still counts as derivable → 1.

"answer_type_ok" — 0 if the gold answer is the wrong kind of thing for the question (a \
"what year" question answered with a day-and-month; a "which country" question answered with a \
county). Otherwise 1.

"multi_answer" — 1 if several different answers would be equally correct for the question as \
asked. This does not make the question broken.

Default to 1 on all four when unsure. Flagging a merely hard question as broken is the worst \
error you can make here.

Reply with ONLY a JSON object, no prose and no code fence:
{{"well_posed": 1, "gold_derivable": 1, "answer_type_ok": 1, "multi_answer": 0, "reason": "<one short sentence; only if you gave a 0 to one of the first three>"}}"""


def _chain_text(r) -> str:
    para = {p["idx"]: p for p in r["paragraphs"]}
    out = []
    for i, h in enumerate(r["question_decomposition"], 1):
        p = para.get(h.get("paragraph_support_idx"))
        out.append(f"{i}. {h['question']}\n   answer: {h['answer']}\n"
                   f"   passage [{p['title'] if p else '?'}]: "
                   f"{((p or {}).get('paragraph_text') or '')[:700]}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="标出评测集里的坏题，产出可复核的 blocklist")
    ap.add_argument("--per-type", type=int, default=30, help="与 eval_agentic.py 同值 = 同一批题")
    ap.add_argument("--auditor", default=None,
                    help="审计模型。**必须与裁判不同**（否则坏题判定与答案判定的错误相关，"
                         "剔除坏题会自动让分变好，是循环论证）")
    ap.add_argument("--all", action="store_true", help="审计整个评测集，而不只是抽样的那批")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--report", action="store_true", help="不跑审计，只汇总已有 blocklist")
    args = ap.parse_args()

    from rag.corpus_musique import _raw
    raw = {r["id"]: r for r in _raw() if r.get("answerable")}

    if args.report:
        if not BLOCKLIST.exists():
            raise SystemExit(f"还没有 {BLOCKLIST}，先跑一次不带 --report 的。")
        bl = json.loads(BLOCKLIST.read_text(encoding="utf-8"))
        print(f"blocklist: {len(bl['bad'])} 题，审计模型 {bl['auditor']}，{bl['ts']}")
        c = collections.Counter(f["kind"] for f in bl["bad"].values())
        for k, v in c.most_common():
            print(f"   {k:16s}{v:>4}")
        return

    examples, qtype, _ = load_benchmark("musique")
    picked = examples if args.all else sample(examples, qtype, ["2hop", "3hop", "4hop"], args.per_type)
    ids = [str(e.id) for e in picked if str(e.id) in raw]
    print(f"审计 {len(ids)} 题，审计模型 = {args.auditor or '(endpoints.json 的 _roles.answer)'}")

    model = build_model(args.auditor)

    def one(i):
        r = raw[i]
        det = deterministic_flags(r["question"], r["answer"])
        prompt = _PROMPT.format(q=r["question"], gold=r["answer"], chain=_chain_text(r))
        got = None
        for wait in (0, 5, 20, 60):
            if wait:
                time.sleep(wait)
            try:
                m = _JSON.search(model.invoke(prompt).content or "")
                if m:
                    got = json.loads(m.group(0))
                    break
            except Exception:                                          # noqa: BLE001
                pass
        return i, det, got

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        res = []
        for n, x in enumerate(pool.map(one, ids), 1):
            res.append(x)
            if n % 20 == 0:
                print(f"  … {n}/{len(ids)} ({(time.time() - t0) / n:.1f}s/题)", flush=True)

    bad, multi, dead = {}, {}, []
    for i, det, got in res:
        if got is None:
            dead.append(i)                       # 审计失败 ≠ 坏题，**不进 blocklist**
            continue
        flags = list(det)
        for k, label in (("well_posed", "问句不成立"), ("gold_derivable", "gold 推不出"),
                         ("answer_type_ok", "答案类型不符")):
            if got.get(k) == 0:
                flags.append(label)
        rec = {"reason": str(got.get("reason", ""))[:200],
               "question": raw[i]["question"], "gold": raw[i]["answer"]}
        if flags:
            bad[i] = {"kind": flags[0], "flags": flags, **rec}
        # 多解题**不进 blocklist** —— 它不是坏题，是**判分口径**的问题（gold 只是若干正确答案之一）。
        # 混进来正是第一版审计误伤 67% 的原因：把"有多个正确答案"读成了"题目坏了"。
        elif got.get("multi_answer") == 1:
            multi[i] = rec

    # ── 审计的验收：拿人工种子算一致率 ──────────────────────────────────
    seen = set(ids)
    sb = [i for i in SEED_BAD if i in seen]
    sg = [i for i in SEED_GOOD if i in seen]
    hit = sum(i in bad for i in sb)
    fp = sum(i in bad for i in sg)
    print(f"\n{'=' * 78}\n审计结果\n{'=' * 78}")
    print(f"  标为**坏题**（进 blocklist，应剔除）：{len(bad)}/{len(ids)} = {len(bad) / len(ids):.1%}")
    print(f"  标为**多解题**（不剔除，是判分口径问题）：{len(multi)}/{len(ids)} = {len(multi) / len(ids):.1%}")
    if dead:
        print(f"  ⚠️ {len(dead)} 题审计调用失败 —— **不进 blocklist**（审计不出结果 ≠ 题是坏的）")
    print(f"\n  ▸ 验收（人工种子标注，2026-08-04 逐题读过原文）：")
    print(f"      人工判坏的 {len(sb)} 题，自动抓到 {hit}（召回 {hit / len(sb):.0%}）" if sb else "      (本批没有种子坏题)")
    print(f"      人工判好的 {len(sg)} 题，被误标 {fp}（误伤 {fp / len(sg):.0%}）" if sg else "      (本批没有种子好题)")
    print(f"      ⇒ 召回不高说明**漏标**（剔除后的分仍偏低）；误伤高说明**过度剔除**（剔除后的分虚高）。"
          f"\n        两个数都要报，blocklist 才能被别人信。")
    c = collections.Counter(v["kind"] for v in bad.values())
    print(f"\n  ▸ 坏题类型：")
    for k, v in c.most_common():
        print(f"      {k:16s}{v:>4}")
    print(f"\n  ▸ 抽三条看判得对不对：")
    for i, v in list(bad.items())[:3]:
        print(f"      {i}\n        Q: {v['question'][:100]}\n        gold: {v['gold']}\n"
              f"        判定: {v['flags']} — {v['reason'][:110]}")

    BLOCKLIST.write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "auditor": args.auditor,
        "benchmark": "musique", "n_audited": len(ids),
        "seed_recall": (hit / len(sb)) if sb else None,
        "seed_false_positive": (fp / len(sg)) if sg else None,
        "bad": bad, "multi_answer": multi}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {BLOCKLIST}（**入库**，每条带理由，可独立复核、可推翻）")
    print("⚠️ 用法：评测表必须**同时**报剔除前和剔除后两个数。静默剔除 = 用新工具刷分。")


if __name__ == "__main__":
    main()
