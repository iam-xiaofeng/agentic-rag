"""**评测集探针**：判一个"多跳"数据集是不是真多跳 —— 只看结构，不看名字、不看论文。

    python eval_benchmark_probe.py                    # 三个数据集横向对比
    python eval_benchmark_probe.py --n 2000

起因（实验25）：MultiHop-RAG 名字里的 "MultiHop" 指**证据跨多篇文章**，不指**推理要分步**。
99.3% 的问句已把每篇 gold 文章的出处点名 —— 一次宽检索就能同时覆盖，agentic 的多跳能力
在它上面**结构性地无处发挥**。踩过这个坑之后，任何候选评测集都必须先过这把尺子。

━━━ 三个判据 ━━━

**① 桥接率（最重要）**
   桥接式多跳的定义：第二跳搜什么，必须等第一跳的结果出来才知道。
   机器可检的等价物：**是否存在一篇 gold 文档，它的标题无法从问句里推出来**。
     - 推得出来 → 问句已经点名了检索目标 → 一次检索就能覆盖 → **不是桥接**
     - 推不出来 → 必须先查别的、才知道要查它 → **是桥接**
   判定**偏向保守**（宽松匹配标题 → 更容易判成"非桥接"），所以报出来的桥接率是**下界**。

**② 猜测下限**
   答案分布集中度。Yes/No 题或答案高度集中的数据集，一个不检索的空壳就能拿高分，
   区分度被吃掉。报最高频答案占比与前 3 名合计。

**③ 证据冗余**
   单条支撑句是否**独立包含答案**。若是，则"多跳"是装饰性的 —— 拿到任意一条就能答。

━━━ 已知结论（--n 2000 实测，见 EXPERIMENTS 实验25/26）━━━
   不要因为论文说它是多跳就信；**HotpotQA 的 comparison 子集两个实体全在问句里**，
   与 MultiHop-RAG 同病。要看的是 bridge 子集。
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
from collections import Counter

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

_PAREN = re.compile(r"\([^)]*\)")
_NONWORD = re.compile(r"[^a-z0-9 ]+")
# 标题里对"能不能从问句推出来"没有信息量的词，匹配时忽略
_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "for", "film",
         "movie", "album", "song", "band", "novel", "book", "series", "tv", "season"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", _NONWORD.sub(" ", (s or "").lower())).strip()


def _title_in_question(title: str, question: str) -> bool:
    """标题能不能从问句里推出来 —— **故意放宽**，宁可误判成"非桥接"。

    放宽的两处：① 去掉括号消歧后整串匹配；② 退一步，标题的**实词全都**出现在问句里也算。
    这样得到的桥接率是**下界**：真桥接只会比报出来的多，不会少。

    ⚠️ **只对「标题即检索目标」的数据集有意义**（维基类：标题=实体名）。
    对新闻类（MultiHop-RAG）无意义 —— 那里的标题是新闻大标题（"Chiefs vs. Packers live score…"），
    问句用「出处+主题」指代它（"the Sporting News article on…"），照抄标题的概率约等于 0，
    于是这个判据会把它**一律判成桥接**，与事实相反。跨数据集要看下面的 `_single_shot`。
    """
    q, t = _norm(question), _norm(_PAREN.sub(" ", title))
    if not t:
        return True
    if t in q:
        return True
    toks = [w for w in t.split() if w not in _STOP and len(w) > 2]
    return bool(toks) and all(w in q for w in toks)


def _bm25_rank(query: str, docs: list[str]) -> list[int]:
    """例内 BM25 排序，返回 docs 的下标（相关性降序）。自带实现，避免依赖差异。"""
    import math
    toks = [_norm(d).split() for d in docs]
    avg = sum(len(t) for t in toks) / max(1, len(toks))
    df: Counter = Counter()
    for t in toks:
        df.update(set(t))
    n, k1, b = len(toks), 1.5, 0.75
    scores = []
    for t in toks:
        tf, L, s = Counter(t), len(t), 0.0
        for w in set(_norm(query).split()):
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * L / avg))
        scores.append(s)
    return sorted(range(n), key=lambda i: -scores[i])


def _single_shot(rows: list[dict], rng_seed: int = 0) -> tuple[float, float, float]:
    """**跨数据集唯一可比的判据**：只拿原问句去检索，能拿到什么？

    做法：为每题拼一个**同样大小**的候选池（gold + 从别题借来的干扰项，共 `POOL` 篇），
    例内 BM25 排序，取前 `TOPK`。不依赖任何外部索引，三个数据集口径完全一致。

    返回 (全部 gold 都进前 TOPK, 至少 1 篇进前 TOPK, **含答案那篇**进前 TOPK)。

    第三个是**捷径率**，也是最要命的那个：如果光靠原问句就能直接够到含答案的文档，
    那么中间那些"跳"是装饰性的 —— 模型可以跳过推理链直接命中。**它高 = 这个数据集能被单跳刷掉。**
    ⚠️ 干扰项是**随机借**的，比 HotpotQA 原生的对抗式干扰项容易 → 三个数都偏乐观；
    但三边同样乐观，横比仍成立。
    """
    import random
    POOL, TOPK = 20, 5
    rng = random.Random(rng_seed)
    allx = [(x["title"], x["text"]) for r in rows for x in r["docs"]]
    full = any_ = short = n_short = 0
    for r in rows:
        gold = [(x["title"], x["text"]) for x in r["docs"]]
        gs = {t for t, _ in gold}
        cand = list(gold)
        while len(cand) < POOL:                       # 借别题的文档当干扰项
            t, x = allx[rng.randrange(len(allx))]
            if t not in gs:
                cand.append((t, x))
        rng.shuffle(cand)
        top = {cand[i][0] for i in _bm25_rank(r["question"], [f"{t} {x}" for t, x in cand])[:TOPK]}
        full += gs <= top
        any_ += bool(gs & top)
        a = _norm(r["answer"])
        if a and a not in ("yes", "no"):              # Yes/No 题没有"含答案的那篇"
            hit = {t for t, x in gold if a in _norm(x)}
            if hit:
                n_short += 1
                short += bool(hit & top)
    n = len(rows)
    return full / n, any_ / n, (short / n_short if n_short else float("nan"))




# ── 适配器：把各数据集拍成统一的 {question, answer, docs:[{title,text}], type} ────────────

def _load_multihoprag(n: int) -> list[dict]:
    raw = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    out = []
    for r in raw:
        ev = [e for e in (r.get("evidence_list") or []) if e.get("fact")]
        if not ev or r.get("question_type") == "null_query":
            continue
        out.append({"question": r["query"], "answer": (r.get("answer") or "").strip(),
                    "docs": [{"title": e.get("title", ""), "text": e["fact"]} for e in ev],
                    "type": r.get("question_type", "?").replace("_query", "")})
    return out[:n]


def _load_musique(n: int) -> list[dict]:
    from datasets import load_dataset
    out = []
    for r in load_dataset("dgslibisey/MuSiQue", split="validation"):
        if not r.get("answerable"):
            continue
        para = {p["idx"]: p for p in r["paragraphs"]}
        docs, chain = [], 0
        for h in r["question_decomposition"]:
            p = para.get(h.get("paragraph_support_idx"))
            if p:
                docs.append({"title": p["title"], "text": p["paragraph_text"]})
            if "#" in (h.get("question") or ""):      # 子问句引用了前一跳的答案
                chain += 1
        if docs:
            out.append({"question": r["question"], "answer": r["answer"], "docs": docs,
                        "type": f"{len(r['question_decomposition'])}hop", "chain": chain})
        if len(out) >= n:
            break
    return out


def _load_hotpot(n: int) -> list[dict]:
    from datasets import load_dataset
    out = []
    for r in load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"):
        ctx = dict(zip(r["context"]["title"], r["context"]["sentences"]))
        sf = r["supporting_facts"]
        docs = {}
        for t, sid in zip(sf["title"], sf["sent_id"]):
            sents = ctx.get(t) or []
            if sid < len(sents):
                docs.setdefault(t, []).append(sents[sid])
        if docs:
            out.append({"question": r["question"], "answer": r["answer"],
                        "docs": [{"title": t, "text": " ".join(s)} for t, s in docs.items()],
                        "type": r.get("type", "?")})
        if len(out) >= n:
            break
    return out


LOADERS = {"MultiHop-RAG": (_load_multihoprag, False), "MuSiQue": (_load_musique, True),
           "HotpotQA": (_load_hotpot, True)}


def _probe(name: str, rows: list[dict], wiki: bool) -> None:
    by = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r)
    print(f"\n{'=' * 112}\n{name}   n={len(rows)}\n{'=' * 112}")
    print(f"{'子集':16s}{'n':>6}{'⚑捷径率':>10}{'一次拿全':>10}{'桥接率':>9}{'gold篇':>7}"
          f"{'最高频答案':>21}{'占比':>7}{'前3合计':>8}")
    for t in [*sorted(by), "**全部**"]:
        d = rows if t == "**全部**" else by[t]
        full, any_, short = _single_shot(d)
        bridge = sum(1 for r in d if any(not _title_in_question(x["title"], r["question"])
                                         for x in r["docs"])) / len(d)
        c = Counter(_norm(r["answer"]) for r in d)
        top = c.most_common(3)
        # 证据冗余：某一条支撑段落**单独**含答案（答案是 yes/no 时无意义，标 —）
        n_doc = sum(len(r["docs"]) for r in d) / len(d)
        print(f"{t:16s}{len(d):>6}"
              + (f"{short:>10.1%}" if short == short else f"{'—':>10}")
              + f"{full:>10.1%}"
              + (f"{bridge:>9.1%}" if wiki else f"{'n/a*':>9}")
              + f"{n_doc:>7.2f}{top[0][0][:19]!r:>21}"
              f"{top[0][1] / len(d):>7.1%}{sum(x[1] for x in top) / len(d):>8.1%}")
    if not wiki:
        print("  * 桥接率对新闻类不适用：标题是新闻大标题，问句用「出处+主题」指代它，"
              "照抄标题的概率≈0 →\n    这个判据会把它一律判成桥接，与事实相反。看「一次拿全」。")
    chain = [r for r in rows if "chain" in r]
    if chain:
        print(f"\n  子问句显式引用前一跳答案（`#1` 占位符）的题：占 "
              f"{sum(1 for r in chain if r['chain'] > 0) / len(chain):.1%} "
              f"—— 这是**构造上保证**的桥接，不是推断出来的。")


def main() -> None:
    ap = argparse.ArgumentParser(description="判一个「多跳」数据集是不是真多跳（只看结构）")
    ap.add_argument("--n", type=int, default=2000, help="每个数据集最多取多少题")
    ap.add_argument("--only", default=None, help="只跑其中一个（名字见 LOADERS）")
    args = ap.parse_args()

    for name, (fn, wiki) in LOADERS.items():
        if args.only and args.only.lower() not in name.lower():
            continue
        try:
            _probe(name, fn(args.n), wiki)
        except Exception as e:                        # noqa: BLE001 —— 某个源挂了不该拖垮其余
            print(f"\n⚠️ {name} 加载失败：{type(e).__name__} {str(e)[:160]}")
    print("\n" + "─" * 112)
    print("三边口径完全一致：例内 BM25、候选池统一 20 篇（gold + 从别题随机借的干扰项）、取 top5。")
    print("**⚑捷径率** = 光靠**原问句**就够到了**含答案那篇**文档的比例 —— 中间的跳可以被跳过。")
    print("  **高 = 这个数据集能被单跳刷掉**，是判「真不真多跳」最要命的一个数。Yes/No 题不适用，标 —。")
    print("**一次拿全** = gold 文档全部进 top5。低 → 一次检索拿不全 → 真的需要后续跳。")
    print("⚠️ 干扰项是随机借的，比 HotpotQA 原生对抗式干扰项容易 → 三个数都**偏乐观**（真捷径率更低、")
    print("   真难度更高）；但三边同样乐观，横比成立。")
    print("**桥接率** = 存在一篇 gold 文档，其标题无法从问句推出（判定偏保守，是**下界**）。仅适用于维基类。")


if __name__ == "__main__":
    main()

