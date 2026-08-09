"""查询侧诊断：**问句怎么写**比**用什么 reranker** 更决定召回吗？（确定性 + 几乎免费）

EXPERIMENTS 实验11-12 把检索侧走到了头：候选池覆盖能补到 100%、换最强的 instruction-aware reranker
（Qwen3-4B）也只能把 temporal 的交付从 0.42 抬到 0.53——**证据在池子里，就是排不出来**。
原因是 reranker 无论多强都只是 `(query, doc) → 分数`：**query 里没有的信号，打分器变不出来**。
temporal 的问句是**元层面追问**（"A 在 X 日的报道与 B 在 Y 日的报道是否一致？"），
gold 证据却是**具体事实陈述**，两者不共享词汇也不共享语义焦点。

所以本脚本把变量换成**问句本身**，对比四种策略（检索器/reranker 全程不变，唯一变量 = 用什么 query 去查）：

  1. baseline        原问句直接查
  2. multiquery      LLM 生成 N 个**同义改写**（LangChain MultiQueryRetriever 的做法）→ 各自召回 → RRF 合并
  3. decompose       LLM **分解**成事实性子问句（"A 在 X 日报道了什么"）→ 各自召回 → RRF 合并 → 用原问句重排
  4. decompose-each  同上分解，但**每个子问句各自重排**取名额再拼（真正的"分而治之"，不让原问句的
                     元层面措辞再把证据压回去）

3 vs 4 是关键对照：如果 3 不涨而 4 涨，说明**光换召回不够，重排也必须跟着子问句走**。
2 vs 3 回答"同义改写够不够，还是必须降到事实层"。
**0（不重排）是实验14 用的那条"天花板"基线**，必须一起量——实验20 证明当年那条基线本身
是 15 题的点估计、落在噪声里，所以本轮的判据改成 **相对 baseline 的配对 bootstrap 置信区间**：
区间跨 0 就是"测不出差异"，不许再拿点估计下结论。

    python eval_query.py --n 50 --types temporal,comparison,inference --model deepseek-v4-pro
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import pathlib
import re
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rag.corpus_multihop import CHUNK_SIZE, load_corpus
from rag.llm import build_model
from rag.retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
# 交付深度按 chunk 对齐：chunk=600 时 k=16/32 才与旧配置(1200) 的 k=8/16 交付同样多字符。
_KS = (8, 16, 32) if CHUNK_SIZE <= 700 else (4, 8, 16)
_STRATEGIES = ("0 不重排", "1 baseline", "2 multiquery", "3 decompose", "4 decompose-each")

_MULTI = ("Rewrite the following question in {n} different ways to maximize retrieval coverage. "
          "Keep the same meaning; vary wording and phrasing. "
          "Output ONLY the {n} rewrites, one per line, no numbering.\n\nQuestion: {q}")
_DECOMP = ("Decompose the following multi-hop question into {n} INDEPENDENT, FACT-SEEKING sub-questions "
           "that can each be answered by a single news article. Do NOT ask about consistency, comparison, "
           "or agreement between reports — instead ask directly about the underlying FACTS each report states "
           "(e.g. 'What did <outlet> report about <entity> on <date>?'). "
           "Output ONLY the {n} sub-questions, one per line, no numbering.\n\nQuestion: {q}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _lines(text: str, n: int) -> list[str]:
    out = [re.sub(r"^\s*[-*\d.)]+\s*", "", ln).strip() for ln in (text or "").splitlines()]
    return [ln for ln in out if len(ln) > 8][:n]


def _rrf(lists: list[list], k: int = 60) -> list:
    """把多个子问句各自的候选列表按 RRF 合并（只看排名，跨问句可比）。"""
    fused: dict[str, list] = {}
    for lst in lists:
        for rank, d in enumerate(lst, start=1):
            slot = fused.setdefault(d.id, [d, 0.0])
            slot[1] += 1.0 / (k + rank)
    return [d for d, _ in sorted(fused.values(), key=lambda v: v[1], reverse=True)]


def _rerank(retr, query: str, docs: list) -> list:
    if not docs:
        return []
    sc = np.asarray(retr.reranker.predict([(query, d.text) for d in docs]))
    return [docs[i] for i in np.argsort(-sc)]


def _score(docs: list, facts: list[str], ks=_KS) -> dict:
    """fact 级召回。这里沿用「证据句前 120 字符」口径——实验19 ① 指出它**看不见证据被切断**，
    因而不适合跨 chunk 大小比较；但本脚本所有策略**共用同一份切分**，这个偏差对各策略等量作用，
    故对策略间比较是中立的。"""
    out = {}
    for k in ks:
        blob = _norm(" ".join(d.text for d in docs[:k]))
        out[f"@{k}"] = sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts)
    blob = _norm(" ".join(d.text for d in docs))
    out["pool"] = sum(1 for f in facts if _norm(f)[:120] in blob) / len(facts)
    return out


def _paired_ci(base: list[float], alt: list[float], boots: int = 20000) -> tuple[float, float, float]:
    """alt − base 的**配对**均值及 95% bootstrap 置信区间（同一批题，消掉题目难易这个方差源）。"""
    d = np.asarray(alt, dtype=float) - np.asarray(base, dtype=float)
    if not len(d):
        return (float("nan"),) * 3
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), size=(boots, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="每种题型抽几题（实验20 的教训：15 题分不出 0.1 的效应）")
    ap.add_argument("--types", default="temporal", help="题型，逗号分隔（comparison/inference/temporal）")
    ap.add_argument("--sub", type=int, default=3, help="生成几个改写/子问句")
    ap.add_argument("--model", default=None, help="生成改写/子问句的模型（缺省读 .env 的 RAG_MODEL）")
    ap.add_argument("--reranker", default="bge", choices=["bge", "qwen"],
                    help="重排器；用 qwen 可测『查询分解 × 强 reranker』能否叠加")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="改写/分解阶段的并发数。这阶段是**纯网关等待**，串行做会把整轮时间乘以 5；"
                         "但并发开太大会撞网关的并发上限（实测 6 会吃 429），默认 3。重排在 GPU 上仍是串行的")
    ap.add_argument("--cache", default=".cache/query_rewrites.json",
                    help="改写结果的落盘缓存（按 问句+子问句数 哈希）。网关很不稳，重跑时不该再向它要一遍")
    ap.add_argument("--dump", metavar="OUT.jsonl", default=None,
                    help="把**逐题逐策略**的分数落 JSONL —— 事后想算任意两策略的配对区间"
                         "（比如 4 vs 3 这个机制对照）就不必重跑一遍")
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    qs = []
    for t in types:
        picked = [r for r in rows if r.get("question_type", "").startswith(t)
                  and [e for e in (r.get("evidence_list") or []) if e.get("fact")]]
        for r in picked[: args.n]:
            qs.append((t, r["query"], [e["fact"] for e in r["evidence_list"] if e.get("fact")]))

    if args.reranker == "qwen":
        from rag.reranker_qwen import QwenReranker
        retr = HybridRetriever(load_corpus(), reranker=QwenReranker())
    else:
        retr = HybridRetriever(load_corpus())
    llm = build_model(args.model)

    # 阶段一：**并发**生成改写/子问句。纯网关等待，串行做等于把整轮实验的时间乘以 5。
    # ⚠️ 这里必须重试：一次 502 就丢一题，而丢题是**静默**的——实验12 的教训（失败必须可见、
    #    不能悄悄降级）在这里换了个马甲又出现了一次。实测并发 6 会撞上网关的并发上限（429），
    #    默认降到 3；改写结果按问句哈希落盘缓存，重跑时不必再向网关要一遍。
    cache_p = pathlib.Path(args.cache)
    cache: dict[str, list] = (json.loads(cache_p.read_text(encoding="utf-8"))
                              if cache_p.exists() else {})

    def _ask(prompt: str) -> str:
        last = None
        for wait in (0, 5, 20, 60):
            if wait:
                time.sleep(wait)
            try:
                return llm.invoke(prompt).content
            except Exception as e:      # noqa: BLE001 —— 网关 5xx/429/超时
                last = e
        raise RuntimeError(f"网关连续 4 次失败：{type(last).__name__}: {str(last)[:80]}")

    def _rewrite(item):
        _, q, _ = item
        key = hashlib.md5(f"{args.sub}|{q}".encode()).hexdigest()
        if key in cache:
            return cache[key]
        try:
            out = [_lines(_ask(_MULTI.format(n=args.sub, q=q)), args.sub) or [q],
                   _lines(_ask(_DECOMP.format(n=args.sub, q=q)), args.sub) or [q]]
        except Exception as e:      # noqa: BLE001 —— 重试尽了；该题整体跳过（会计入 attrition）
            print(f"  ⚠️ 改写失败，跳过该题：{str(e)[:90]}", flush=True)
            return None
        cache[key] = out
        return out

    print(f"阶段一：并发 {args.concurrency} 生成 {len(qs)} 题的改写/子问句"
          f"（缓存命中 {sum(1 for t, q, _ in qs if hashlib.md5(f'{args.sub}|{q}'.encode()).hexdigest() in cache)} 题）…",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rewrites = list(pool.map(_rewrite, qs))
    cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # 掉队率必须**按题型报出来**：如果某一类掉得特别多，各类之间就不再可比了。
    lost: dict[str, list[int]] = {}
    for (t, _, _), rw in zip(qs, rewrites):
        s = lost.setdefault(t, [0, 0])
        s[1] += 1
        s[0] += rw is None
    tot = sum(v[0] for v in lost.values())
    if tot:
        print("⚠️ 改写阶段掉队（网关重试耗尽）：" + "｜".join(
            f"{t} {v[0]}/{v[1]}" for t, v in lost.items())
            + "  ← 若各题型掉队率差别大，跨题型比较不可信，应重跑")
    print("阶段二：检索 + 重排（GPU 串行）…", flush=True)

    agg: dict[str, dict[str, list]] = {}
    qtypes: list[str] = []          # 与 agg 里每条 list 的下标一一对应 → 支持分题型切片 + 配对
    dump: list[dict] = []

    for i, ((t, q, facts), rw) in enumerate(zip(qs, rewrites), 1):
        if rw is None:              # 改写失败 → 整题跳过，各策略下标仍对齐
            continue
        multi, decomp = rw

        fused = retr._fuse(q)
        strategies = {
            "0 不重排": fused,                                # 实验14 拿来当"天花板"的那条基线
            "1 baseline": _rerank(retr, q, fused),
            "2 multiquery": _rerank(retr, q, _rrf([retr._fuse(s) for s in multi])),
            "3 decompose": _rerank(retr, q, _rrf([retr._fuse(s) for s in decomp])),
        }
        # 4: 每个子问句各自重排，按名额轮流取（不让原问句的元层面措辞把证据压回去）
        per = [_rerank(retr, s, retr._fuse(s)) for s in decomp]
        inter, seen = [], set()
        for rank in range(max((len(p) for p in per), default=0)):
            for p in per:
                if rank < len(p) and p[rank].id not in seen:
                    seen.add(p[rank].id)
                    inter.append(p[rank])
        strategies["4 decompose-each"] = inter

        row = {"type": t, "question": q}
        for name, docs in strategies.items():
            slot = agg.setdefault(name, {})
            for k, v in _score(docs, facts).items():
                slot.setdefault(k, []).append(v)
                row[f"{name}|{k}"] = v
        qtypes.append(t)
        dump.append(row)
        if i == 1:
            print(f"[样例] 原问句：{q[:100]}\n  改写→ {multi[0][:90]}\n  分解→ {decomp[0][:90]}\n")
        if i % 10 == 0:
            print(f"  {i}/{len(qs)}", flush=True)

    if args.dump:
        pathlib.Path(args.dump).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in dump), encoding="utf-8")
        print(f"逐题分数已写入 {args.dump}（{len(dump)} 行）")

    cols = [f"@{k}" for k in _KS] + ["pool"]
    kk = f"@{_KS[1]}"
    print(f"\n== fact 级召回 · 按**查询策略** · n={len(qtypes)} · 子问句 {args.sub} 个"
          f" · chunk={CHUNK_SIZE} · reranker={args.reranker} ==")
    for t in ([*types, "全部"] if len(types) > 1 else types):
        idx = [j for j, x in enumerate(qtypes) if t == "全部" or x == t]
        if not idx:
            continue
        print(f"\n--- {t}（n={len(idx)}）---")
        print(f"{'策略':18s}" + "".join(f"{c:>9}" for c in cols)
              + f"{'Δ' + kk + ' vs baseline [95%CI]':>36}")
        base = [agg["1 baseline"][kk][j] for j in idx]
        for name in _STRATEGIES:
            d = agg.get(name)
            if not d:
                continue
            cells = "".join(f"{st.mean(d[c][j] for j in idx):>9.3f}" for c in cols)
            if name == "1 baseline":
                print(f"{name:18s}{cells}{'（基准）':>32}")
                continue
            m, lo, hi = _paired_ci(base, [d[kk][j] for j in idx])
            sig = "✅" if lo > 0 else ("⛔" if hi < 0 else "  ")
            print(f"{name:18s}{cells}{m:>+18.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        print("  ✅/⛔ = 配对区间不跨 0；空白 = **测不出差异**，不许拿点估计下结论（实验20 的教训）")


if __name__ == "__main__":
    main()
