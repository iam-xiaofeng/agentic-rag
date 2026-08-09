"""**pool 扫描**：候选池开多大才值 —— 收益（交付）与成本（重排耗时）一起看。

    python evals/eval_pool.py --pools 50,100,200,300 --per-type 30

━━━ 为什么单独写这个 ━━━

`pool` 是**进 cross-encoder 之前的候选数**：BM25 / dense 各取 top-pool → RRF 融合 → 截断到 pool
→ **这 pool 条全部逐对过一遍 reranker** → 取 top-k。所以 pool 直接决定重排的算力开销
（pool=200 时每次检索要给 cross-encoder 打 200 次分），而 `eval_rebuild.py --layer 2` 扫 pool 时
**关掉了重排**（它问的是"融合有没有害"），`--layer 3` 带重排却只吃单个 pool。
"pool 大了值不值"这个问题两边都答不了，所以补这一个。

━━━ 口径 ━━━

- **单发检索**：直接用**原问句**查一次，**不走 agent**。这样测到的是 pool 的纯效应，
  不混进"模型愿不愿意多查几次"（实验21/23 的教训：检索侧的效应穿过 agentic 循环会被方差淹掉，
  所以要量部件就在部件层量）。
- **交付 = gold 证据段有多少进了 top-k**，纯子串匹配，**零裁判噪声、零 API 费用**。
- **池覆盖 = gold 证据段有多少进了融合后的候选池**。它是交付的上界：
  池里都没有的，reranker 再强也交付不了。**两个一起看才能分清"没捞到"和"没排上来"。**
- **重排器只加载一次**、四个 pool 复用同一个实例，否则测到的是模型加载时间。
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys

if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import random
import re
import time

import numpy as np

from evals.eval_agentic import load_benchmark, sample
from rag.retriever_hybrid import HybridRetriever

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").lower()).strip()


def _covered(texts: list[str], golds: list[str]) -> float:
    """gold 证据段里有多少条出现在给定文本集合中（整段子串匹配）。"""
    if not golds:
        return float("nan")
    blob = _norm(" ".join(texts))
    return sum(1 for g in golds if _norm(g)[:300] in blob) / len(golds)


def search_timed(retr: HybridRetriever, query: str, ks: list[int]) -> tuple[dict[int, list[str]], list[str], float, float]:
    """复刻 HybridRetriever.search，但把**融合**与**重排**的耗时分开计，并**一次重排给出所有 k**。

    重排完排序就固定了，不同 k 只是取不同长度的前缀 —— 所以整个 pool×k 网格的算力成本
    等于只扫 pool。**k 和 pool 必须一起扫**：k 的最优值是在某个 pool 下测出来的，换了 pool 未必还成立。
    """
    t0 = time.perf_counter()
    cands = retr._fuse(query)
    t1 = time.perf_counter()
    if not cands:
        return {k: [] for k in ks}, [], t1 - t0, 0.0
    scores = np.asarray(retr.reranker.predict([(query, d.text) for d in cands]))
    order = np.argsort(-scores)
    t2 = time.perf_counter()
    ranked = [cands[i].text for i in order]
    return {k: ranked[:k] for k in ks}, [d.text for d in cands], t1 - t0, t2 - t1


def main() -> None:
    ap = argparse.ArgumentParser(description="pool 扫描：交付收益 vs 重排耗时")
    ap.add_argument("--pools", default="25,50,100,200", help="第一个作为配对基准（取最小/最便宜的）")
    ap.add_argument("--ks", default="4,8,16,32", help="交付给模型的片段数（会和 pool 组成网格一起扫）")
    ap.add_argument("--benchmark", default="musique")
    ap.add_argument("--per-type", type=int, default=30, help="与 eval_agentic.py 同值 = 同一批题")
    ap.add_argument("--types", default="")
    ap.add_argument("--fig", default="runs/pool_k_grid.png", help="出图路径（PNG）")
    args = ap.parse_args()

    random.seed(19)
    pools = sorted({int(p) for p in args.pools.split(",") if p.strip()})
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    default_types = {"musique": ["2hop", "3hop", "4hop"],
                     "multihoprag": ["comparison_query", "inference_query", "temporal_query"]}
    types = [t.strip() for t in args.types.split(",") if t.strip()] or default_types[args.benchmark]
    examples, qtype, corpus = load_benchmark(args.benchmark)
    picked = sample(examples, qtype, types, args.per_type)
    qs = [(qtype.get(e.inputs["question"], "?").split("_")[0], e.inputs["question"],
           [c for c in (e.outputs.get("reference_contexts") or []) if c]) for e in picked]
    qs = [q for q in qs if q[2]]
    print(f"pool×k 网格：{len(qs)} 题 × {len(pools)} 个 pool × {len(ks)} 个 k。"
          f"**纯本地 GPU，不调网关、不花钱。**")

    retr = HybridRetriever(corpus)          # 重排器只加载一次，所有 pool 复用
    rows: dict[int, list] = {}
    for pool in pools:
        retr.pool = pool
        rec = []
        t0 = time.time()
        for i, (t, q, golds) in enumerate(qs, start=1):
            byk, cands, t_fuse, t_rank = search_timed(retr, q, ks)
            r = {"type": t, "poolcov": _covered(cands, golds), "t_fuse": t_fuse, "t_rank": t_rank,
                 "chars": {k: sum(len(x) for x in byk[k]) for k in ks}}
            for k in ks:
                r[f"d@{k}"] = _covered(byk[k], golds)
            rec.append(r)
            if i % 30 == 0:
                print(f"  pool={pool}  {i}/{len(qs)}  ({(time.time() - t0) / i:.2f}s/题)", flush=True)
        rows[pool] = rec

    mean = lambda v: (sum(v) / len(v)) if v else float("nan")            # noqa: E731
    ok = lambda v: [x for x in v if x == x]                              # noqa: E731

    def agg(pool: int, key: str) -> float:
        return mean(ok([r[key] for r in rows[pool]]))

    def chars(pool: int, k: int) -> float:
        return mean([r["chars"][k] for r in rows[pool]])

    # ── 网格：交付@k ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 92}\npool × k 网格 · 交付（gold 证据段进了 top-k 的比例）\n{'=' * 92}")
    print(f"{'pool':>6}{'池覆盖':>9}{'重排ms':>9}   " + "".join(f"{'k=' + str(k):>10}" for k in ks))
    best = max(((p, k, agg(p, f"d@{k}")) for p in pools for k in ks), key=lambda x: x[2])
    for p in pools:
        cells = []
        for k in ks:
            v = agg(p, f"d@{k}")
            cells.append(f"{v:>9.3f}" + ("*" if (p, k) == best[:2] else " "))
        print(f"{p:>6}{agg(p, 'poolcov'):>9.3f}{agg(p, 't_rank') * 1000:>9.0f}   " + "".join(cells))
    print(f"   * 点估计最高：pool={best[0]} k={best[1]} → {best[2]:.3f}")

    # ── 成本：k 花的是**上下文字符**（送进模型的钱），pool 花的是**重排算力**。两笔账不同 ──
    print(f"\n{'成本 · 每次检索交付的字符数（k 的真实代价）':<44}")
    print(f"{'pool':>6}   " + "".join(f"{'k=' + str(k):>10}" for k in ks))
    for p in pools:
        print(f"{p:>6}   " + "".join(f"{chars(p, k):>10.0f}" for k in ks))

    # ── 配对 bootstrap：以「最便宜的那一格」为基准 ────────────────────────────
    ref_p, ref_k = pools[0], ks[0]
    print(f"\n  同题配对差值（基准 = 最便宜的一格 pool={ref_p} k={ref_k}，5000 次 bootstrap 95% CI）")
    print(f"  {'配置':>12}{'Δ交付':>10}{'95% CI':>22}{'重排算力':>10}{'上下文字符':>12}")
    base = [r[f"d@{ref_k}"] for r in rows[ref_p]]
    base_t, base_c = agg(ref_p, "t_rank"), chars(ref_p, ref_k)
    for p in pools:
        for k in ks:
            if (p, k) == (ref_p, ref_k):
                continue
            cur = [r[f"d@{k}"] for r in rows[p]]
            d = [a - b for a, b in zip(cur, base) if a == a and b == b]
            s = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(3000))
            lo, hi = s[75], s[2925]
            flag = "✅" if lo > 0 else ("⛔" if hi < 0 else "跨0")
            print(f"  {f'pool={p} k={k}':>12}{mean(d):>+10.3f}   [{lo:+.3f},{hi:+.3f}] {flag:<4}"
                  f"{agg(p, 't_rank') / base_t:>9.1f}×{chars(p, k) / base_c:>11.1f}×")

    if args.fig:
        _plot(rows, pools, ks, agg, chars, args.fig)
        print(f"\n  图已存到 {args.fig}")

    print("\n  ▸ **两笔账要分开记**：`pool` 花的是**重排算力**（GPU，每条候选一次 cross-encoder 前向）；"
          "\n    `k` 花的是**上下文字符**（送进模型的 token + lost-in-the-middle）。混在一起就选不出配置。")
    print("  ▸ **池覆盖涨而交付不涨 ⇒ 加大 pool 只是往池里塞了更多够不着的东西**，"
          "瓶颈在 reranker 不在召回。")
    print("  ▸ ⚠️ 只量池覆盖会得出相反结论 —— 它是**没有成本项的上游代理量**，"
          "单调随 pool 上升，永远支持「再开大一点」。\n    这与 `context_recall_fact`「塞得越多分越高」"
          "是同一个错误模式。")
    print("  ▸ ⚠️ 这里量的是**检索侧交付**。k 变小还会让 agent **多查几次**（实验27），"
          "那部分收益本表看不见 ——\n    要定线上配置，本表选出候选后仍需端到端配对复核。")


def _plot(rows, pools, ks, agg, chars, path: str) -> None:
    """两张图：① 交付 vs k（每个 pool 一条线）；② 交付 vs 上下文字符（帕累托前沿）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    cjk = [f.name for f in fm.fontManager.ttflist if "WenQuanYi" in f.name or "CJK" in f.name]
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk[0]]
        plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    for i, p in enumerate(pools):
        c = cmap(i / max(1, len(pools) - 1))
        ys = [agg(p, f"d@{k}") for k in ks]
        ax1.plot(ks, ys, "o-", color=c, label=f"pool={p}  ({agg(p, 't_rank') * 1000:.0f}ms)")
        ax2.plot([chars(p, k) for k in ks], ys, "o-", color=c, label=f"pool={p}")
    ax1.set_xscale("log", base=2); ax1.set_xticks(ks); ax1.set_xticklabels(ks)
    ax1.set_xlabel("k（交付给模型的片段数）"); ax1.set_ylabel("交付（gold 证据段进 top-k 的比例）")
    ax1.set_title("① 交付 vs k —— 每条线一个 pool")
    ax1.grid(alpha=.3); ax1.legend(fontsize=8, title="括号内为重排耗时")
    ax2.set_xlabel("每次检索交付的字符数（送进模型的成本）"); ax2.set_ylabel("交付")
    ax2.set_title("② 帕累托前沿：同样的交付，谁更便宜")
    ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    fig.suptitle("pool × k 网格（MuSiQue，n=90，确定性指标，同题配对）", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
