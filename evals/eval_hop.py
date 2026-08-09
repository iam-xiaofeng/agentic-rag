"""**逐跳检索**：给定每一跳该问的问题，它那一段有没有被排进 top-k —— 用来选 `pool` / `k`。

    python evals/eval_hop.py --pools 25,50,100,200 --ks 4,8,16,32 --n 90

━━━ 为什么必须换掉「单发全量召回」这个靶子 ━━━

`eval_pool.py` 量的是：**用原问句查一次**，gold 的 N 段证据能捞回几段。
**对多跳题这是错的靶子** —— 多跳题的构造就是要让"一次拿全"不可能：
第 2 跳的段落必须先知道第 1 跳的答案才找得到。

后果不是"不准"，是**结构性偏向**：既然要一次凑齐 N 段，指标就必然奖励更大的 k。
实测那张网格给出"k=32 最好、比 k=8 高 +0.13"——**那不是发现，是这个指标的算术性质**。
`池覆盖` 更甚：它连排序都不看，单调随 pool 上升，永远支持"再开大一点"。
（这是本项目第 4 次栽在「没有成本项的上游代理量」上，前三次是 title 级 context_recall、
`context_recall_fact`、以及可达空间利用率的分母。）

━━━ 正确的靶子 ━━━

MuSiQue 每一跳都给了 `question` 和 `paragraph_support_idx`。把 `#N` 换成前跳的**真答案**，
就得到"这一跳理想情况下该发的 query"；它的 gold 就是**唯一一段**。于是问题变成：

    **给定这一跳该问的问题，它的那一段有没有进 top-k？**

这个口径的三个好处：
  · **与链长无关** —— 4hop 题拆成 4 次独立检索，不再"一次凑齐"；
  · **k 不再被结构性奖励** —— 要找的只有 1 段，k=8 与 k=32 是公平比的；
  · **能按跳位拆** —— 第 1 跳 vs 第 4 跳的召回差，才是"越往后越难检索"的直接证据。

⚠️ 它量的是**检索器的上限**（喂了理想 query），不是线上表现 —— 线上 agent 得自己
把那个 query 想出来。所以：**本表用来选 pool/k，不用来预测端到端分数。**

⚠️ `A >> relation` 这种模板式子问句按 `"A relation"` 处理（去掉 `>>`）——
对 BM25/dense 而言这是个合理的词袋 query，不必因为它不像人话就丢掉整跳。
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

from rag.corpus_musique import _raw, load_corpus
from rag.retriever_hybrid import HybridRetriever

_WS = re.compile(r"\s+")
_REF = re.compile(r"#(\d+)")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").lower()).strip()


def build_hops(n_per_type: int) -> list[dict]:
    """→ [{qid, type, hop(1-based), query, gold}]，每条是**一次理想的单跳检索**。"""
    by_type: dict[str, list] = {}
    for r in _raw():
        if not r.get("answerable"):
            continue
        by_type.setdefault(f"{len(r['question_decomposition'])}hop", []).append(r)
    out = []
    for t, rs in sorted(by_type.items()):
        step = max(1, len(rs) // n_per_type)
        for r in rs[::step][:n_per_type]:
            para = {p["idx"]: p for p in r["paragraphs"]}
            answers = [h.get("answer") or "" for h in r["question_decomposition"]]
            for i, h in enumerate(r["question_decomposition"], start=1):
                p = para.get(h.get("paragraph_support_idx"))
                if not p or not (p.get("paragraph_text") or "").strip():
                    continue
                # `#N` → 第 N 跳的**真答案**：这就是"理想情况下该发的 query"
                q = _REF.sub(lambda m: answers[int(m.group(1)) - 1] if int(m.group(1)) <= len(answers) else "",
                             h.get("question") or "")
                q = q.replace(">>", " ").strip()
                if len(q) < 3:
                    continue
                out.append({"qid": r["id"], "type": t, "hop": i, "query": q,
                            "gold": (p.get("paragraph_text") or "").strip()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="逐跳检索：pool × k 网格（确定性、纯本地）")
    ap.add_argument("--pools", default="25,50,100,200")
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--n", type=int, default=30, help="每种题型取几道题（每道会展开成多跳）")
    ap.add_argument("--fig", default="runs/hop_recall_grid.png")
    ap.add_argument("--ablate", action="store_true",
                    help="改为跑**整栈消融**：BM25 单路 → dense 单路 → 融合 → 重排，"
                         "每层的边际贡献用同一把尺量，外加融合权重扫描")
    args = ap.parse_args()

    random.seed(23)
    if args.ablate:
        return ablate(args)
    grid(args)


def ablate(args) -> None:
    """整栈消融：**同一批逐跳 query、同一把尺**，从下往上一层层加，看每层的边际贡献。

    为什么必须用同一把尺：项目里 chunk / 融合 / reranker 三层的历史结论分别来自
    `fact@k(前120字符)`、`池覆盖`、`fact@8` 三种口径，**互相不可比**，
    所以"哪一层最值得投"这个问题以前答不了。
    """
    from rag.retriever_bm25 import BM25Retriever
    from rag.retriever_dense import DenseRetriever

    hops = build_hops(args.n)
    k = int(args.ks.split(",")[0]) if "," in args.ks else int(args.ks)
    k = 8 if k not in (4, 8, 16, 32) else k
    corpus = load_corpus()
    print(f"整栈消融：{len({h['qid'] for h in hops})} 题 → **{len(hops)} 次单跳检索**，k={k}。"
          f"**纯本地 GPU，不调网关。**")
    # 三个部件各只建一次：两条召回腿 + 一个 cross-encoder。
    # ⚠️ 不这样做的话每个臂都会重新加载一遍重排模型，量到的一大半是模型加载时间。
    bm25, dense = BM25Retriever(corpus), DenseRetriever(corpus)
    from sentence_transformers import CrossEncoder
    from rag.retriever_hybrid import _RERANKER
    ce = CrossEncoder(_RERANKER)

    def mk(rerank=True, **kw):
        return HybridRetriever(corpus, bm25=bm25, dense=dense, pool=50,
                               reranker=(ce if rerank else None), **kw)

    arms = [
        ("① BM25 单路（不融合、不重排）", mk(rerank=False, w_bm25=1.0, w_dense=0.0)),
        ("② dense 单路（不融合、不重排）", mk(rerank=False, w_bm25=0.0, w_dense=1.0)),
        ("③ RRF 融合（不重排）", mk(rerank=False)),
        ("④ minmax 融合（不重排）", mk(rerank=False, fusion="minmax")),
        ("⑤ RRF + reranker ← **现行**", mk()),
    ]
    # 融合权重扫描（都带重排，因为线上就带）
    arms += [(f"   └ w_bm25={w:.2f} / w_dense={1 - w:.2f}", mk(w_bm25=w, w_dense=1 - w))
             for w in (0.0, 0.25, 0.5, 0.75, 1.0)]

    print(f"\n{'=' * 88}\n整栈消融 · 逐跳召回@k={k}（每跳只有 1 段 gold）\n{'=' * 88}")
    print(f"{'配置':<34}{'召回@' + str(k):>10}{'Δ vs 上一层':>13}{'ms/跳':>9}")
    prev, results = None, []
    for name, retr in arms:
        hit, t_tot = [], 0.0
        for h in hops:
            t0 = time.perf_counter()
            got = retr.search(h["query"], k=k)
            t_tot += time.perf_counter() - t0
            g = _norm(h["gold"])[:300]
            hit.append(1.0 if any(g in _norm(x.doc.text) for x in got) else 0.0)
        r = sum(hit) / len(hit)
        results.append((name, r, hit, t_tot / len(hops) * 1000))
        delta = "" if prev is None or name.startswith("   ") else f"{r - prev:+.3f}"
        print(f"{name:<34}{r:>10.3f}{delta:>13}{t_tot / len(hops) * 1000:>9.0f}")
        if not name.startswith("   "):
            prev = r

    # 关键对照的配对区间：④ vs ③（融合算法）、⑤ vs ③（重排到底值多少）
    def paired(a_idx: int, b_idx: int, label: str) -> None:
        _, _, ha, _ = results[a_idx]
        _, _, hb, _ = results[b_idx]
        d = [x - y for x, y in zip(ha, hb)]
        s = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(3000))
        lo, hi = s[75], s[2925]
        flag = "✅" if lo > 0 else ("⛔" if hi < 0 else "跨0")
        print(f"  {label:<44}{sum(d) / len(d):>+8.3f}   [{lo:+.3f},{hi:+.3f}] {flag}")

    print(f"\n  同跳配对差值（3000 次 bootstrap 95% CI）")
    paired(2, 0, "融合 − BM25 单路（融合值不值）")
    paired(2, 1, "融合 − dense 单路")
    paired(3, 2, "minmax − RRF（融合算法选哪个）")
    paired(4, 2, "★ 重排 − 不重排（reranker 到底提升多少）")

    print("\n  ▸ **每一层都用同一把尺**（逐跳召回），所以「Δ vs 上一层」才是可比的边际贡献。")
    print("  ▸ ⚠️ 权重扫描那几行若**完全一样**，不是两条腿等价，是**下游 reranker 把上游差异抹平了**"
          "\n    —— 这时候调融合权重是无效功（实验16 的归因订正）。")


def grid(args) -> None:
    pools = sorted({int(p) for p in args.pools.split(",") if p.strip()})
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    hops = build_hops(args.n)
    print(f"逐跳检索：{len({h['qid'] for h in hops})} 道题展开成 **{len(hops)} 次单跳检索**"
          f" × {len(pools)} 个 pool。**纯本地 GPU，不调网关。**")

    retr = HybridRetriever(load_corpus())              # 重排器只加载一次
    rows: dict[int, list] = {}
    for pool in pools:
        retr.pool = pool
        rec, t0 = [], time.time()
        for i, h in enumerate(hops, start=1):
            t1 = time.perf_counter()
            cands = retr._fuse(h["query"])
            scores = np.asarray(retr.reranker.predict([(h["query"], d.text) for d in cands]))
            ranked = [cands[j].text for j in np.argsort(-scores)]
            dt = time.perf_counter() - t1
            g = _norm(h["gold"])[:300]
            # 名次：gold 段排第几（0-based）；找不到记 10**6
            rank = next((j for j, txt in enumerate(ranked) if g in _norm(txt)), 10 ** 6)
            rec.append({**{kk: h[kk] for kk in ("type", "hop")}, "rank": rank, "t": dt,
                        "inpool": rank < len(ranked)})
            if i % 100 == 0:
                print(f"  pool={pool}  {i}/{len(hops)}  ({(time.time() - t0) / i:.2f}s/跳)", flush=True)
        rows[pool] = rec

    mean = lambda v: (sum(v) / len(v)) if v else float("nan")                       # noqa: E731
    rec_at = lambda rs, k: mean([1.0 if r["rank"] < k else 0.0 for r in rs])        # noqa: E731

    print(f"\n{'=' * 92}\n逐跳召回 · gold 段进 top-k 的比例（每跳只有 1 段，k 不再被结构性奖励）\n{'=' * 92}")
    print(f"{'pool':>6}{'池内':>8}{'重排ms':>9}   " + "".join(f"{'k=' + str(k):>10}" for k in ks))
    best = max(((p, k, rec_at(rows[p], k)) for p in pools for k in ks), key=lambda x: x[2])
    for p in pools:
        rs = rows[p]
        cells = "".join(f"{rec_at(rs, k):>9.3f}" + ("*" if (p, k) == best[:2] else " ") for k in ks)
        print(f"{p:>6}{mean([1.0 if r['inpool'] else 0.0 for r in rs]):>8.3f}"
              f"{mean([r['t'] for r in rs]) * 1000:>9.0f}   " + cells)
    print(f"   * 点估计最高：pool={best[0]} k={best[1]} → {best[2]:.3f}")

    print(f"\n按**跳位**拆（pool={pools[len(pools) // 2]}）—— 越往后越难检索？")
    mid = pools[len(pools) // 2]
    hop_ids = sorted({r["hop"] for r in rows[mid]})
    print(f"{'跳位':>6}{'n':>6}   " + "".join(f"{'k=' + str(k):>10}" for k in ks))
    for hp in hop_ids:
        rs = [r for r in rows[mid] if r["hop"] == hp]
        print(f"{'第' + str(hp) + '跳':>6}{len(rs):>6}   " + "".join(f"{rec_at(rs, k):>10.3f}" for k in ks))

    ref_p, ref_k = pools[0], ks[0]
    print(f"\n  同跳配对差值（基准 = 最便宜的一格 pool={ref_p} k={ref_k}，3000 次 bootstrap 95% CI）")
    print(f"  {'配置':>14}{'Δ召回':>10}{'95% CI':>22}{'重排算力':>10}")
    base = [1.0 if r["rank"] < ref_k else 0.0 for r in rows[ref_p]]
    bt = mean([r["t"] for r in rows[ref_p]])
    for p in pools:
        for k in ks:
            if (p, k) == (ref_p, ref_k):
                continue
            cur = [1.0 if r["rank"] < k else 0.0 for r in rows[p]]
            d = [a - b for a, b in zip(cur, base)]
            s = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(3000))
            lo, hi = s[75], s[2925]
            flag = "✅" if lo > 0 else ("⛔" if hi < 0 else "跨0")
            print(f"  {f'pool={p} k={k}':>14}{mean(d):>+10.3f}   [{lo:+.3f},{hi:+.3f}] {flag:<4}"
                  f"{mean([r['t'] for r in rows[p]]) / bt:>9.1f}×")

    if args.fig:
        _plot(rows, pools, ks, rec_at, mean, args.fig)
        print(f"\n  图已存到 {args.fig}")

    print("\n  ▸ **这张表才是选 pool/k 的依据**：每跳只有 1 段 gold，k 大不再自动占便宜。")
    print("  ▸ 按跳位拆的那张表回答「越往后越难检索吗」—— 若第 4 跳明显低于第 1 跳，"
          "\n    那 4hop 的失败就有一部分是**检索**问题；若各跳持平，失败就在**链条本身**（推不出下一跳的 query）。")
    print("  ▸ ⚠️ 这里喂的是**理想 query**（`#N` 换成真答案），量的是检索器上限。"
          "\n    线上 agent 得自己想出那个 query —— 本表选配置，不预测端到端分数。")


def _plot(rows, pools, ks, rec_at, mean, path: str) -> None:
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
        ax1.plot(ks, [rec_at(rows[p], k) for k in ks], "o-", color=c,
                 label=f"pool={p}  ({mean([r['t'] for r in rows[p]]) * 1000:.0f}ms)")
    ax1.set_xscale("log", base=2); ax1.set_xticks(ks); ax1.set_xticklabels(ks)
    ax1.set_xlabel("k"); ax1.set_ylabel("逐跳召回（gold 段进 top-k）")
    ax1.set_title("① 逐跳召回 vs k"); ax1.grid(alpha=.3); ax1.legend(fontsize=8, title="括号内为重排耗时")

    mid = pools[len(pools) // 2]
    hop_ids = sorted({r["hop"] for r in rows[mid]})
    for i, k in enumerate(ks):
        ax2.plot(hop_ids, [rec_at([r for r in rows[mid] if r["hop"] == h], k) for h in hop_ids],
                 "o-", color=cmap(i / max(1, len(ks) - 1)), label=f"k={k}")
    ax2.set_xticks(hop_ids); ax2.set_xlabel("第几跳"); ax2.set_ylabel("逐跳召回")
    ax2.set_title(f"② 越往后越难检索吗（pool={mid}）"); ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    fig.suptitle("逐跳检索（MuSiQue，理想 query，确定性指标）", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
