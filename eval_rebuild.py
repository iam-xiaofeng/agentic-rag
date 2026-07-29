"""逐层重建检索栈的**验收脚本**：一次只动一层，每层用「对它最有利的输入」自检。

起因（实验17-18）：全链路排查发现 chunk 配置从项目第一天起就是错的，而**所有端到端指标都没报警**
——因为端到端只会告诉你「就这样了」，不会告诉你是哪个部件半残。所以重建时每层都必须有两个验收：

  ① **单点自检**：给这一层最有利的输入，看它能否接近满分。
     embedding 层的最有利输入 = **证据句的原文**（query 就是答案本身，没有任何语义鸿沟）。
     连这个都做不到 top1，就与「多跳难」「问句刁钻」无关，是部件本身坏了。
  ② **按题型均衡**：绝不看全类型聚合均值——实验15 的教训是一个子群的系统性伤害
     会被另一个子群的收益完全掩盖（reranker 在 temporal 上净减益 −0.100，平均值却是 +0.088）。

层次与顺序（数据流方向，上游先修；BM25 已验证健康，不动，只作对照基线）：

    --layer 0   前置自检 —— 本脚本的精确余弦探针 与 生产用的 Chroma(HNSW) 是否排出同一个结果。
    --layer 1   embedding —— chunk 配置。上游中的上游，选错下游补不回来。
    --layer 2   融合 —— pool 随 chunk 缩小而放大；判据是「融合不该比更好的那条单路腿差」。
    --layer 3   重排 —— 判据是「重排不该比不重排交付得少」；并看证据 chunk 被推前还是推后。

用法：
    python eval_rebuild.py --layer 1 --n 30
    python eval_rebuild.py --layer 1 --cfgs 1200x150,600x150,600x200,500x200
    python eval_rebuild.py --layer 2 --cfgs 1200x150,600x150,500x200 --pools 100,200,300
    python eval_rebuild.py --layer 3 --cfgs 1200x150,600x150 --pool 200 --reranker bge

实现注记：**故意不走 Chroma**，直接精确余弦。每建一个 Chroma 集合要占 ~1.3G 磁盘，而 `.npy` 编码缓存
与 `DenseRetriever` 用同一套指纹，所以选定配置后真正建库时不会重算编码。
⚠️ 两者**并不完全等价**（`--layer 0` 实测）：**top1 一致率 100%**，但 top100 的集合重合只有 96%、
完整名次相同仅 17~22%——HNSW 的近似误差都落在 k 截断的边界上。所以自查类指标（看 top1）可以放心用探针；
凡是要**推翻既有结论**的对比，用 `--layer 3 --chroma` 拿生产真身复核一遍。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics as st
import time

import numpy as np

import corpus_multihop as cm
from retriever import Doc, Hit
from retriever_bm25 import BM25Retriever
from retriever_dense import _MODEL, _QUERY_PROMPT
from retriever_hybrid import HybridRetriever

DATA = pathlib.Path(__file__).resolve().parent / "data"
_CACHE = pathlib.Path(__file__).resolve().parent / ".cache"
_PROBE = 200            # 排名探测深度；超出（含证据句根本没留下来）一律记 _MISS
_MISS = 10**6
_KEY = 120             # 与下游 fact@k 口径一致：证据句前 120 字符作为匹配键
_MAIN_K = 16           # 「截断税」这张表固定在这个交付深度上比较（chunk=600 的等信息量档位）


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _fingerprint(docs: list[Doc], model: str = _MODEL) -> str:
    """与 DenseRetriever._fingerprint 完全一致 —— 好让 .npy 编码缓存互相复用。"""
    h = hashlib.md5()
    h.update(model.encode())
    h.update(str(len(docs)).encode())
    if docs:
        h.update(docs[0].id.encode())
        h.update(docs[-1].id.encode())
    return h.hexdigest()[:12]


class _DenseProbe:
    """只做「编码 + 精确余弦排名」的轻量 dense 探针（不建 Chroma，见模块注释）。"""

    def __init__(self, docs: list[Doc]) -> None:
        from sentence_transformers import SentenceTransformer

        self.docs = docs
        self.model = SentenceTransformer(_MODEL)
        _CACHE.mkdir(exist_ok=True)
        fp = _CACHE / f"dense_{_fingerprint(docs)}.npy"
        if fp.exists():
            self.emb = np.load(fp)
            print(f"    编码缓存命中 {fp.name}", flush=True)
        else:
            t0 = time.time()
            self.emb = self.model.encode(
                [d.text for d in docs], normalize_embeddings=True,
                batch_size=64, show_progress_bar=True).astype("float32")
            np.save(fp, self.emb)
            print(f"    编码 {len(docs)} 片段耗时 {time.time() - t0:.0f}s → {fp.name}", flush=True)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """满足 `Retriever` 协议 —— 好让它能被注入进真实的 `HybridRetriever` 当 dense 腿。"""
        q = self.model.encode([_QUERY_PROMPT + query], normalize_embeddings=True)[0].astype("float32")
        sims = self.emb @ q
        k = min(k, len(self.docs))
        top = np.argpartition(-sims, k - 1)[:k] if k < len(self.docs) else np.arange(len(self.docs))
        order = top[np.argsort(-sims[top])]
        return [Hit(doc=self.docs[int(i)], score=float(sims[int(i)])) for i in order]

    def rank_of(self, queries: list[str], owners: list[set[int]]) -> list[int]:
        """一批 query 各自的「首个 owner 片段」名次（0 起）；owner 为空或落在 _PROBE 外记 _MISS。"""
        qe = self.model.encode([_QUERY_PROMPT + q for q in queries],
                               normalize_embeddings=True, batch_size=64).astype("float32")
        sims = qe @ self.emb.T                                   # 归一化向量 → 点积即余弦
        top = np.argpartition(-sims, _PROBE, axis=1)[:, :_PROBE]  # 先粗筛前 _PROBE 个再排序
        out = []
        for row, cand, own in zip(sims, top, owners):
            if not own:
                out.append(_MISS)
                continue
            order = cand[np.argsort(-row[cand])]
            hit = [i for i, d in enumerate(order) if int(d) in own]
            out.append(hit[0] if hit else _MISS)
        return out


def _free_gpu() -> None:
    """连跑多个配置时，bge-large 会一个个堆在显存里（16G 卡上 5 个就 OOM）——显式回收。"""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _fmt(ranks: list[int]) -> str:
    """一组名次 → 中位名次 / top1 / top10（top1、top10 的分母是**全部**证据句，不剔除未留存的）。"""
    if not ranks:
        return f"{'—':>9}{'—':>7}{'—':>7}"
    hit = [r for r in ranks if r < _MISS]
    med = f"{st.median(hit):.0f}" if hit else "—"
    return (f"{med:>9}{sum(1 for r in ranks if r == 0) / len(ranks):>7.0%}"
            f"{sum(1 for r in ranks if r < 10) / len(ranks):>7.0%}")


def _load_questions(types: list[str], n: int) -> list[tuple[str, str, list[str]]]:
    """按题型均衡取样（每型前 n 条有 gold fact 的题）→ [(type, query, [fact...])]"""
    rows = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    out = []
    for t in types:
        picked = [r for r in rows if r.get("question_type", "").startswith(t)
                  and [e for e in (r.get("evidence_list") or []) if e.get("fact")]]
        for r in picked[:n]:
            out.append((t, r["query"], [e["fact"] for e in r["evidence_list"] if e.get("fact")]))
    return out


def _fact_recall(docs: list[Doc], facts: list[str], k: int, full: bool = False) -> float:
    """交付的前 k 个片段里，覆盖了多少条 gold 证据。

    `full=False` 是**历史口径**（只查证据句前 `_KEY`=120 字符）——它与实验1-18 可比，
    但有个致命盲区：**块被切小以后证据句被截断，这个指标完全看不见**（实测 chunk=400 时
    全句留存只剩 89~95%，而"前120留存"仍是 100%）。于是越小的块越占便宜，比较是不公平的。
    `full=True` 要求**整句**都在交付文本里，两者之差就是「截断税」。
    """
    blob = _norm(" ".join(d.text for d in docs[:k]))
    keys = [_norm(f) if full else _norm(f)[:_KEY] for f in facts]
    return sum(1 for key in keys if key in blob) / len(facts)


def _layer2(cfgs: list[tuple[int, int]], pools: list[int], types: list[str], n: int,
            ks: tuple[int, ...] = (4, 8, 16, 32)) -> None:
    """第 2 层 · 融合。

    自检判据（这一层唯一该保证的性质）：**融合后的 fact@k 不该低于更好的那条单路腿**。
    融合的全部意义是"把两种召回的盲区互补起来"，如果它比单独用 BM25 还差，那它在做负功
    ——实验16 实测到的正是这种情况（三类里持平或低于纯 BM25，temporal 上连池覆盖都是负的：
    弱腿把强腿的好候选挤出了 pool）。所以本层先问"融合有没有害"，再问"pool 该多大"。

    pool 随 chunk 缩小而放大：pool×chunk ≈ 恒定，池子里的**信息量**才可比（实验18 的教训）。
    同时用**全句**口径复核（见 `_fact_recall`），免得小块靠"截断税看不见"白赚一截。
    """
    qs = _load_questions(types, n)
    res: dict = {}
    for size, ovl in cfgs:
        docs = cm.load_corpus(chunk_size=size, chunk_overlap=ovl)
        print(f"\n配置 chunk={size} overlap={ovl}｜{len(docs)} 片段｜{len(qs)} 题（每型 {n}）")
        print(f"  等信息量：chunk={size} 时 k={round(1200 * 8 / size)} 才与 基线(1200,k=8) 交付同样多字符")
        probe, bm25 = _DenseProbe(docs), BM25Retriever(docs)
        for pool in pools:
            retr = HybridRetriever(docs, pool=pool, reranker=None, bm25=bm25, dense=probe)
            print(f"  pool={pool} 检索中…", flush=True)
            for t, q, facts in qs:
                legs = {"BM25": [h.doc for h in bm25.search(q, k=pool)],
                        "dense": [h.doc for h in probe.search(q, k=pool)],
                        "融合": retr._fuse(q)}
                for name, lst in legs.items():
                    slot = res.setdefault((size, ovl, pool, t, name),
                                          {k: [] for k in (*ks, "pool")} | {"full": [], "key": []})
                    for k in ks:
                        slot[k].append(_fact_recall(lst, facts, k))
                    slot["pool"].append(_fact_recall(lst, facts, len(lst)))
                    slot["full"].append(_fact_recall(lst, facts, _MAIN_K, full=True))
                    slot["key"].append(_fact_recall(lst, facts, _MAIN_K))
        del probe, bm25
        _free_gpu()

    kh = "".join(f"{'@' + str(k):>8}" for k in ks)
    for size, ovl in cfgs:
        for pool in pools:
            print(f"\n{'=' * 84}\nchunk={size}/{ovl} · pool={pool} · fact 级召回（**未重排**）\n{'=' * 84}")
            print(f"{'type':12s}{'策略':>8}{kh}{'覆盖@pool':>11}")
            for t in types:
                for name in ("BM25", "dense", "融合"):
                    d = res[(size, ovl, pool, t, name)]
                    print(f"{t:12s}{name:>8}" + "".join(f"{st.mean(d[k]):>8.3f}" for k in ks)
                          + f"{st.mean(d['pool']):>11.3f}")
                cells = []
                for k in (*ks, "pool"):
                    best = max(st.mean(res[(size, ovl, pool, t, nm)][k]) for nm in ("BM25", "dense"))
                    delta = st.mean(res[(size, ovl, pool, t, "融合")][k]) - best
                    cells.append(f"{delta:>+6.3f}{'✅' if delta >= -0.001 else '⛔'}")
                print(f"{'':12s}{'自检Δ':>8}" + "".join(f"{c:>8}" for c in cells[:len(ks)])
                      + f"{cells[-1]:>11}   ← 融合 − 更好单路；⛔ = 融合在做负功")

    print(f"\n{'=' * 84}\n截断税：同一批交付文本(@{_MAIN_K})，'整句都在' vs '前120字符在'\n{'=' * 84}")
    print(f"{'chunk':>6}{'ovl':>5}{'pool':>6}  {'type':12s}{'全句':>8}{'前120':>8}{'截断税':>9}")
    for size, ovl in cfgs:
        for pool in pools:
            for t in types:
                d = res[(size, ovl, pool, t, "融合")]
                f_, k_ = st.mean(d["full"]), st.mean(d["key"])
                print(f"{size:>6}{ovl:>5}{pool:>6}  {t:12s}{f_:>8.3f}{k_:>8.3f}{k_ - f_:>+9.3f}")
            print()


def _layer0(cfgs: list[tuple[int, int]], types: list[str], n: int, k: int = 100) -> None:
    """第 0 层 · 前置自检：**本脚本的精确余弦探针 与 生产用的 Chroma(HNSW) 排出来的是不是同一个东西**。

    本脚本为省磁盘用精确余弦替代了 Chroma，但**只要拿它的结论去推翻旧实验，这个替换本身就成了
    一个未受控的变量**。实验17 在 6711 片段上验过 100% 一致，可 chunk=600 的片段数翻了一倍多，
    HNSW 的近似误差会不会跟着涨？这里直接量：同一批 query，两者 top-k 的集合重合度与首位一致率。
    （顺带把选定配置的 Chroma 集合建出来——生产要用。）
    """
    from retriever_dense import DenseRetriever

    qs = _load_questions(types, n)
    print(f"{'chunk':>6}{'ovl':>5}{'片段数':>8}{'query数':>8}{'top1一致':>10}{'top' + str(k) + '重合':>10}{'名次相同':>10}")
    for size, ovl in cfgs:
        docs = cm.load_corpus(chunk_size=size, chunk_overlap=ovl)
        probe = _DenseProbe(docs)
        chroma = DenseRetriever(docs)                    # 若集合不存在会在此建好（生产要用）
        same1 = overlap = exact = 0
        for _, q, _ in qs:
            a = [h.doc.id for h in chroma.search(q, k=k)]
            b = [h.doc.id for h in probe.search(q, k=k)]
            same1 += a[:1] == b[:1]
            overlap += len(set(a) & set(b)) / max(len(b), 1)
            exact += a == b
        m = len(qs)
        print(f"{size:>6}{ovl:>5}{len(docs):>8}{m:>8}{same1 / m:>10.1%}{overlap / m:>10.1%}{exact / m:>10.1%}")
        del probe, chroma
        _free_gpu()


def _paired_ci(fuse: list[float], rr: list[float], boots: int = 5000) -> tuple[float, float, float]:
    """重排 − 不重排 的**配对**均值及 95% bootstrap 置信区间。

    为什么必须配对、必须带区间：实验15 用 n=15 的点估计 −0.100 得出"bge 在 temporal 上是净减益"，
    但同一个量在 n=30 上换了符号、还随 k 变号。**同一批题上两种做法的差值是配对数据**，
    配对能消掉"题目本身难易"这个最大的方差来源；带上区间才能看出这个数到底稳不稳。
    """
    d = np.asarray(rr) - np.asarray(fuse)
    rng = np.random.default_rng(0)                       # 固定种子 → 结论可复现
    idx = rng.integers(0, len(d), size=(boots, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _layer3(cfgs: list[tuple[int, int]], pool: int, types: list[str], n: int,
            reranker: str, chroma: bool = False, ks: tuple[int, ...] = (4, 8, 16, 32)) -> None:
    """第 3 层 · 重排。

    自检判据（与第 2 层对称）：**重排后的 fact@k 不该低于不重排**。重排的全部意义是"把相关的
    排到前面"，若它交付得比原始融合顺序还少，那它就是在做负功——实验15 实测 bge 在 temporal 上
    净减益 −0.100，51% 的 gold 证据 chunk 被**往后推**，中位名次从第 6 推到第 9，正好越过 k=8。

    但那是在**坏掉的 embedding**（chunk=1200，dense 自查 top1 仅 56%）上量的。本层要回答的是：
    上游修好之后，这个负增益是**消失了**（说明它是坏输入的下游症状），还是**还在**
    （说明 reranker 本身与多跳目标不匹配，得换成查询分解）。两种结论指向完全不同的下一步。
    """
    qs = _load_questions(types, n)
    for size, ovl in cfgs:
        docs = cm.load_corpus(chunk_size=size, chunk_overlap=ovl)
        if chroma:                                   # 用生产真身，排除"探针 ≠ Chroma"这个变量
            from retriever_dense import DenseRetriever
            probe = DenseRetriever(docs)
        else:
            probe = _DenseProbe(docs)
        bm25 = BM25Retriever(docs)
        if reranker == "qwen":
            from reranker_qwen import QwenReranker
            retr = HybridRetriever(docs, pool=pool, reranker=QwenReranker(), bm25=bm25, dense=probe)
        else:
            retr = HybridRetriever(docs, pool=pool, bm25=bm25, dense=probe)
        print(f"\nchunk={size}/{ovl}｜{len(docs)} 片段｜pool={pool}｜reranker={reranker}"
              f"｜dense={'Chroma(生产真身)' if chroma else '精确余弦探针'}", flush=True)

        agg: dict = {}
        for i, (t, q, facts) in enumerate(qs, 1):
            cands = retr._fuse(q)
            sc = np.asarray(retr.reranker.predict([(q, d.text) for d in cands]))
            ranked = [cands[j] for j in np.argsort(-sc)]
            slot = agg.setdefault(t, {"fuse": {k: [] for k in ks}, "rr": {k: [] for k in ks},
                                      "fpos": [], "rpos": [], "back": 0, "fwd": 0, "n_ev": 0})
            for k in ks:
                slot["fuse"][k].append(_fact_recall(cands, facts, k))
                slot["rr"][k].append(_fact_recall(ranked, facts, k))
            fpos = {d.id: j for j, d in enumerate(cands)}
            rpos = {d.id: j for j, d in enumerate(ranked)}
            for d in cands:                      # 只看**真含 gold 证据**的 chunk 被推到哪去了
                txt = _norm(d.text)
                if any(_norm(f)[:_KEY] in txt for f in facts):
                    a, b = fpos[d.id], rpos[d.id]
                    slot["fpos"].append(a)
                    slot["rpos"].append(b)
                    slot["n_ev"] += 1
                    if b > a:
                        slot["back"] += 1
                    elif b < a:
                        slot["fwd"] += 1
            if i % 20 == 0:
                print(f"    {i}/{len(qs)}", flush=True)

        print(f"\n{'=' * 84}\nchunk={size}/{ovl} pool={pool} · 融合(不重排) vs {reranker} 重排\n{'=' * 84}")
        print(f"{'type':12s}" + "".join(f"{'融合@' + str(k):>10}{'重排@' + str(k):>10}{'Δ':>8}" for k in ks))
        for t in types:
            d = agg[t]
            cells = []
            for k in ks:
                f_, r_ = st.mean(d["fuse"][k]), st.mean(d["rr"][k])
                cells.append(f"{f_:>10.3f}{r_:>10.3f}{r_ - f_:>+7.3f}{'✅' if r_ >= f_ else '⛔'}")
            print(f"{t:12s}" + "".join(cells))
        print("  ⛔ = 重排交付得比「什么都不做」还少，即 reranker 在做负功")

        print(f"\n配对 bootstrap · Δ(重排−不重排) 的 95% 置信区间（跨过 0 = 与「没有效果」无法区分）")
        print(f"{'type':12s}" + "".join(f"{'@' + str(k) + ' Δ [95%CI]':>26}" for k in ks))
        for t in types:
            d, cells = agg[t], []
            for k in ks:
                m, lo, hi = _paired_ci(d["fuse"][k], d["rr"][k])
                sig = "  " if lo <= 0 <= hi else ("✅" if lo > 0 else "⛔")
                cells.append(f"{m:>+8.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
            print(f"{t:12s}" + "".join(f"{c:>26}" for c in cells))
        print("  ✅/⛔ = 区间不跨 0（效应可信）；空白 = 区间跨 0，这个数**不足以下任何结论**")

        print(f"\n{'type':12s}{'证据chunk数':>11}{'融合中位名次':>13}{'重排中位名次':>13}{'被推后':>9}{'被推前':>9}")
        for t in types:
            d = agg[t]
            if not d["n_ev"]:
                continue
            print(f"{t:12s}{d['n_ev']:>11}{st.median(d['fpos']):>13.1f}{st.median(d['rpos']):>13.1f}"
                  f"{d['back'] / d['n_ev']:>8.0%}{d['fwd'] / d['n_ev']:>9.0%}")
        del retr, probe, bm25
        _free_gpu()


def _layer1(cfgs: list[tuple[int, int]], types: list[str], n: int) -> None:
    qs = _load_questions(types, n)
    facts = [(t, f) for t, _, fl in qs for f in fl]
    lens = [len(f) for _, f in facts]
    print(f"取样：{len(qs)} 题 / {len(facts)} 条 gold 证据句（每型 {n} 题）")
    print(f"证据句长度：中位 {st.median(lens):.0f}｜均值 {st.mean(lens):.0f}｜"
          f"p90 {np.percentile(lens, 90):.0f}｜最长 {max(lens)} 字符")
    print("→ **overlap 必须 ≥ 这个长度**，否则跨块的证据句会被切断，且这是检索之前的、不可恢复的损失。\n")

    keep_full: dict = {}     # (cfg, type) -> [bool]  全句留存
    keep_key: dict = {}      # (cfg, type) -> [bool]  前 120 字符留存（= 下游 fact@k 的口径）
    rank_d: dict = {}
    rank_b: dict = {}
    meta: dict = {}

    for size, ovl in cfgs:
        docs = cm.load_corpus(chunk_size=size, chunk_overlap=ovl)
        blobs = [_norm(d.text) for d in docs]
        meta[(size, ovl)] = (len(docs), st.mean([len(d.text) for d in docs]))
        print(f"[chunk={size} overlap={ovl}] {len(docs)} 片段，均长 {meta[(size, ovl)][1]:.0f} 字符", flush=True)

        owners, qtexts, tlist = [], [], []
        for t, f in facts:
            nf, key = _norm(f), _norm(f)[:_KEY]
            own = {i for i, b in enumerate(blobs) if key in b}
            keep_full.setdefault((size, ovl, t), []).append(any(nf in blobs[i] for i in own))
            keep_key.setdefault((size, ovl, t), []).append(bool(own))
            owners.append(own)
            qtexts.append(f)
            tlist.append(t)

        probe = _DenseProbe(docs)
        for t, r in zip(tlist, probe.rank_of(qtexts, owners)):
            rank_d.setdefault((size, ovl, t), []).append(r)
        del probe                                     # 及时放掉 GPU/内存，下个配置还要编码
        _free_gpu()

        bm25 = BM25Retriever(docs)
        id2i = {d.id: i for i, d in enumerate(docs)}
        for t, f, own in zip(tlist, qtexts, owners):
            if not own:
                rank_b.setdefault((size, ovl, t), []).append(_MISS)
                continue
            hits = bm25.search(f, k=_PROBE)
            pos = [i for i, h in enumerate(hits) if id2i[h.doc.id] in own]
            rank_b.setdefault((size, ovl, t), []).append(pos[0] if pos else _MISS)
        del bm25

    print(f"\n{'=' * 78}\n表A · 切分层：证据句还完不完整（**检索之前**就定死的上限）\n{'=' * 78}")
    print(f"{'chunk':>6}{'ovl':>5}{'片段数':>8}  {'type':12s}{'留存(全句)':>11}{'留存(前120)':>12}")
    for size, ovl in cfgs:
        for t in types:
            print(f"{size:>6}{ovl:>5}{meta[(size, ovl)][0]:>8}  {t:12s}"
                  f"{st.mean(keep_full[(size, ovl, t)]):>11.1%}"
                  f"{st.mean(keep_key[(size, ovl, t)]):>12.1%}")
        print()

    print(f"{'=' * 78}\n表B · embedding 自检：**用证据句原文当 query**，含它的片段排第几\n"
          f"      （最有利输入，理想 top1≈100%；分母是全部证据句，未留存的算未命中）\n{'=' * 78}")
    print(f"{'chunk':>6}{'ovl':>5}  {'type':12s}"
          f"{'d中位':>9}{'d@1':>7}{'d@10':>7}   {'b中位':>9}{'b@1':>7}{'b@10':>7}")
    for size, ovl in cfgs:
        for t in types:
            print(f"{size:>6}{ovl:>5}  {t:12s}"
                  f"{_fmt(rank_d[(size, ovl, t)])}   {_fmt(rank_b[(size, ovl, t)])}")
        print()

    print(f"{'=' * 78}\n表C · 三类均值（**仅供横向排序**，不可用来下结论 —— 见模块注释 ②）\n{'=' * 78}")
    print(f"{'chunk':>6}{'ovl':>5}{'片段数':>8}{'留存(前120)':>12}{'dense@1':>9}{'dense@10':>10}{'BM25@1':>8}")
    for size, ovl in cfgs:
        kk = [v for t in types for v in keep_key[(size, ovl, t)]]
        rd = [v for t in types for v in rank_d[(size, ovl, t)]]
        rb = [v for t in types for v in rank_b[(size, ovl, t)]]
        print(f"{size:>6}{ovl:>5}{meta[(size, ovl)][0]:>8}{st.mean(kk):>12.1%}"
              f"{sum(1 for r in rd if r == 0) / len(rd):>9.1%}"
              f"{sum(1 for r in rd if r < 10) / len(rd):>10.1%}"
              f"{sum(1 for r in rb if r == 0) / len(rb):>8.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--n", type=int, default=30, help="每个题型取多少题（均衡抽样）")
    ap.add_argument("--types", default="temporal,comparison,inference")
    ap.add_argument("--cfgs", default="1200x150,600x150,600x200,500x200",
                    help="chunk 配置列表，形如 600x150（第一项通常放现状基线）")
    ap.add_argument("--pools", default="100,200,300", help="第2层：候选池大小")
    ap.add_argument("--pool", type=int, default=200, help="第3层：已定档的候选池大小")
    ap.add_argument("--reranker", default="bge", choices=["bge", "qwen"])
    ap.add_argument("--chroma", action="store_true",
                    help="第3层：dense 腿改用生产的 Chroma 而非精确余弦探针（排除两者的差异）")
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    cfgs = [tuple(int(x) for x in c.split("x")) for c in args.cfgs.split(",") if c.strip()]
    if args.layer == 1:
        _layer1(cfgs, types, args.n)
    elif args.layer == 0:
        _layer0(cfgs, types, args.n)
    elif args.layer == 2:
        _layer2(cfgs, [int(p) for p in args.pools.split(",")], types, args.n)
    elif args.layer == 3:
        _layer3(cfgs, args.pool, types, args.n, args.reranker, args.chroma)
    else:
        raise SystemExit(f"第 {args.layer} 层尚未实现（本次重建按 embedding→融合→reranker 逐层追加）")


if __name__ == "__main__":
    main()
