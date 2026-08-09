"""用 **RAGAS** 交叉验证本项目的自研指标（默认只跑小样本）。

    python evals/eval_ragas.py runs/dumps/m1_strong_judged.jsonl -n 20

━━━ 为什么不是"再加几个指标" ━━━

这个项目已经退役过一批第三方指标（openevals 的 groundedness / retrieval_relevance /
helpfulness —— 从未提供过信息量），所以**再往表里加一列不是收益，是风险**。

RAGAS 在这里的正确用法是**对表**：我们有三个头号量，其中 `delivered` 是**确定性**的
（gold 证据段 ∩ 检索上下文，纯子串匹配，零裁判噪声）。拿它当尺，去量 RAGAS 的尺：

| RAGAS 指标 | 对应我们的 | 参照系是否真的一样 |
|---|---|---|
| `Faithfulness` | `grounded` | ✅ 都是「答案的论断能否由**检索到的上下文**支撑」 |
| `LLMContextRecall` | `delivered` | ⚠️ **不完全一样**：RAGAS 拆的是 **gold 答案**的 claim， |
|  |  | 我们查的是 **gold 证据段**是否进了上下文。MuSiQue 的 gold 答案 |
|  |  | 常常是一个实体（"Claudia Wells"），拆出来只有一条 claim ⇒ 近似二值。 |
| `AnswerCorrectness` | `correct` | ⚠️ RAGAS 混了 claim F1 + 语义相似度两项 |

**三种结果各有各的用处**：
- **高度一致** → 我的自研指标等价于业界标准口径，可以直接引用 RAGAS 的名字沟通；
- **系统性偏移** → 说明两者参照系不同，得写清差在哪（这本身就是结论）；
- **不相关** → 至少有一把尺是坏的，而 `delivered` 是确定性的 ⇒ 坏的那把不是它。

⚠️ **成本**：Faithfulness 每题 2 次 LLM 调用（先拆 claim 再逐条判），ContextRecall 每题 1 次。
n=20 约 60 次调用。**默认 n=20，别一上来就全量。**

⚠️ **兼容 shim**：ragas 0.4.3 无条件 `import langchain_community.chat_models.vertexai`，
而 langchain-community 0.4.2 已把它移除（该包已 sunset）。我们不用 Vertex，
所以在 import ragas **之前**塞一个占位类。不动主依赖 —— 为了一个死代码路径去降级
langchain-community 会波及 agent 和检索栈。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# ── 兼容 shim：必须在 import ragas 之前 ────────────────────────────────────────
_vertex = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAIStub:            # noqa: D101 —— 仅为满足 ragas 的 import
    ...


_vertex.ChatVertexAI = _ChatVertexAIStub
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertex)
# ─────────────────────────────────────────────────────────────────────────────

from ragas.dataset_schema import SingleTurnSample                    # noqa: E402
from ragas.llms import LangchainLLMWrapper                           # noqa: E402
from ragas.metrics._context_recall import LLMContextRecall           # noqa: E402
from ragas.metrics._faithfulness import Faithfulness                 # noqa: E402

from evals.eval_agentic import load_benchmark                        # noqa: E402
from rag.llm import build_judge                                      # noqa: E402
from rag.runctx import fmt_meta, read_dump                           # noqa: E402

_CHUNK = re.compile(r"(?=\[source:)")


def split_chunks(blobs: list[str]) -> list[str]:
    """把 `format_hits` 拼出来的整块上下文拆回**一条条片段**。

    RAGAS 的 context_precision 之类按「第几个片段相关」算分，整块塞进去会让它失去分辨力。
    """
    out: list[str] = []
    for b in blobs or []:
        out += [c.strip() for c in _CHUNK.split(b) if c.strip()]
    return out


def collect(dump: str, n: int, gold: dict) -> tuple[dict, list[dict]]:
    """从已有 dump 取样 —— **不重跑 agent**，所以这一步不花钱、也不引入新的运行方差。"""
    meta, rows = read_dump(dump)
    rows = [r for r in rows if r.get("answer") and r.get("contexts")]
    if not rows:
        raise SystemExit(f"⛔ {dump} 里没有带 contexts 的行。RAGAS 需要检索上下文，"
                         f"请用 eval_agentic.py 产出的**未判分** dump（判分文件会丢掉 contexts）。")
    step = max(1, len(rows) // n)                       # 等距抽样，保证题型分布不偏
    picked = rows[::step][:n]
    for r in picked:
        r["_reference"] = gold.get(r["example_id"], "")
    return meta, picked


async def score(samples: list[dict], llm) -> list[dict]:
    faith, recall = Faithfulness(llm=llm), LLMContextRecall(llm=llm)
    out = []
    for i, r in enumerate(samples, start=1):
        s = SingleTurnSample(user_input=r["question"], response=r["answer"],
                             retrieved_contexts=split_chunks(r.get("contexts")),
                             reference=r["_reference"])
        row = {"example_id": r["example_id"], "type": r.get("type")}
        for name, m in (("ragas_faithfulness", faith), ("ragas_context_recall", recall)):
            try:
                row[name] = float(await m.single_turn_ascore(s))
            except Exception as e:                       # noqa: BLE001 —— 网关/解析失败
                row[name] = None                         # **失败记 None 排除出均值，绝不当 0 分**
                print(f"  ⚠️ {r['example_id']} {name} 失败：{type(e).__name__}: {str(e)[:70]}", flush=True)
        # 我们自己的三个量，原样带上，后面直接对表
        for k in ("correct", "grounded"):
            row[f"ours_{k}"] = (1.0 if (r.get("judge") or {}).get(k) else 0.0) if r.get("judge") else None
        row["ours_delivered"] = r.get("delivered")
        out.append(row)
        if i % 5 == 0:
            print(f"  … {i}/{len(samples)}", flush=True)
    return out


def _corr(xs: list[float], ys: list[float]) -> float:
    """Pearson r，纯手算（不引 scipy）。样本 <3 或任一侧无方差时返回 nan。"""
    p = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(p) < 3:
        return float("nan")
    n = len(p)
    mx, my = sum(a for a, _ in p) / n, sum(b for _, b in p) / n
    sxy = sum((a - mx) * (b - my) for a, b in p)
    sxx = sum((a - mx) ** 2 for a, _ in p)
    syy = sum((b - my) ** 2 for _, b in p)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")


def report(rows: list[dict]) -> None:
    col = lambda k: [r.get(k) for r in rows]                                  # noqa: E731
    ok = lambda v: [x for x in v if x is not None]                            # noqa: E731
    mean = lambda v: (sum(ok(v)) / len(ok(v))) if ok(v) else float("nan")     # noqa: E731

    print(f"\n{'=' * 84}\nRAGAS ↔ 自研指标 对表（n={len(rows)}）\n{'=' * 84}")
    print(f"{'指标':<26}{'均值':>9}{'有效n':>7}")
    for k in ("ragas_faithfulness", "ragas_context_recall", "ours_grounded", "ours_correct", "ours_delivered"):
        print(f"{k:<26}{mean(col(k)):>9.3f}{len(ok(col(k))):>7}")

    print(f"\n{'对表（Pearson r）':<44}{'r':>8}   读法")
    pairs = [("ragas_faithfulness", "ours_grounded", "两者参照系一致，应当强相关"),
             ("ragas_context_recall", "ours_delivered", "参照系不同（答案claim vs 证据段），弱相关是预期"),
             ("ragas_faithfulness", "ours_correct", "不该强相关：有依据 ≠ 答对"),
             ("ragas_context_recall", "ours_correct", "检索到了 ≠ 用对了")]
    for a, b, how in pairs:
        r = _corr(col(a), col(b))
        print(f"  {a} ↔ {b:<18}{r:>8.3f}   {how}")

    print("\n  ▸ **`ours_delivered` 是确定性的**（gold 证据段 ∩ 检索上下文，纯子串匹配，零裁判噪声）。")
    print("    它与某个 RAGAS 指标不一致时，先查那个 RAGAS 指标的参照系是什么，别默认自研的错。")
    print("  ▸ r 只说明**同向**，不说明**同量纲**。要替换指标必须再看逐题分歧的样本，不能只看 r。")


def main() -> None:
    ap = argparse.ArgumentParser(description="用 RAGAS 交叉验证自研指标（小样本）")
    ap.add_argument("dump", help="eval_agentic.py 产出的 dump（**要带 contexts**，即未判分那个）")
    ap.add_argument("-n", type=int, default=20, help="抽几题（默认 20 —— 这是小样本试跑，别一上来全量）")
    ap.add_argument("--benchmark", default=None, help="缺省从 dump 的 __meta__ 读")
    ap.add_argument("--out", default=None, help="逐题结果 JSONL")
    args = ap.parse_args()

    meta0, _ = read_dump(args.dump)
    bench = args.benchmark or meta0.get("benchmark") or "musique"
    examples, _, _ = load_benchmark(bench)
    gold = {str(e.id): e.outputs.get("reference", "") for e in examples}

    meta, picked = collect(args.dump, args.n, gold)
    print(f"RAGAS 交叉验证：{len(picked)} 题 × 2 个指标 ≈ {len(picked) * 3} 次 LLM 调用")
    print(f"   被测 dump：{fmt_meta(meta)}")
    llm = LangchainLLMWrapper(build_judge())
    rows = asyncio.run(score(picked, llm))
    if args.out:
        pathlib.Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    report(rows)


if __name__ == "__main__":
    main()
