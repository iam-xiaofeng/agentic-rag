"""**多臂矩阵**：一批题、一个检索器、N 个配置 —— 一次跑完，产出可直接配对的 dump。

    python evals/run_matrix.py --per-type 30 \\
        --arm base   model=deepseek-v4-pro agent=react   topk=8 \\
        --arm strong model=gpt-5.6-luna    agent=react   topk=8 \\
        --arm plan   model=deepseek-v4-pro agent=planner topk=8

━━━ 为什么需要它（重构 Step 2）━━━

此前所有 MuSiQue 结论都卡在 **n=21**，而追的效应是 0.05 量级 —— 区间宽 ±0.2，
**在设计上就不可能有结论**（实验26/27/28 共 14 个区间，13 个跨 0）。样本量上不去的原因有三个，
这个脚本把三个一起解决：

  ① 每加一个臂就要重建一次检索器（重排器加载 + 21100 段向量库）→ 这里**建一次，所有臂复用**。
  ② 跑一半挂掉就全丢 → 每臂独立 dump + `--resume`，挂了接着跑。
  ③ 两个臂抽到的题不保证一样 → **抽样在最外层做一次**，所有臂共用同一批 `example_id`，
     配对是天然成立的。配对做不成，差值会被题目难度的方差整个淹掉。

**臂之间只应差一个变量。** 每个 dump 的 `__meta__` 记着完整配置，
`eval_judge.py --baseline` 会在配对前自动比对并把「这次实验真正改了什么」打印出来。

臂参数（`key=value`，缺省沿用当前环境）：
    model=<endpoints.json 里的模型名>   agent=react|planner   topk=<int>
    decompose=1                        reranker=bge|qwen
"""

from __future__ import annotations

# 让 `python evals/xxx.py` 直接可跑：把仓库根放进 sys.path（否则 rag.* 导不到）。
import pathlib as _pl, sys as _sys
if str(_pl.Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import os
import time

from evals.eval_agentic import build_retriever, execute, load_benchmark, report, sample
from rag.agent import build_runner
from rag.runctx import fmt_meta, read_dump, snapshot

_ENV = {"topk": "RAG_TOPK", "decompose": "RAG_DECOMPOSE", "agent": "RAG_AGENT",
        "pool": "RAG_POOL", "max_hops": "RAG_MAX_HOPS"}


def parse_arm(tokens: list[str]) -> tuple[str, dict]:
    name, cfg = tokens[0], {}
    for t in tokens[1:]:
        if "=" not in t:
            raise SystemExit(f"臂 {name!r} 的参数 {t!r} 不是 key=value")
        k, v = t.split("=", 1)
        cfg[k.strip()] = v.strip()
    return name, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="多臂矩阵：同一批题跑 N 个配置，产出可配对的 dump")
    ap.add_argument("--arm", nargs="+", action="append", required=True, metavar="NAME K=V",
                    help="可重复。第一个 token 是臂名，其余是 key=value")
    ap.add_argument("--benchmark", choices=["musique", "multihoprag"], default="musique")
    ap.add_argument("--per-type", type=int, default=30)
    ap.add_argument("--types", default="")
    ap.add_argument("--outdir", default="runs/dumps", help="每臂写 <outdir>/<臂名>.jsonl")
    ap.add_argument("--tag", default="", help="文件名前缀，用来区分几轮矩阵（如 m1_）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="每臂都接着已有 dump 跑")
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arm]
    names = [n for n, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"臂名重复：{names}")

    default_types = {"musique": ["2hop", "3hop", "4hop"],
                     "multihoprag": ["comparison_query", "inference_query", "temporal_query"]}
    types = [t.strip() for t in args.types.split(",") if t.strip()] or default_types[args.benchmark]

    examples, qtype, corpus = load_benchmark(args.benchmark)
    picked = sample(examples, qtype, types, args.per_type)          # ★ 抽一次，所有臂共用
    print(f"矩阵：{len(arms)} 臂 × {len(picked)} 题（{', '.join(types)} 各 {args.per_type}）"
          f" = {len(arms) * len(picked)} 次 agent 运行")

    outdir = _pl.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    retr_cache: dict[tuple, object] = {}
    results: dict[str, tuple[dict, list]] = {}
    snapshot_env = dict(os.environ)

    for name, cfg in arms:
        os.environ.clear()
        os.environ.update(snapshot_env)
        for k, v in cfg.items():
            if k in _ENV:
                os.environ[_ENV[k]] = v
        model = cfg.get("model")
        reranker = cfg.get("reranker", "bge")
        # 检索器按「影响检索的配置」缓存：换模型/换 agent 不需要重建，换 reranker/分解才要。
        key = (reranker, cfg.get("decompose", ""), cfg.get("pool", ""))
        if key not in retr_cache:
            print(f"\n[{name}] 建检索器 {key}…", flush=True)
            retr_cache[key] = build_retriever(corpus, reranker, model)
        runner = build_runner(retr_cache[key], model, cfg.get("agent"))
        out = str(outdir / f"{args.tag}{name}.jsonl")
        meta = snapshot(benchmark=args.benchmark, per_type=args.per_type, types=types,
                        n_sampled=len(picked), reranker=reranker, arm=name,
                        answer_model=model, agent=runner.kind)
        print(f"\n{'━' * 92}\n[{name}] {fmt_meta(meta)}\n  → {out}\n{'━' * 92}", flush=True)
        t0 = time.time()
        rows = execute(picked, qtype, runner, out, meta, args.concurrency, args.resume)
        print(f"[{name}] 完成，用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
        results[name] = (meta, rows)

    os.environ.clear()
    os.environ.update(snapshot_env)
    for name, (meta, rows) in results.items():
        print(f"\n\n{'█' * 92}\n█ 臂 {name}\n{'█' * 92}")
        report(rows, meta)

    base = names[0]
    print(f"\n\n{'=' * 92}\n下一步：裁判打头号三分 + 同题配对（基线 = 第一个臂 {base!r}）\n{'=' * 92}")
    for n in names[1:]:
        print(f"  python evals/eval_judge.py {outdir / (args.tag + n)}.jsonl "
              f"--baseline {outdir / (args.tag + base)}.jsonl --out runs/dumps/{args.tag}{n}_judged.jsonl")
    print(f"\n  地板/天花板（回答「还剩多少空间」，与本轮同 --per-type 才是同一批题）：\n"
          f"  python evals/eval_ceiling.py --per-type {args.per_type} "
          f"--dump runs/dumps/{args.tag}{base}_judged.jsonl")


if __name__ == "__main__":
    main()
