"""一条命令把两个评测集拉到 `data/` —— **clone 下来就能复现**。

    python scripts/fetch_data.py            # 两个都拉
    python scripts/fetch_data.py --only musique

为什么不把数据直接提交进仓库：
  · MuSiQue（**默认评测集**）走 HuggingFace，validation 全量约 100MB 级，提交进 git 不合适；
  · MultiHop-RAG 的两个文件合计 11.5MB，提交虽可行但会让 `git clone` 变重，
    而它**已被本项目的探针证伪**（捷径率 99.3%，见 README §4.2），只作历史对照。
  ⇒ 统一用这个脚本拉，`data/` 保持 gitignore。**可复现性靠脚本保证，不靠把数据塞进版本库。**

拉完自检会打印每个数据集的规模，与 README 里写的数字对得上才算成功。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# MultiHop-RAG 官方仓库的原始文件（Apache-2.0）
_MHRAG = {
    "MultiHopRAG.json": "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json",
    "corpus.json": "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/corpus.json",
}


def fetch_musique() -> None:
    """MuSiQue 走 HuggingFace `datasets` 缓存 —— 本项目的加载器直接读缓存，不落 data/。"""
    print("▸ MuSiQue（默认评测集）：通过 HuggingFace datasets 拉取并缓存…")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("⛔ 缺 datasets：pip install -r requirements.txt")
    d = load_dataset("dgslibisey/MuSiQue", split="validation")
    n_para = len({(p["title"], p["paragraph_text"]) for r in d for p in r["paragraphs"]})
    print(f"  ✅ {len(d)} 题，去重后 {n_para} 段候选（README 写的是 2417 题 / 21100 段）")
    print("  （缓存在 ~/.cache/huggingface，本项目 rag/corpus_musique.py 直接读它，不落 data/）")


def fetch_multihoprag() -> None:
    print("▸ MultiHop-RAG（历史对照，已被探针证伪）：下载到 data/ …")
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in _MHRAG.items():
        dst = DATA / name
        if dst.exists():
            print(f"  · {name} 已存在，跳过（{dst.stat().st_size / 1e6:.1f} MB）")
            continue
        print(f"  · 下载 {name} …", flush=True)
        urllib.request.urlretrieve(url, dst)      # noqa: S310 —— 固定的 https 官方源
        print(f"    ✅ {dst.stat().st_size / 1e6:.1f} MB")
    q = json.loads((DATA / "MultiHopRAG.json").read_text(encoding="utf-8"))
    import collections
    c = collections.Counter(x.get("question_type") for x in q)
    print(f"  ✅ {len(q)} 题 {dict(c)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="拉取评测集到 data/")
    ap.add_argument("--only", choices=["musique", "multihoprag"], default=None)
    args = ap.parse_args()
    if args.only in (None, "musique"):
        fetch_musique()
    if args.only in (None, "multihoprag"):
        fetch_multihoprag()
    print("\n完成。自检：python -c \"from rag.corpus_musique import load_corpus; print(len(load_corpus()),'段')\"")


if __name__ == "__main__":
    main()
