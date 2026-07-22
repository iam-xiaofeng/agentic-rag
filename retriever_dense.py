"""稠密向量检索器（bge）—— 实现同一套 `Retriever` 协议。

它是 hybrid 的两条腿之一：BM25 看「词面重叠」，dense 看「语义相近」，两者互补。
bge 把 query 和 passage 都编码成**归一化向量**，用余弦相似度（= 点积）排序。

模型（本地跑，首次自动下载）：
  - 编码：BAAI/bge-small-en-v1.5（384 维，英文、CPU 友好；可升级 bge-base-en-v1.5）
  - bge-*-en-v1.5 官方建议：**只给 query 加检索指令前缀**，passage 不加。

语料向量按 (模型, 语料指纹) 缓存到 .cache/*.npy，第二次起免重算。
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from retriever import Doc, Hit

_MODEL = "BAAI/bge-small-en-v1.5"
# bge-en-v1.5 检索时给 query 的指令前缀（passage 不加）。
_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
_CACHE = pathlib.Path(__file__).resolve().parent / ".cache"


class DenseRetriever:
    """bge 向量检索。建索引 = 把每个片段编码成一个归一化向量。"""

    def __init__(self, docs: list[Doc], model: str = _MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入，别让轻量路径背上 torch

        self.docs = docs
        self.model_name = model
        self.model = SentenceTransformer(model)
        self._emb = self._corpus_embeddings(docs)  # (N, d) float32，已归一化

    def _corpus_embeddings(self, docs: list[Doc]) -> np.ndarray:
        _CACHE.mkdir(exist_ok=True)
        fp = _CACHE / f"dense_{self._fingerprint(docs)}.npy"
        if fp.exists():
            return np.load(fp)
        emb = self.model.encode(
            [d.text for d in docs],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        ).astype("float32")
        np.save(fp, emb)
        return emb

    def _fingerprint(self, docs: list[Doc]) -> str:
        h = hashlib.md5()
        h.update(self.model_name.encode())
        h.update(str(len(docs)).encode())
        if docs:
            h.update(docs[0].id.encode())
            h.update(docs[-1].id.encode())
        return h.hexdigest()[:12]

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = self.model.encode([_QUERY_PROMPT + query], normalize_embeddings=True)[0]
        scores = self._emb @ q  # 归一化后点积 = 余弦相似度
        order = np.argsort(-scores)[:k]
        return [Hit(doc=self.docs[i], score=float(scores[i])) for i in order]
