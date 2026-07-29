"""稠密向量检索器（bge 编码 + Chroma 向量库）—— 实现同一套 `Retriever` 协议。

它是 hybrid 的两条腿之一：BM25 看「词面重叠」，dense 看「语义相近」，两者互补。
bge 把 query 和 passage 都编码成**归一化向量**，用余弦相似度排序；向量存进 **Chroma**
（持久化到 .cache/chroma/），检索走 Chroma 的近邻查询——比手写 numpy 点积更规范、
可持久化、易扩展；换成 Chroma 后上层（hybrid / 生成 / 评测）一行不动（Retriever 协议）。

模型（本地跑，首次自动下载；有 CUDA 时 sentence-transformers 自动用 GPU）：
  - 编码：BAAI/bge-large-en-v1.5（1024 维，英文强档；GPU 上编码很快）
  - bge-*-en-v1.5 官方建议：**只给 query 加检索指令前缀**，passage 不加。

Chroma 集合按 (模型, 语料指纹) 命名并持久化：语料不变则第二次起直接复用、无需重建。
编码本身另按 .npy 缓存，重建集合时也免重算。
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from retriever import Doc, Hit

_MODEL = "BAAI/bge-large-en-v1.5"
# bge-en-v1.5 检索时给 query 的指令前缀（passage 不加）。
_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
_CACHE = pathlib.Path(__file__).resolve().parent / ".cache"
_CHROMA_DIR = _CACHE / "chroma"
_ADD_BATCH = 2000  # 保守值，Chroma 单次写入上限约 5461


class DenseRetriever:
    """bge 向量检索：编码语料 → 存入 Chroma 持久化集合 → 查询走近邻检索。"""

    def __init__(self, docs: list[Doc], model: str = _MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入，别让轻量路径背上 torch
        import chromadb  # 延迟导入（chromadb 带 onnxruntime 等，较重）

        self.docs = docs
        self.model_name = model
        self.model = SentenceTransformer(model)

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        self.collection = client.get_or_create_collection(
            name=f"dense_{self._fingerprint(docs)}",
            metadata={"hnsw:space": "cosine"},  # 归一化向量下 距离 = 1 - 余弦
        )
        if self.collection.count() < len(docs):  # 空集合或上次没建完 → （重）建
            self._index(docs)

    def _index(self, docs: list[Doc]) -> None:
        """把语料编码成归一化向量，分批 upsert 进 Chroma（幂等：按 id 覆盖）。"""
        emb = self._corpus_embeddings(docs)
        for s in range(0, len(docs), _ADD_BATCH):
            e = min(s + _ADD_BATCH, len(docs))
            self.collection.upsert(
                ids=[str(i) for i in range(s, e)],          # 用列表下标当 id，保证唯一、可映射回 Doc
                embeddings=emb[s:e].tolist(),
                metadatas=[{"source": docs[i].source} for i in range(s, e)],
                documents=[docs[i].text for i in range(s, e)],
            )

    def _corpus_embeddings(self, docs: list[Doc]) -> np.ndarray:
        """bge 编码全部片段（归一化）；按 (模型,语料) 缓存到 .npy，重建 Chroma 时免重算。"""
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
        res = self.collection.query(query_embeddings=[q.tolist()], n_results=k)
        ids, dists = res["ids"][0], res["distances"][0]
        # cosine 空间：距离 = 1 - 余弦 → 转回相似度分（越大越相关），并用下标映射回原始 Doc。
        return [Hit(doc=self.docs[int(i)], score=1.0 - float(d)) for i, d in zip(ids, dists)]
