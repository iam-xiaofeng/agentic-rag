"""Qwen3-Reranker：instruction-aware 的 LLM 式重排器（对症多跳「半相关证据被压低」）。

pointwise cross-encoder（如 bge-reranker）把「只覆盖多跳问题一部分」的证据句判成半相关、压到 top-k 外
（EXPERIMENTS 诊断：claim1「别篇半匹配挤掉」占 ~16%）。Qwen3-Reranker 是 LLM 式重排，可加 **instruction**：
"是否含**任一**回答所需证据、哪怕只覆盖问题一部分"，直接治这个目标错配。

接口对齐 sentence-transformers 的 `CrossEncoder.predict(list[(query, doc)]) -> list[float]`，
所以 `HybridRetriever` 可无感替换（传字符串名 → CrossEncoder；传本类实例 → Qwen3）。

⚠️ 依赖：transformers（Qwen3 需 ≥4.51）+ GPU triton 需系统 `python3.14-dev` 头文件（否则 GPU 推理会在
triton 编译处报错；CPU 可绕开但慢）。
"""

from __future__ import annotations

import torch

_MODEL = "Qwen/Qwen3-Reranker-4B"  # 0.6B 与 bge 持平；4B 才把 fact@8 0.65→0.78、claim1 16%→7%（实验11）
_INSTR = ("Given a multi-hop question, judge whether the document contains ANY evidence needed "
          "to answer it, even if it only covers part of the question.")
_PFX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
_SFX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenReranker:
    """instruction-aware 重排。`predict([(query, doc), ...]) -> list[float]`（yes 概率，越大越相关）。"""

    def __init__(self, model: str = _MODEL, instruction: str = _INSTR, batch_size: int = 8) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).eval().to(dev)
        self.instr, self.bs = instruction, batch_size
        self._no = self.tok.convert_tokens_to_ids("no")
        self._yes = self.tok.convert_tokens_to_ids("yes")

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """对 (query, doc) 列表打分（分批），返回每对的相关概率 [0,1]。"""
        out: list[float] = []
        for i in range(0, len(pairs), self.bs):
            batch = pairs[i:i + self.bs]
            txt = [_PFX + f"<Instruct>: {self.instr}\n<Query>: {q}\n<Document>: {d}" + _SFX for q, d in batch]
            enc = self.tok(txt, return_tensors="pt", padding=True, truncation=True,
                           max_length=2048).to(self.model.device)
            with torch.no_grad():
                logit = self.model(**enc).logits[:, -1, :]
            pair = torch.stack([logit[:, self._no], logit[:, self._yes]], dim=1)
            out += torch.softmax(pair, dim=1)[:, 1].tolist()
        return out
