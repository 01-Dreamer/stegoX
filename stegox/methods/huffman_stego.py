from __future__ import annotations

from typing import ByteString

import torch
import torch.nn.functional as F

from stegox.core import (
    HEADER_BITS,
    StegoMethod,
    bits_to_bytes,
    bytes_to_bits,
    limit_past,
    load_causal_lm,
    model_context_limit,
)
from stegox.methods.huffman import HuffmanCoding, HeapNode


SKIP_TOKENS = {",", ";", ":", '"', "“", "”"}


class HuffmanStego(StegoMethod):
    def __init__(
        self,
        model_name: str = "gpt2",
        bits_per_word: int = 2,
        encoding_method: str = "vlc",
        device: str = "cpu",
        finish_max_tokens: int = 30,
    ):
        self.tokenizer, self.model = load_causal_lm(model_name, device=device)
        self.bits_per_word = max(1, bits_per_word)
        self.top_m = 2**self.bits_per_word
        self.encoding_method = encoding_method.lower()
        if self.encoding_method not in {"flc", "vlc"}:
            raise ValueError("encoding_method must be 'flc' or 'vlc'")
        self.device = device
        self.finish_max_tokens = finish_max_tokens
        self.max_context = model_context_limit(self.model) - 1

    def encrypt(self, cover: str, payload: ByteString) -> str:
        bitstr = bytes_to_bits(payload, include_header=True)
        offset = 0
        past, last_id = self._prime(cover)
        suffix_ids = []

        with torch.no_grad():
            while offset < len(bitstr):
                logits, past = self._next_logits(last_id, past)
                candidates, probs = self._candidate_distribution(logits)
                if self.encoding_method == "flc":
                    chunk = bitstr[offset : offset + self.bits_per_word].ljust(self.bits_per_word, "0")
                    rank = min(int(chunk, 2), len(candidates) - 1)
                    offset += self.bits_per_word
                else:
                    root, _ = self._huffman(probs)
                    rank, consumed = self._walk_tree(root, bitstr[offset:])
                    offset += consumed
                token_id = candidates[rank]
                suffix_ids.append(token_id)
                last_id = torch.tensor([[token_id]], device=self.device)

            suffix_ids.extend(self._finish_sentence(last_id, past))

        return self.tokenizer.decode(suffix_ids, skip_special_tokens=True)

    def decrypt(self, cover: str, stego_text: str) -> bytes:
        suffix_ids = self.tokenizer.encode(stego_text, add_special_tokens=False)
        past, last_id = self._prime(cover)
        bits = []

        with torch.no_grad():
            for token_id in suffix_ids:
                logits, past = self._next_logits(last_id, past)
                candidates, probs = self._candidate_distribution(logits)
                if token_id not in candidates:
                    break
                rank = candidates.index(token_id)
                if self.encoding_method == "flc":
                    bits.append(f"{rank:0{self.bits_per_word}b}")
                else:
                    _, huffman = self._huffman(probs)
                    bits.append(huffman.codes[rank])
                last_id = torch.tensor([[token_id]], device=self.device)
                if self._has_complete_payload("".join(bits)):
                    break

        return bits_to_bytes("".join(bits), has_header=True)

    def _prime(self, cover: str):
        cover_ids = self.tokenizer.encode(cover, add_special_tokens=False)
        if cover_ids:
            ids = torch.tensor(cover_ids[-self.max_context :], device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(ids, use_cache=True)
            return limit_past(out.past_key_values, self.max_context), ids[:, -1:]
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        return None, torch.tensor([[bos]], device=self.device)

    def _next_logits(self, last_id, past):
        out = self.model(last_id, past_key_values=past, use_cache=True)
        return out.logits[0, -1], limit_past(out.past_key_values, self.max_context)

    def _candidate_distribution(self, logits: torch.Tensor) -> tuple[list[int], list[float]]:
        top_logits, top_ids = logits.topk(self.top_m * 4)
        candidates = []
        candidate_logits = []
        for logit, token_id in zip(top_logits.tolist(), top_ids.tolist()):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token in SKIP_TOKENS:
                continue
            candidates.append(token_id)
            candidate_logits.append(logit)
            if len(candidates) >= self.top_m:
                break
        if len(candidates) < 2:
            candidates = top_ids[: self.top_m].tolist()
            candidate_logits = top_logits[: self.top_m].tolist()
        probs = F.softmax(torch.tensor(candidate_logits), dim=-1).tolist()
        return candidates, probs

    @staticmethod
    def _huffman(probs: list[float]) -> tuple[HeapNode, HuffmanCoding]:
        huffman = HuffmanCoding()
        huffman.make_heap_from_array(probs)
        huffman.merge_nodes()
        root = huffman.make_codes()
        if root is None:
            raise ValueError("empty Huffman candidate distribution")
        return root, huffman

    @staticmethod
    def _walk_tree(root: HeapNode, bits: str) -> tuple[int, int]:
        node = root
        consumed = 0
        for bit in bits or "0":
            if node.token is not None:
                break
            node = node.right if bit == "1" else node.left
            consumed += 1
            if node is None:
                return 0, consumed
        while node.token is None:
            node = node.left
            consumed += 1
        return node.token, max(1, consumed)

    def _finish_sentence(self, last_id, past) -> list[int]:
        out_ids = []
        for _ in range(self.finish_max_tokens):
            logits, past = self._next_logits(last_id, past)
            token_id = torch.argmax(logits).item()
            out_ids.append(token_id)
            last_id = torch.tensor([[token_id]], device=self.device)
            if any(mark in self.tokenizer.decode([token_id]) for mark in (".", "!", "?")):
                break
        return out_ids

    @staticmethod
    def _has_complete_payload(bitstr: str) -> bool:
        if len(bitstr) < HEADER_BITS:
            return False
        payload_len = int(bitstr[:HEADER_BITS], 2)
        return len(bitstr) >= HEADER_BITS + payload_len
