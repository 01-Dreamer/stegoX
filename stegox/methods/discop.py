from __future__ import annotations

import random
from typing import ByteString

import torch

from stegox.core import (
    HEADER_BITS,
    StegoMethod,
    bits_to_bytes,
    bytes_to_bits,
    limit_past,
    load_causal_lm,
    model_context_limit,
)


class DiscopStego(StegoMethod):
    """A compact Distribution Copies-inspired implementation.

    At each step we take a high-probability candidate set, shuffle it with a
    deterministic seed derived from the shared seed and context token, then
    split it into distribution copies. The next bit block selects the copy and
    the language model still picks the best token inside that copy.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        bits_per_token: int = 1,
        candidate_pool_size: int = 64,
        seed: int = 42,
        device: str = "cpu",
        finish_max_tokens: int = 30,
    ):
        self.tokenizer, self.model = load_causal_lm(model_name, device=device)
        self.bits_per_token = max(1, bits_per_token)
        self.num_copies = 2**self.bits_per_token
        self.candidate_pool_size = max(self.num_copies * 2, candidate_pool_size)
        self.seed = seed
        self.device = device
        self.finish_max_tokens = finish_max_tokens
        self.max_context = model_context_limit(self.model) - 1

    def encrypt(self, cover: str, payload: ByteString) -> str:
        bitstr = bytes_to_bits(payload, include_header=True)
        blocks = [
            bitstr[index : index + self.bits_per_token].ljust(self.bits_per_token, "0")
            for index in range(0, len(bitstr), self.bits_per_token)
        ]
        past, last_id = self._prime(cover)
        suffix_ids = []

        with torch.no_grad():
            for block in blocks:
                logits, past = self._next_logits(last_id, past)
                copies = self._distribution_copies(logits, last_id.item())
                copy = copies[int(block, 2)]
                token_id = self._best_token(logits, copy)
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
                copies = self._distribution_copies(logits, last_id.item())
                copy_index = self._find_copy(copies, token_id)
                if copy_index is None:
                    break
                bits.append(f"{copy_index:0{self.bits_per_token}b}")
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

    def _distribution_copies(self, logits: torch.Tensor, context_token_id: int) -> list[list[int]]:
        _, top_ids = logits.topk(self.candidate_pool_size)
        ids = [token_id for token_id in top_ids.tolist() if token_id not in self.tokenizer.all_special_ids]
        rnd = random.Random((self.seed << 20) ^ context_token_id)
        rnd.shuffle(ids)
        copies = [[] for _ in range(self.num_copies)]
        for index, token_id in enumerate(ids):
            copies[index % self.num_copies].append(token_id)
        return copies

    def _best_token(self, logits: torch.Tensor, copy: list[int]) -> int:
        copy_tensor = torch.tensor(copy, dtype=torch.long, device=self.device)
        return copy_tensor[torch.argmax(logits[copy_tensor])].item()

    @staticmethod
    def _find_copy(copies: list[list[int]], token_id: int) -> int | None:
        for index, copy in enumerate(copies):
            if token_id in copy:
                return index
        return None

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
