from __future__ import annotations

import random
from typing import ByteString

import torch

from stegox.core import (
    HEADER_BITS,
    StegoMethod,
    bits_to_bytes,
    bytes_to_bits,
    chunk_bits,
    limit_past,
    load_causal_lm,
    model_context_limit,
)


class BinsStego(StegoMethod):
    def __init__(
        self,
        model_name: str = "gpt2",
        bit_block_size: int = 2,
        seed: int = 42,
        device: str = "cpu",
        finish_max_tokens: int = 30,
    ):
        self.tokenizer, self.model = load_causal_lm(model_name, device=device)
        self.bit_block_size = max(1, bit_block_size)
        self.seed = seed
        self.device = device
        self.finish_max_tokens = finish_max_tokens
        self.max_context = model_context_limit(self.model) - 1

        vocab_ids = [idx for idx in range(len(self.tokenizer)) if idx not in self.tokenizer.all_special_ids]
        random.Random(seed).shuffle(vocab_ids)
        num_bins = 2**self.bit_block_size
        base = len(vocab_ids) // num_bins
        self.bins = []
        for index in range(num_bins):
            start = index * base
            end = (index + 1) * base if index < num_bins - 1 else len(vocab_ids)
            self.bins.append(vocab_ids[start:end])
        self.id_to_bits = {
            token_id: f"{bin_index:0{self.bit_block_size}b}"
            for bin_index, bucket in enumerate(self.bins)
            for token_id in bucket
        }

    def encrypt(self, cover: str, payload: ByteString) -> str:
        blocks = chunk_bits(bytes_to_bits(payload, include_header=True), self.bit_block_size)
        past, last_id = self._prime(cover)
        suffix_ids = []

        with torch.no_grad():
            for block in blocks:
                logits, past = self._next_logits(last_id, past)
                bucket = self.bins[int(block, 2)]
                selected = self._best_token_in_bucket(logits, bucket)
                suffix_ids.append(selected)
                last_id = torch.tensor([[selected]], device=self.device)

            suffix_ids.extend(self._finish_sentence(last_id, past))

        return self.tokenizer.decode(suffix_ids, skip_special_tokens=True)

    def decrypt(self, cover: str, stego_text: str) -> bytes:
        suffix_ids = self.tokenizer.encode(stego_text, add_special_tokens=False)
        past, last_id = self._prime(cover)
        bits = []

        with torch.no_grad():
            for token_id in suffix_ids:
                _, past = self._next_logits(last_id, past)
                bits.append(self.id_to_bits.get(token_id, ""))
                last_id = torch.tensor([[token_id]], device=self.device)
                bitstr = "".join(bits)
                if self._has_complete_payload(bitstr):
                    break

        return bits_to_bytes("".join(bits), has_header=True)

    def _prime(self, cover: str):
        cover_ids = self.tokenizer.encode(cover, add_special_tokens=False)
        if cover_ids:
            ids = torch.tensor(cover_ids[-self.max_context :], device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(ids, use_cache=True)
            past = limit_past(out.past_key_values, self.max_context)
            last_id = ids[:, -1:]
        else:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            past = None
            last_id = torch.tensor([[bos]], device=self.device)
        return past, last_id

    def _next_logits(self, last_id, past):
        out = self.model(last_id, past_key_values=past, use_cache=True)
        return out.logits[0, -1], limit_past(out.past_key_values, self.max_context)

    def _best_token_in_bucket(self, logits: torch.Tensor, bucket: list[int]) -> int:
        bucket_tensor = torch.tensor(bucket, dtype=torch.long, device=self.device)
        return bucket_tensor[torch.argmax(logits[bucket_tensor])].item()

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
