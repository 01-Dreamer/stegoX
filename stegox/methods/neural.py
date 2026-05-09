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


class NeuralStego(StegoMethod):
    def __init__(
        self,
        model_name: str = "gpt2",
        temperature: float = 1.0,
        top_k: int = 512,
        device: str = "cpu",
    ):
        self.tokenizer, self.model = load_causal_lm(model_name, device=device)
        self.temperature = temperature
        self.top_k = top_k
        self.device = device
        self.max_context = model_context_limit(self.model) - 1

    def encrypt(self, cover: str, payload: ByteString) -> str:
        bits = [int(bit) for bit in bytes_to_bits(payload, include_header=True)]
        context_ids = self.tokenizer.encode(cover, add_special_tokens=False)
        stego_ids = self._encode_arithmetic(bits, context_ids)
        return self.tokenizer.decode(stego_ids, skip_special_tokens=True)

    def decrypt(self, cover: str, stego_text: str) -> bytes:
        context_ids = self.tokenizer.encode(cover, add_special_tokens=False)
        stego_ids = self.tokenizer.encode(stego_text, add_special_tokens=False)
        header_bits = self._decode_arithmetic(stego_ids, context_ids, HEADER_BITS)
        if len(header_bits) < HEADER_BITS:
            return b""
        payload_len = int("".join(str(bit) for bit in header_bits), 2)
        all_bits = self._decode_arithmetic(stego_ids, context_ids, HEADER_BITS + payload_len)
        return bits_to_bytes("".join(str(bit) for bit in all_bits), has_header=True)

    def _encode_arithmetic(self, message_bits: list[int], context_ids: list[int]) -> list[int]:
        prev_ids, past = self._initial_context(context_ids)
        low, high = 0.0, 1.0
        out_ids = []

        with torch.no_grad():
            for bit in message_bits:
                logits, top_ids, probs, past = self._top_distribution(prev_ids, past)
                mid = (low + high) / 2
                if bit == 0:
                    high = mid
                else:
                    low = mid
                cumulative = probs.cumsum(dim=-1)
                index = torch.searchsorted(cumulative, torch.tensor(low, device=self.device), right=True)
                index = min(index.item(), len(top_ids) - 1)
                token_id = top_ids[index].item()
                out_ids.append(token_id)
                prev_ids = torch.tensor([[token_id]], device=self.device)

        return out_ids

    def _decode_arithmetic(self, stego_ids: list[int], context_ids: list[int], total_bits: int) -> list[int]:
        prev_ids, past = self._initial_context(context_ids)
        low, high = 0.0, 1.0
        bits = []

        with torch.no_grad():
            for token_id in stego_ids:
                if len(bits) >= total_bits:
                    break
                _, top_ids, probs, past = self._top_distribution(prev_ids, past)
                matches = (top_ids == token_id).nonzero()
                if len(matches) == 0:
                    break
                rank = matches[0].item()
                upper = probs.cumsum(dim=-1)[rank].item()
                mid = (low + high) / 2
                bit = 0 if upper <= mid else 1
                bits.append(bit)
                if bit == 0:
                    high = mid
                else:
                    low = mid
                prev_ids = torch.tensor([[token_id]], device=self.device)

        return bits

    def _initial_context(self, context_ids: list[int]):
        if context_ids:
            ids = torch.tensor(context_ids[-self.max_context :], device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(ids, use_cache=True)
            return ids[:, -1:], limit_past(out.past_key_values, self.max_context)
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        return torch.tensor([[bos]], device=self.device), None

    def _top_distribution(self, prev_ids, past):
        out = self.model(prev_ids, past_key_values=past, use_cache=True)
        logits = out.logits[0, -1]
        if self.tokenizer.eos_token_id is not None:
            logits[self.tokenizer.eos_token_id] = -1e9
        top_logits, top_ids = logits.topk(self.top_k)
        probs = F.softmax(top_logits / self.temperature, dim=-1)
        return logits, top_ids, probs, limit_past(out.past_key_values, self.max_context)
