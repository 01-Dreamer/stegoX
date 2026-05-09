from __future__ import annotations

from io import StringIO
from typing import ByteString

import torch

from stegox.core import StegoMethod, bits_to_bytes, bytes_to_bits, load_masked_lm


FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


class EditStego(StegoMethod):
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        mask_interval: int = 2,
        candidate_pool_size: int = 16,
        device: str = "cpu",
    ):
        self.tokenizer, self.model = load_masked_lm(model_name, device=device)
        self.mask_interval = max(1, mask_interval)
        self.candidate_pool_size = max(2, candidate_pool_size)
        self.device = device
        self.stopwords = FALLBACK_STOPWORDS

    def encrypt(self, cover: str, payload: ByteString) -> str:
        bit_stream = StringIO(bytes_to_bits(payload, include_header=True))
        input_ids = self._encode_tokens(cover)
        masked_ids = self._mask(input_ids.clone())
        _, indices = self._predict(masked_ids)

        for pos in range(len(masked_ids)):
            if masked_ids[pos].item() != self.tokenizer.mask_token_id:
                continue
            candidates = self._pick_candidates(indices[pos])
            if len(candidates) < 2:
                continue
            input_ids[pos] = self._block_encode_single(candidates, bit_stream)

        return self._decode_tokens(input_ids.tolist())

    def decrypt(self, cover: str, stego_text: str) -> bytes:
        del cover
        input_ids = self._encode_tokens(stego_text)
        masked_ids = self._mask(input_ids.clone())
        _, indices = self._predict(masked_ids)

        message_bits: list[str] = []
        for pos in range(len(masked_ids)):
            if masked_ids[pos].item() != self.tokenizer.mask_token_id:
                continue
            candidates = self._pick_candidates(indices[pos])
            if len(candidates) < 2:
                continue
            message_bits.append(self._block_decode_single(candidates, input_ids[pos].item()))
            decoded = bits_to_bytes("".join(message_bits), has_header=True)
            if self._has_complete_header_payload("".join(message_bits)):
                return decoded
        return bits_to_bytes("".join(message_bits), has_header=True)

    def _encode_tokens(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _decode_tokens(self, ids: list[int]) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def _mask(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
        eligible_count = 0
        for index, token in enumerate(tokens):
            if token.startswith("##"):
                continue
            if token.lower() in self.stopwords or not token.isalpha():
                continue
            if eligible_count % self.mask_interval == 0:
                ids[index] = self.tokenizer.mask_token_id
            eligible_count += 1
        return ids

    def _predict(self, ids: torch.Tensor):
        with torch.no_grad():
            logits = self.model(ids.unsqueeze(0))["logits"][0]
        return logits.sort(dim=-1, descending=True)

    def _pick_candidates(self, id_tensor: torch.Tensor) -> list[int]:
        candidates = []
        for token_id in id_tensor.tolist():
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith("##") or token.lower() in self.stopwords or not token.isalpha():
                continue
            candidates.append(token_id)
            if len(candidates) >= self.candidate_pool_size:
                break
        return candidates

    @staticmethod
    def _block_encode_single(ids: list[int], message_io: StringIO) -> int:
        width = (len(ids) - 1).bit_length()
        bits = message_io.read(width)
        if len(bits) < width:
            bits += "0" * (width - len(bits))
        return ids[min(int(bits, 2), len(ids) - 1)]

    @staticmethod
    def _block_decode_single(ids: list[int], chosen: int) -> str:
        width = (len(ids) - 1).bit_length()
        try:
            index = ids.index(chosen)
        except ValueError:
            return ""
        return f"{index:0{width}b}"

    @staticmethod
    def _has_complete_header_payload(bitstr: str) -> bool:
        if len(bitstr) < 32:
            return False
        payload_len = int(bitstr[:32], 2)
        return len(bitstr) >= 32 + payload_len
