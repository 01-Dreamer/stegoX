from __future__ import annotations

import abc
from typing import ByteString, Iterable, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from transformers.models.bert import BertConfig
from transformers.models.bloom import BloomConfig
from transformers.models.distilbert import DistilBertConfig
from transformers.models.gpt2 import GPT2Config
from transformers.models.opt import OPTConfig
from transformers.models.roberta import RobertaConfig


HEADER_BITS = 32


class StegoMethod(abc.ABC):
    @abc.abstractmethod
    def encrypt(self, cover: str, payload: ByteString) -> str:
        """Embed payload bytes into cover text and return stego text."""

    @abc.abstractmethod
    def decrypt(self, cover: str, stego_text: str) -> bytes:
        """Extract payload bytes from stego text."""


def bytes_to_bits(payload: ByteString, include_header: bool = True) -> str:
    payload_bits = "".join(f"{byte:08b}" for byte in payload)
    if not include_header:
        return payload_bits
    return f"{len(payload_bits):0{HEADER_BITS}b}" + payload_bits


def bits_to_bytes(bitstr: str, has_header: bool = True) -> bytes:
    if has_header:
        if len(bitstr) < HEADER_BITS:
            return b""
        payload_len = int(bitstr[:HEADER_BITS], 2)
        bitstr = bitstr[HEADER_BITS : HEADER_BITS + payload_len]
    valid_len = len(bitstr) - (len(bitstr) % 8)
    if valid_len <= 0:
        return b""
    bitstr = bitstr[:valid_len]
    return int(bitstr, 2).to_bytes(valid_len // 8, byteorder="big")


def chunk_bits(bitstr: str, size: int) -> list[str]:
    return [bitstr[i : i + size].ljust(size, "0") for i in range(0, len(bitstr), size)]


def load_masked_lm(model_name: str, device: str = "cpu"):
    config = AutoConfig.from_pretrained(model_name)
    if not isinstance(config, (BertConfig, RobertaConfig, DistilBertConfig)):
        raise ValueError(
            f"{model_name!r} is not a supported masked language model. "
            "Use a BERT, RoBERTa, or DistilBERT checkpoint."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def load_causal_lm(model_name: str, device: str = "cpu"):
    config = AutoConfig.from_pretrained(model_name)
    if not isinstance(config, (GPT2Config, BloomConfig, OPTConfig)):
        raise ValueError(
            f"{model_name!r} is not a common causal language model. "
            "Use GPT-2, Bloom, OPT, or a compatible checkpoint."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def model_context_limit(model, fallback: int = 1024) -> int:
    for name in ("n_positions", "max_position_embeddings", "seq_length"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and value > 1:
            return value
    return fallback


def limit_past(past_key_values, max_length: Optional[int] = None):
    if past_key_values is None:
        return None
    max_len = max_length or 1023
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_len)
        return past_key_values

    truncated = []
    for layer in past_key_values:
        if len(layer) < 2:
            truncated.append(layer)
            continue
        key, value = layer[:2]
        if key.size(-2) > max_len:
            key = key[..., -max_len:, :].contiguous()
            value = value[..., -max_len:, :].contiguous()
        truncated.append((key, value, *layer[2:]))
    return tuple(truncated)


def tensor_from_ids(ids: Iterable[int], device: str) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.long, device=device)
