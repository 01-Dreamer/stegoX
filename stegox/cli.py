from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text steganography command line tool")
    parser.add_argument("action", choices=["encrypt", "decrypt"], help="operation to run")
    parser.add_argument(
        "--method",
        default="bins",
        metavar="METHOD",
        help="one of: edit, bins, huffman, neural, discop",
    )
    parser.add_argument("--model", default=None, help="Hugging Face model name or local path")
    parser.add_argument("--cover", default="cover.txt", help="cover text file")
    parser.add_argument("--payload", default="payload.txt", help="payload file for encryption")
    parser.add_argument("--stego", default="stego.txt", help="stego text input/output file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--mask-interval", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=16)
    parser.add_argument("--bit-block-size", type=int, default=2)
    parser.add_argument("--bits-per-word", type=int, default=2)
    parser.add_argument("--bits-per-token", type=int, default=1)
    parser.add_argument("--encoding-method", default="vlc", choices=["flc", "vlc"])
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finish-max-tokens", type=int, default=30)
    return parser


def create_method(args):
    model = args.model
    if args.method == "edit":
        from stegox.methods.edit import EditStego

        return EditStego(
            model_name=model or "bert-base-cased",
            mask_interval=args.mask_interval,
            candidate_pool_size=args.candidate_pool_size,
            device=args.device,
        )
    if args.method == "bins":
        from stegox.methods.bins import BinsStego

        return BinsStego(
            model_name=model or "gpt2",
            bit_block_size=args.bit_block_size,
            seed=args.seed,
            device=args.device,
            finish_max_tokens=args.finish_max_tokens,
        )
    if args.method == "huffman":
        from stegox.methods.huffman_stego import HuffmanStego

        return HuffmanStego(
            model_name=model or "gpt2",
            bits_per_word=args.bits_per_word,
            encoding_method=args.encoding_method,
            device=args.device,
            finish_max_tokens=args.finish_max_tokens,
        )
    if args.method == "neural":
        from stegox.methods.neural import NeuralStego

        return NeuralStego(
            model_name=model or "gpt2",
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
    from stegox.methods.discop import DiscopStego

    return DiscopStego(
        model_name=model or "gpt2",
        bits_per_token=args.bits_per_token,
        candidate_pool_size=args.candidate_pool_size,
        seed=args.seed,
        device=args.device,
        finish_max_tokens=args.finish_max_tokens,
    )


def main() -> None:
    args = build_parser().parse_args()
    valid_methods = {"edit", "bins", "huffman", "neural", "discop"}
    if args.method not in valid_methods:
        raise SystemExit(f"unknown method: {args.method}. Use one of: edit, bins, huffman, neural, discop")
    stego_method = create_method(args)
    cover = Path(args.cover).read_text(encoding="utf-8")

    if args.action == "encrypt":
        payload = Path(args.payload).read_bytes()
        stego_text = stego_method.encrypt(cover, payload)
        Path(args.stego).write_text(stego_text, encoding="utf-8")
        print(stego_text)
        return

    stego_text = Path(args.stego).read_text(encoding="utf-8")
    secret = stego_method.decrypt(cover, stego_text)
    print(secret.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
