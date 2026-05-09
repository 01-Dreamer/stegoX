from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRoSS-style image steganography tool")
    parser.add_argument("action", choices=["hide", "reveal", "roundtrip"])
    parser.add_argument("--image", required=True, help="secret image for hide/roundtrip, stego image for reveal")
    parser.add_argument("--private-key", required=True, help="private prompt used for secret image inversion")
    parser.add_argument("--public-key", required=True, help="public prompt used for stego image generation")
    parser.add_argument("--output", default=None, help="output image path for hide/reveal")
    parser.add_argument("--output-dir", default="output", help="output directory for roundtrip")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc.; default auto-detects")
    parser.add_argument("--no-resize", action="store_true", help="pad to a multiple of 8 instead of resizing to 512x512")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from stegox.image.cross import CrossImageStego

    stego = CrossImageStego(
        model_name=args.model,
        num_steps=args.num_steps,
        device=args.device,
        show_progress=not args.no_progress,
    )
    resize = not args.no_resize

    if args.action == "hide":
        output = args.output or "hide.png"
        stego.hide_file(args.image, args.private_key, args.public_key, output, resize=resize)
        print(output)
        return

    if args.action == "reveal":
        output = args.output or "reverse.png"
        stego.reveal_file(args.image, args.private_key, args.public_key, output, resize=resize)
        print(output)
        return

    stego.roundtrip(args.image, args.private_key, args.public_key, args.output_dir, resize=resize)
    print(args.output_dir)


if __name__ == "__main__":
    main()
