# stegoX

`stegoX` is a small learning-oriented steganography toolkit. It includes five text steganography methods and a CRoSS-style image steganography workflow.

- edit-based text steganography with a masked language model
- bins-based generation with a causal language model
- fixed-length and Huffman variable-length generation
- a lightweight neural arithmetic-coding style method
- a practical Distribution Copies-inspired method

The code keeps a unified API:

```python
encrypt(cover: str, payload: bytes) -> str
decrypt(cover: str, stego_text: str) -> bytes
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model-backed methods download Hugging Face models on first use.

## Text CLI

```bash
python -m stegox.cli encrypt \
  --method bins \
  --model gpt2 \
  --cover cover.txt \
  --payload payload.txt \
  --stego stego.txt
```

```bash
python -m stegox.cli decrypt \
  --method bins \
  --model gpt2 \
  --cover cover.txt \
  --stego stego.txt
```

For successful decryption, use the same method, model, seed, and method parameters used during encryption.

## Methods

- `edit`: Masks eligible words in the cover and replaces them with ranked masked-LM candidates to encode bits.
- `bins`: Shuffles the vocabulary into deterministic bins and chooses the highest-probability token inside the bin indicated by each bit block.
- `huffman`: Selects candidates from the language model distribution and encodes with either fixed-length ranks or Huffman codes.
- `neural`: Uses a simple floating-point interval update to map bits to top-k language-model samples.
- `discop`: Builds deterministic distribution copies from top-k candidates and encodes bits by choosing a candidate from the selected copy.

All methods prepend a 32-bit payload-length header so generated filler can be ignored during decoding.

## Image CLI

The image workflow is adapted from CRoSS. `private-key` is the prompt bound to the secret image and `public-key` is the prompt used to generate the container image.

Run the full hide and reveal demo:

```bash
python -m stegox.image_cli roundtrip \
  --image asserts/1.png \
  --private-key "Eiffel tower" \
  --public-key "a tree" \
  --output-dir output \
  --num-steps 50
```

Only hide:

```bash
python -m stegox.image_cli hide \
  --image secret.png \
  --private-key "Eiffel tower" \
  --public-key "a tree" \
  --output hide.png
```

Only reveal:

```bash
python -m stegox.image_cli reveal \
  --image hide.png \
  --private-key "Eiffel tower" \
  --public-key "a tree" \
  --output reverse.png
```
