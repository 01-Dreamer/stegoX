from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb_image(path: str | Path, resize: bool = True, size: int = 512) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if resize:
        return np.array(image.resize((size, size), Image.LANCZOS))

    width, height = image.size
    pad_width = (-width) % 8
    pad_height = (-height) % 8
    if pad_width or pad_height:
        image = ImageOps.expand(image, (0, 0, pad_width, pad_height), fill=0)
    return np.array(image)


def array_to_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


def save_rgb_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array_to_image(image).save(path)
