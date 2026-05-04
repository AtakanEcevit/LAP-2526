"""
Image preprocessor for inference.

Uses the shared preprocessing functions from data.preprocessing
(single source of truth) to ensure training/inference consistency.
"""

import io
import numpy as np
import cv2
import torch
from PIL import Image

from data.preprocessing import PREPROCESS_FN, IMAGE_SIZES
from inference.config import VALID_MODALITIES


def preprocess_image(
    image_input,
    modality: str,
    in_channels: int = 1,
    image_size: tuple = None,
) -> torch.Tensor:
    """
    Preprocess a raw image for model inference.

    Args:
        image_input: bytes, str path, PIL.Image, or numpy.ndarray
        modality:    "signature", "face", or "fingerprint"
        in_channels: 1 for grayscale models, 3 for RGB models
        image_size:  (H, W) override; defaults to IMAGE_SIZES[modality]

    Returns:
        Tensor of shape (1, C, H, W) ready to feed into the model.
        Values in [0.0, 1.0], dtype float32.
    """
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Must be one of: {sorted(VALID_MODALITIES)}"
        )

    # ── Load image ───────────────────────────────────────────────────────
    if in_channels == 3:
        img = _load_rgb(image_input)
    else:
        img = _load_grayscale(image_input)

    # ── Modality-specific processing (shared with training loaders) ──────
    img = PREPROCESS_FN[modality](img)

    # ── Resize to model's expected input size ────────────────────────────
    h, w = image_size if image_size is not None else IMAGE_SIZES[modality]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # ── Convert to tensor ────────────────────────────────────────────────
    tensor = torch.from_numpy(img).float() / 255.0  # normalize to [0, 1]
    if in_channels == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    else:
        tensor = tensor.unsqueeze(0).unsqueeze(0)       # (1, 1, H, W)

    return tensor


# ═════════════════════════════════════════════════════════════════════════
# Private helpers
# ═════════════════════════════════════════════════════════════════════════

def _load_pil(image_input) -> "Image.Image":
    """Load any supported input type into a PIL image."""
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input))
    if isinstance(image_input, str):
        if not image_input.strip():
            raise ValueError("Empty image path provided.")
        return Image.open(image_input)
    if isinstance(image_input, Image.Image):
        return image_input
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input)
    raise ValueError(
        f"Unsupported image input type: {type(image_input).__name__}. "
        "Expected bytes, str path, PIL.Image, or numpy.ndarray."
    )


def _load_grayscale(image_input) -> np.ndarray:
    """Load any supported input type into a single-channel uint8 numpy array."""
    pil_img = _load_pil(image_input)
    return np.array(pil_img.convert("L"), dtype=np.uint8)


def _load_rgb(image_input) -> np.ndarray:
    """Load any supported input type into an (H, W, 3) uint8 RGB numpy array."""
    pil_img = _load_pil(image_input)
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)
