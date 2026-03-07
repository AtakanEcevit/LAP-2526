"""
Standalone image preprocessor for inference.

Replicates the exact preprocessing from the training data loaders
(CEDARDataset, ATTFaceDataset, SOCOFingDataset) without importing
from the `data/` package, keeping the inference package self-contained.
"""

import io
import numpy as np
import cv2
import torch
from PIL import Image

from inference.config import IMAGE_SIZES, VALID_MODALITIES


def preprocess_image(image_input, modality: str) -> torch.Tensor:
    """
    Preprocess a raw image for model inference.

    Args:
        image_input: One of:
            - bytes: raw image file bytes
            - str:   path to an image file on disk
            - PIL.Image.Image: already-loaded PIL image
            - numpy.ndarray:   already-loaded image array
        modality: "signature", "face", or "fingerprint"

    Returns:
        Tensor of shape (1, 1, H, W) ready to feed into the model.
        Values in [0.0, 1.0], dtype float32.

    Raises:
        ValueError: if modality is unsupported or image cannot be decoded.
    """
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Must be one of: {sorted(VALID_MODALITIES)}"
        )

    # ── Load to grayscale numpy array ────────────────────────────────────
    img = _load_grayscale(image_input)

    # ── Modality-specific processing ─────────────────────────────────────
    if modality == "signature":
        img = _preprocess_signature(img)
    elif modality == "face":
        img = _preprocess_face(img)
    elif modality == "fingerprint":
        img = _preprocess_fingerprint(img)

    # ── Resize to model's expected input size ────────────────────────────
    h, w = IMAGE_SIZES[modality]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # ── Convert to tensor ────────────────────────────────────────────────
    tensor = torch.from_numpy(img).float() / 255.0  # normalize to [0, 1]
    tensor = tensor.unsqueeze(0).unsqueeze(0)        # (1, 1, H, W)

    return tensor


# ═════════════════════════════════════════════════════════════════════════
# Private helpers
# ═════════════════════════════════════════════════════════════════════════

def _load_grayscale(image_input) -> np.ndarray:
    """Load any supported input type into a single-channel uint8 numpy array."""
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        if not image_input.strip():
            raise ValueError("Empty image path provided.")
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        return image_input.astype(np.uint8)
    else:
        raise ValueError(
            f"Unsupported image input type: {type(image_input).__name__}. "
            "Expected bytes, str path, PIL.Image, or numpy.ndarray."
        )

    # Convert to grayscale
    pil_img = pil_img.convert("L")
    return np.array(pil_img, dtype=np.uint8)


def _preprocess_signature(img: np.ndarray) -> np.ndarray:
    """
    Signature preprocessing (matches CEDARDataset._preprocess):
    Grayscale → Otsu binarization → conditional inversion.
    """
    # Otsu binarization
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is dark (want: white background, dark ink)
    if np.mean(img) < 127:
        img = 255 - img

    return img


def _preprocess_face(img: np.ndarray) -> np.ndarray:
    """
    Face preprocessing (matches ATTFaceDataset._preprocess):
    Grayscale → histogram equalization.
    """
    return cv2.equalizeHist(img)


def _preprocess_fingerprint(img: np.ndarray) -> np.ndarray:
    """
    Fingerprint preprocessing (matches SOCOFingDataset._preprocess):
    Grayscale → CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)
