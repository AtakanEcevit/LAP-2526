"""
Shared preprocessing functions for all biometric modalities.

Single source of truth used by both data loaders (training) and
the inference pipeline. This module intentionally has zero heavyweight
dependencies beyond cv2/numpy so it can be imported from any context.
"""

import cv2
import numpy as np


# ── Image sizes (H, W) per modality — canonical source ───────────────────
IMAGE_SIZES = {
    "signature":   (155, 220),
    "face":        (105, 105),
    "fingerprint": (224, 224),
}


# ── Modality-specific preprocessing functions ────────────────────────────

def preprocess_signature(img: np.ndarray) -> np.ndarray:
    """CLAHE enhancement for signature images.

    Contrast Limited Adaptive Histogram Equalization preserves
    gradient information in handwriting strokes.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_face(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization for face images.

    Normalizes lighting variations across face photographs.
    """
    return cv2.equalizeHist(img)


def preprocess_fingerprint(img: np.ndarray) -> np.ndarray:
    """CLAHE enhancement for fingerprint images.

    Better than global histogram equalization for fingerprints because
    it preserves ridge/valley detail.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# ── Registry: modality name → preprocessing function ─────────────────────
PREPROCESS_FN = {
    "signature":   preprocess_signature,
    "face":        preprocess_face,
    "fingerprint": preprocess_fingerprint,
}
