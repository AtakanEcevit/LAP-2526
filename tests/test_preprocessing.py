"""
Tests for inference/preprocessing.py

Uses real images from data/raw/ to validate output shapes,
value ranges, and input format support.
"""

import os
import sys
import pytest
import numpy as np
import torch
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.preprocessing import preprocess_image
from inference.config import IMAGE_SIZES

# ── Test image paths (from existing dataset) ────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIGNATURE_IMG = os.path.join(
    PROJECT_ROOT, "data", "raw", "signatures", "CEDAR", "full_org", "original_1_1.png"
)
FACE_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "faces", "att_faces", "s1")
FINGERPRINT_IMG_DIR = os.path.join(
    PROJECT_ROOT, "data", "raw", "fingerprints", "SOCOFing", "Real"
)


def _find_face_img():
    if os.path.isdir(FACE_IMG_DIR):
        for f in os.listdir(FACE_IMG_DIR):
            return os.path.join(FACE_IMG_DIR, f)
    return None


def _find_fingerprint_img():
    if os.path.isdir(FINGERPRINT_IMG_DIR):
        for f in os.listdir(FINGERPRINT_IMG_DIR):
            if f.lower().endswith((".bmp", ".png", ".jpg")):
                return os.path.join(FINGERPRINT_IMG_DIR, f)
    return None


# ── Shape Tests ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_signature_output_shape():
    tensor = preprocess_image(SIGNATURE_IMG, "signature")
    h, w = IMAGE_SIZES["signature"]
    assert tensor.shape == (1, 1, h, w)


@pytest.mark.skipif(_find_face_img() is None, reason="ATT face data not found")
def test_face_output_shape():
    tensor = preprocess_image(_find_face_img(), "face")
    h, w = IMAGE_SIZES["face"]
    assert tensor.shape == (1, 1, h, w)


@pytest.mark.skipif(_find_fingerprint_img() is None, reason="SOCOFing data not found")
def test_fingerprint_output_shape():
    tensor = preprocess_image(_find_fingerprint_img(), "fingerprint")
    h, w = IMAGE_SIZES["fingerprint"]
    assert tensor.shape == (1, 1, h, w)


# ── Value Range ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_output_values_in_range():
    tensor = preprocess_image(SIGNATURE_IMG, "signature")
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert tensor.dtype == torch.float32


# ── Input Format Support ────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_accepts_file_path():
    tensor = preprocess_image(SIGNATURE_IMG, "signature")
    assert isinstance(tensor, torch.Tensor)


@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_accepts_pil_image():
    pil_img = Image.open(SIGNATURE_IMG).convert("L")
    tensor = preprocess_image(pil_img, "signature")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 1  # batch dim


@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_accepts_bytes():
    with open(SIGNATURE_IMG, "rb") as f:
        raw_bytes = f.read()
    tensor = preprocess_image(raw_bytes, "signature")
    assert isinstance(tensor, torch.Tensor)


@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_accepts_numpy_array():
    img = np.array(Image.open(SIGNATURE_IMG).convert("L"), dtype=np.uint8)
    tensor = preprocess_image(img, "signature")
    assert isinstance(tensor, torch.Tensor)


# ── Error Handling ──────────────────────────────────────────────────────

def test_invalid_modality_raises():
    # Create a small dummy image
    dummy = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown modality"):
        preprocess_image(dummy, "invalid_modality")


def test_unsupported_input_type_raises():
    with pytest.raises(ValueError, match="Unsupported image input type"):
        preprocess_image(12345, "signature")
