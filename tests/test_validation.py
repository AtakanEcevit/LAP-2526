"""
Tests for inference/validation.py

Validates hard rejection (corrupt, blank, tiny), soft warnings
(modality mismatch heuristics), and confidence scoring using
both synthetic and real dataset images.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.validation import validate_image, ValidationResult

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


# ── ValidationResult Basics ─────────────────────────────────────────────

def test_validation_result_defaults():
    r = ValidationResult()
    assert r.passed is True
    assert r.warnings == []
    assert r.confidence == 1.0


def test_validation_result_to_dict():
    r = ValidationResult(passed=False, warnings=["bad"], confidence=0.42)
    d = r.to_dict()
    assert d == {"passed": False, "warnings": ["bad"], "confidence": 0.42}


# ── Hard Reject: Corrupt Image ──────────────────────────────────────────

def test_corrupt_bytes_rejected():
    result = validate_image(b"this is not an image", "signature")
    assert result.passed is False
    assert result.confidence == 0.0
    assert len(result.warnings) > 0
    assert "decode" in result.warnings[0].lower() or "cannot" in result.warnings[0].lower()


# ── Hard Reject: Blank Image ────────────────────────────────────────────

def test_blank_white_image_rejected():
    blank = np.ones((100, 100), dtype=np.uint8) * 255
    result = validate_image(blank, "signature")
    assert result.passed is False
    assert result.confidence == 0.0
    assert any("blank" in w.lower() or "solid" in w.lower() for w in result.warnings)


def test_blank_black_image_rejected():
    blank = np.zeros((100, 100), dtype=np.uint8)
    result = validate_image(blank, "face")
    assert result.passed is False
    assert result.confidence == 0.0


# ── Hard Reject: Tiny Image ─────────────────────────────────────────────

def test_tiny_image_rejected():
    tiny = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    result = validate_image(tiny, "fingerprint")
    assert result.passed is False
    assert result.confidence == 0.0
    assert any("small" in w.lower() or "minimum" in w.lower() for w in result.warnings)


# ── Pass: Synthetic Valid Image ──────────────────────────────────────────

def test_random_noise_passes_hard_checks():
    """Random noise is clearly not blank/tiny, should pass hard checks."""
    noisy = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
    result = validate_image(noisy, "signature")
    assert result.passed is True
    # Confidence may be reduced due to soft check mismatches, but should exist
    assert result.confidence > 0.0


# ── Real Image Tests ─────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_real_signature_passes():
    result = validate_image(SIGNATURE_IMG, "signature")
    assert result.passed is True
    assert result.confidence > 0.5, (
        f"Real signature should have decent confidence, got {result.confidence}"
    )


@pytest.mark.skipif(_find_face_img() is None, reason="ATT face data not found")
def test_real_face_passes():
    result = validate_image(_find_face_img(), "face")
    assert result.passed is True
    assert result.confidence > 0.5


@pytest.mark.skipif(
    _find_fingerprint_img() is None, reason="SOCOFing data not found"
)
def test_real_fingerprint_passes():
    result = validate_image(_find_fingerprint_img(), "fingerprint")
    assert result.passed is True
    assert result.confidence > 0.5


# ── Cross-Modality Mismatch ──────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(SIGNATURE_IMG) or _find_face_img() is None,
    reason="Need both CEDAR and ATT data"
)
def test_face_as_signature_has_lower_confidence():
    """A face image passed as 'signature' should have lower confidence
    than a real signature passed as 'signature'."""
    sig_result = validate_image(SIGNATURE_IMG, "signature")
    face_result = validate_image(_find_face_img(), "signature")

    # Real signature should score higher than a face mislabeled as signature
    assert sig_result.confidence >= face_result.confidence, (
        f"Signature confidence ({sig_result.confidence}) should be >= "
        f"face-as-signature confidence ({face_result.confidence})"
    )


# ── Unknown Modality ─────────────────────────────────────────────────────

def test_unknown_modality_returns_warning():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = validate_image(img, "iris")
    assert result.passed is True  # unknown modality skips validation
    assert len(result.warnings) > 0
    assert result.confidence == 0.5


# ── Unsupported Input Type ───────────────────────────────────────────────

def test_unsupported_input_type():
    result = validate_image(12345, "signature")
    assert result.passed is False
    assert result.confidence == 0.0


# ── File Path Input ──────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_accepts_file_path():
    result = validate_image(SIGNATURE_IMG, "signature")
    assert result.passed is True
    assert isinstance(result.confidence, float)


# ── Bytes Input ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(SIGNATURE_IMG), reason="CEDAR data not found")
def test_accepts_bytes():
    with open(SIGNATURE_IMG, "rb") as f:
        img_bytes = f.read()
    result = validate_image(img_bytes, "signature")
    assert result.passed is True
