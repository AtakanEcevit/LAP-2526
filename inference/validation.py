"""
Input validation for biometric verification.

Provides heuristic-based image quality checks to detect out-of-distribution
inputs and cross-modality mismatches BEFORE model inference.

All checks use OpenCV/numpy — no ML model dependency, <5ms per image.
"""

import io
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from PIL import Image


# ── Validation Result ────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Result of image validation checks.

    Attributes:
        passed:     True if all hard checks pass (image is processable).
        warnings:   Non-fatal issues (e.g., modality mismatch hints).
        confidence: 0.0–1.0 confidence that image matches declared modality.
    """
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "warnings": list(self.warnings),
            "confidence": round(self.confidence, 3),
        }


# ── Modality-Specific Thresholds ─────────────────────────────────────────
#
# These are intentionally permissive to avoid false rejections.
# Tuned against real samples from CEDAR, ATT, and SOCOFing datasets.

_MODALITY_PROFILES = {
    "signature": {
        # Signatures are wide/landscape: W/H ≈ 1.2–2.0
        "aspect_ratio_range": (0.8, 3.0),
        # Signatures have sparse ink strokes → moderate edge density
        "edge_density_range": (0.01, 0.45),
        # Binary/near-binary → low texture variance
        "laplacian_var_range": (5.0, 5000.0),
        # Mostly white background with dark strokes
        "mean_intensity_range": (100, 255),
    },
    "face": {
        # Faces are roughly square: W/H ≈ 0.7–1.5
        "aspect_ratio_range": (0.5, 2.0),
        # Faces have moderate edge density (features, hair, etc.)
        "edge_density_range": (0.02, 0.50),
        # Faces have rich texture
        "laplacian_var_range": (10.0, 15000.0),
        # Mid-range intensity (not blank)
        "mean_intensity_range": (30, 230),
    },
    "fingerprint": {
        # Fingerprints are roughly square: W/H ≈ 0.6–1.8
        "aspect_ratio_range": (0.4, 2.5),
        # Fingerprints have high edge density (ridge patterns)
        "edge_density_range": (0.03, 0.60),
        # Strong ridge/valley texture
        "laplacian_var_range": (20.0, 20000.0),
        # Mid-range intensity
        "mean_intensity_range": (20, 235),
    },
}

# Weights for confidence scoring (sums to 1.0)
_CHECK_WEIGHTS = {
    "aspect_ratio": 0.20,
    "edge_density": 0.30,
    "laplacian_var": 0.30,
    "mean_intensity": 0.20,
}


# ── Public API ───────────────────────────────────────────────────────────

def validate_image(image_input, modality: str) -> ValidationResult:
    """
    Validate that an image is suitable for the declared modality.

    Args:
        image_input: bytes, str path, PIL.Image, or numpy array.
        modality:    "signature", "face", or "fingerprint".

    Returns:
        ValidationResult with passed, warnings, and confidence.
    """
    result = ValidationResult()

    if modality not in _MODALITY_PROFILES:
        result.warnings.append(f"Unknown modality '{modality}', skipping validation.")
        result.confidence = 0.5
        return result

    # ── Load to grayscale numpy ──────────────────────────────────────
    try:
        img = _to_grayscale(image_input)
    except Exception as e:
        result.passed = False
        result.warnings.append(f"Cannot decode image: {e}")
        result.confidence = 0.0
        return result

    # ── Hard checks (reject if failed) ───────────────────────────────
    h, w = img.shape[:2]

    if h < 32 or w < 32:
        result.passed = False
        result.warnings.append(f"Image too small ({w}x{h}px). Minimum 32x32.")
        result.confidence = 0.0
        return result

    std_dev = float(np.std(img))
    if std_dev < 2.0:
        result.passed = False
        result.warnings.append(
            "Image is blank or nearly solid color "
            f"(std dev={std_dev:.1f}, min=2.0)."
        )
        result.confidence = 0.0
        return result

    # ── Soft checks (warnings + reduced confidence) ──────────────────
    profile = _MODALITY_PROFILES[modality]
    confidence_hits = 0.0

    # 1. Aspect ratio
    aspect = w / h
    ar_lo, ar_hi = profile["aspect_ratio_range"]
    if not (ar_lo <= aspect <= ar_hi):
        result.warnings.append(
            f"Unusual aspect ratio ({aspect:.2f}) for {modality}. "
            f"Expected {ar_lo:.1f}–{ar_hi:.1f}."
        )
    else:
        confidence_hits += _CHECK_WEIGHTS["aspect_ratio"]

    # 2. Edge density (Canny)
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (h * w)
    ed_lo, ed_hi = profile["edge_density_range"]
    if not (ed_lo <= edge_density <= ed_hi):
        result.warnings.append(
            f"Edge density ({edge_density:.3f}) outside expected range "
            f"for {modality} ({ed_lo:.2f}–{ed_hi:.2f})."
        )
    else:
        confidence_hits += _CHECK_WEIGHTS["edge_density"]

    # 3. Texture variance (Laplacian)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    lap_var = float(laplacian.var())
    lv_lo, lv_hi = profile["laplacian_var_range"]
    if not (lv_lo <= lap_var <= lv_hi):
        result.warnings.append(
            f"Texture variance ({lap_var:.1f}) outside expected range "
            f"for {modality} ({lv_lo:.0f}–{lv_hi:.0f})."
        )
    else:
        confidence_hits += _CHECK_WEIGHTS["laplacian_var"]

    # 4. Mean intensity
    mean_val = float(np.mean(img))
    mi_lo, mi_hi = profile["mean_intensity_range"]
    if not (mi_lo <= mean_val <= mi_hi):
        result.warnings.append(
            f"Mean intensity ({mean_val:.0f}) outside expected range "
            f"for {modality} ({mi_lo}–{mi_hi})."
        )
    else:
        confidence_hits += _CHECK_WEIGHTS["mean_intensity"]

    # Compute final confidence (base 0.5 + up to 0.5 from check scores)
    # Even with all checks failing, confidence stays at 0.5 (hard checks passed)
    result.confidence = 0.5 + 0.5 * (confidence_hits / 1.0)

    return result


# ── Private Helpers ──────────────────────────────────────────────────────

def _to_grayscale(image_input) -> np.ndarray:
    """Convert any supported input to a grayscale uint8 numpy array."""
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        if not image_input.strip():
            raise ValueError("Empty image path.")
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        if image_input.ndim == 2:
            return image_input.astype(np.uint8)
        raise ValueError(
            f"Unexpected array shape: {image_input.shape}. "
            "Expected (H, W) or (H, W, 3)."
        )
    else:
        raise ValueError(
            f"Unsupported input type: {type(image_input).__name__}. "
            "Expected bytes, str, PIL.Image, or numpy array."
        )

    pil_img = pil_img.convert("L")
    return np.array(pil_img, dtype=np.uint8)
