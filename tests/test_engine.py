"""
Tests for inference/engine.py

Uses real checkpoints and dataset images to validate model loading,
embedding extraction, and comparison logic.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import VerificationEngine
from inference.config import MODEL_REGISTRY

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIGNATURE_IMG_1 = os.path.join(
    PROJECT_ROOT, "data", "raw", "signatures", "CEDAR", "full_org", "original_1_1.png"
)
SIGNATURE_IMG_2 = os.path.join(
    PROJECT_ROOT, "data", "raw", "signatures", "CEDAR", "full_org", "original_1_2.png"
)
SIGNATURE_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "results", "siamese_signature_cedar", "checkpoints", "best.pth"
)
PROTO_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "results", "proto_signature_cedar", "checkpoints", "best.pth"
)

HAS_DATA = os.path.exists(SIGNATURE_IMG_1) and os.path.exists(SIGNATURE_IMG_2)
HAS_SIAMESE = os.path.exists(SIGNATURE_CHECKPOINT)
HAS_PROTO = os.path.exists(PROTO_CHECKPOINT)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def siamese_engine():
    """Load Siamese signature engine once for all tests in this module."""
    if not HAS_SIAMESE:
        pytest.skip("Siamese checkpoint not found")
    engine = VerificationEngine()
    engine.load("signature", "siamese")
    return engine


@pytest.fixture(scope="module")
def proto_engine():
    """Load Prototypical signature engine once for all tests in this module."""
    if not HAS_PROTO:
        pytest.skip("Prototypical checkpoint not found")
    engine = VerificationEngine()
    engine.load("signature", "prototypical")
    return engine


# ── Model Loading ───────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_SIAMESE, reason="Siamese checkpoint not found")
def test_load_siamese_signature(siamese_engine):
    assert siamese_engine._loaded
    assert siamese_engine.modality == "signature"
    assert siamese_engine.model_type == "siamese"


@pytest.mark.skipif(not HAS_PROTO, reason="Prototypical checkpoint not found")
def test_load_prototypical_signature(proto_engine):
    assert proto_engine._loaded
    assert proto_engine.modality == "signature"
    assert proto_engine.model_type == "prototypical"


# ── Embedding Extraction ────────────────────────────────────────────────

@pytest.mark.skipif(not (HAS_DATA and HAS_SIAMESE), reason="Data or checkpoint missing")
def test_extract_embedding_shape(siamese_engine):
    emb = siamese_engine.extract_embedding(SIGNATURE_IMG_1)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (128,)


@pytest.mark.skipif(not (HAS_DATA and HAS_SIAMESE), reason="Data or checkpoint missing")
def test_embedding_normalized(siamese_engine):
    emb = siamese_engine.extract_embedding(SIGNATURE_IMG_1)
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 0.05, f"Expected L2 norm ~1.0, got {norm}"


# ── Comparison ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not (HAS_DATA and HAS_SIAMESE), reason="Data or checkpoint missing")
def test_compare_returns_dict(siamese_engine):
    result = siamese_engine.compare(SIGNATURE_IMG_1, SIGNATURE_IMG_2)
    assert "match" in result
    assert "score" in result
    assert "threshold" in result
    assert isinstance(result["match"], bool)
    assert isinstance(result["score"], float)


@pytest.mark.skipif(not (HAS_DATA and HAS_SIAMESE), reason="Data or checkpoint missing")
def test_compare_same_image_high_score(siamese_engine):
    result = siamese_engine.compare(SIGNATURE_IMG_1, SIGNATURE_IMG_1)
    assert result["score"] >= 0.95, \
        f"Same image should score ≥0.95, got {result['score']}"


@pytest.mark.skipif(not (HAS_DATA and HAS_PROTO), reason="Data or checkpoint missing")
def test_proto_compare_returns_score(proto_engine):
    result = proto_engine.compare(SIGNATURE_IMG_1, SIGNATURE_IMG_2)
    assert 0.0 <= result["score"] <= 1.0


# ── Error Handling ──────────────────────────────────────────────────────

def test_not_loaded_raises():
    engine = VerificationEngine()
    with pytest.raises(RuntimeError, match="No model loaded"):
        engine.extract_embedding("dummy.png")


def test_invalid_modality_raises():
    engine = VerificationEngine()
    with pytest.raises(ValueError, match="Unknown modality"):
        engine.load("invalid_modality", "siamese")


def test_invalid_model_type_raises():
    engine = VerificationEngine()
    with pytest.raises(ValueError, match="Unknown model_type"):
        engine.load("signature", "invalid_type")


# ── Info ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_SIAMESE, reason="Siamese checkpoint not found")
def test_info(siamese_engine):
    info = siamese_engine.info()
    assert info["loaded"] is True
    assert info["modality"] == "signature"
    assert info["model_type"] == "siamese"
    assert isinstance(info["threshold"], float)
