import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYBRID_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "results",
    "hybrid_face",
    "checkpoints",
    "best.pth",
)
SAMPLE_FACE = os.path.join(PROJECT_ROOT, "docs", "assets", "sample_face.png")


def test_hybrid_checkpoint_shapes():
    if not os.path.exists(HYBRID_CHECKPOINT):
        pytest.skip("Hybrid checkpoint artifact is not present.")
    import torch

    checkpoint = torch.load(HYBRID_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]

    assert tuple(state["backbone.conv2d_1a.conv.weight"].shape) == (32, 3, 3, 3)
    assert tuple(state["backbone.last_linear.weight"].shape) == (512, 1792)
    assert tuple(state["classifier.weight"].shape) == (6886, 512)
    assert float(checkpoint["val_threshold"]) == pytest.approx(0.3000000119)


def test_hybrid_engine_extracts_normalized_embedding():
    if not os.path.exists(HYBRID_CHECKPOINT):
        pytest.skip("Hybrid checkpoint artifact is not present.")
    from inference.engine import VerificationEngine

    engine = VerificationEngine()
    engine.load("face", "hybrid", device="cpu")
    embedding = engine.extract_embedding(SAMPLE_FACE)

    assert embedding.shape == (512,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_hybrid_same_image_scores_above_threshold():
    if not os.path.exists(HYBRID_CHECKPOINT):
        pytest.skip("Hybrid checkpoint artifact is not present.")
    from inference.engine import VerificationEngine

    engine = VerificationEngine()
    engine.load("face", "hybrid", device="cpu")
    result = engine.compare(SAMPLE_FACE, SAMPLE_FACE)

    assert result["score"] >= result["threshold"]
    assert result["match"] is True
