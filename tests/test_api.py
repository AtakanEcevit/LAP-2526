"""
Integration tests for inference/api.py

Uses FastAPI's TestClient — no real server started.
Tests the full HTTP round-trip with real model checkpoints.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIGNATURE_IMG_1 = os.path.join(
    PROJECT_ROOT, "data", "raw", "signatures", "CEDAR", "full_org", "original_1_1.png"
)
SIGNATURE_IMG_2 = os.path.join(
    PROJECT_ROOT, "data", "raw", "signatures", "CEDAR", "full_org", "original_1_2.png"
)
SIAMESE_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "results", "siamese_signature_cedar", "checkpoints", "best.pth"
)

HAS_DATA = os.path.exists(SIGNATURE_IMG_1) and os.path.exists(SIGNATURE_IMG_2)
HAS_MODEL = os.path.exists(SIAMESE_CHECKPOINT)
CAN_TEST = HAS_DATA and HAS_MODEL

# Try importing FastAPI test client
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a temporary enrollment store."""
    if not HAS_FASTAPI:
        pytest.skip("FastAPI not installed")

    # Patch the enrollment store to use a temp file
    import inference.api as api_module
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "test_enrollments.json")

    from inference.enrollment_store import EnrollmentStore
    api_module._store = EnrollmentStore(store_path=tmp_path)

    from inference.api import app
    return TestClient(app)


# ── Health ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_available" in data
    assert "enrollment_count" in data


# ── Compare ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (CAN_TEST and HAS_FASTAPI), reason="Data/model/FastAPI missing")
def test_compare_round_trip(client):
    with open(SIGNATURE_IMG_1, "rb") as f1, open(SIGNATURE_IMG_2, "rb") as f2:
        resp = client.post(
            "/compare",
            data={"modality": "signature", "model": "siamese"},
            files=[
                ("image1", ("img1.png", f1, "image/png")),
                ("image2", ("img2.png", f2, "image/png")),
            ],
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "match" in data
    assert "score" in data
    assert "threshold" in data
    assert isinstance(data["score"], float)


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_compare_invalid_modality(client):
    # Create a small dummy image in memory
    dummy_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    resp = client.post(
        "/compare",
        data={"modality": "invalid", "model": "siamese"},
        files=[
            ("image1", ("img1.png", dummy_bytes, "image/png")),
            ("image2", ("img2.png", dummy_bytes, "image/png")),
        ],
    )
    assert resp.status_code == 400


# ── Enroll + Verify ─────────────────────────────────────────────────────

@pytest.mark.skipif(not (CAN_TEST and HAS_FASTAPI), reason="Data/model/FastAPI missing")
def test_enroll_and_verify(client):
    # Enroll
    with open(SIGNATURE_IMG_1, "rb") as f:
        resp = client.post(
            "/enroll",
            data={"user_id": "test_user", "modality": "signature", "model": "siamese"},
            files=[("images", ("img1.png", f, "image/png"))],
        )
    assert resp.status_code == 200
    assert resp.json()["sample_count"] == 1

    # Verify
    with open(SIGNATURE_IMG_2, "rb") as f:
        resp = client.post(
            "/verify",
            data={"user_id": "test_user"},
            files=[("image", ("query.png", f, "image/png"))],
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "match" in data
    assert "score" in data
    assert data["user_id"] == "test_user"


# ── Verify Unknown User ────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_verify_unknown_user(client):
    dummy_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    resp = client.post(
        "/verify",
        data={"user_id": "nonexistent_user"},
        files=[("image", ("q.png", dummy_bytes, "image/png"))],
    )
    assert resp.status_code == 404


# ── List Users ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_list_users(client):
    resp = client.get("/users")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ── Delete User ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not (CAN_TEST and HAS_FASTAPI), reason="Data/model/FastAPI missing")
def test_delete_user(client):
    # Enroll first
    with open(SIGNATURE_IMG_1, "rb") as f:
        client.post(
            "/enroll",
            data={"user_id": "delete_me", "modality": "signature", "model": "siamese"},
            files=[("images", ("img.png", f, "image/png"))],
        )

    # Delete
    resp = client.delete("/users/delete_me")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # Verify gone
    resp = client.get("/users")
    ids = [u["user_id"] for u in resp.json()]
    assert "delete_me" not in ids


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_delete_nonexistent_user(client):
    resp = client.delete("/users/ghost_user")
    assert resp.status_code == 404
