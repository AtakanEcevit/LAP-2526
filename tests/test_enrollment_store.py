"""
Tests for inference/enrollment_store.py

Uses a temporary file for each test so nothing touches the real
data/enrollments.json.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.enrollment_store import EnrollmentStore


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Create a fresh EnrollmentStore using a temp directory."""
    path = os.path.join(str(tmp_path), "test_enrollments.json")
    return EnrollmentStore(store_path=path)


@pytest.fixture
def dummy_embedding():
    """A random 128-d normalized embedding."""
    emb = np.random.randn(128).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb


# ── Enrollment ──────────────────────────────────────────────────────────

def test_enroll_new_user(store, dummy_embedding):
    result = store.enroll("alice", "signature", "siamese", dummy_embedding)
    assert result["user_id"] == "alice"
    assert result["sample_count"] == 1


def test_enroll_multiple_samples(store, dummy_embedding):
    store.enroll("bob", "face", "prototypical", dummy_embedding)
    emb2 = np.random.randn(128).astype(np.float32)
    result = store.enroll("bob", "face", "prototypical", emb2)
    assert result["sample_count"] == 2


def test_enroll_modality_mismatch(store, dummy_embedding):
    store.enroll("carol", "signature", "siamese", dummy_embedding)
    with pytest.raises(ValueError, match="modality"):
        store.enroll("carol", "face", "siamese", dummy_embedding)


def test_enroll_model_type_mismatch(store, dummy_embedding):
    store.enroll("dave", "signature", "siamese", dummy_embedding)
    with pytest.raises(ValueError, match="model_type"):
        store.enroll("dave", "signature", "prototypical", dummy_embedding)


# ── Prototype ───────────────────────────────────────────────────────────

def test_get_prototype(store):
    emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 125, dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0] + [0.0] * 125, dtype=np.float32)
    store.enroll("eve", "signature", "siamese", emb1)
    store.enroll("eve", "signature", "siamese", emb2)

    proto = store.get_prototype("eve")
    expected = np.mean([emb1, emb2], axis=0)
    np.testing.assert_allclose(proto, expected, atol=1e-6)


def test_get_prototype_not_enrolled(store):
    with pytest.raises(KeyError, match="not enrolled"):
        store.get_prototype("nobody")


# ── Embeddings ──────────────────────────────────────────────────────────

def test_get_embeddings(store, dummy_embedding):
    store.enroll("frank", "fingerprint", "siamese", dummy_embedding)
    embs = store.get_embeddings("frank")
    assert embs.shape == (1, 128)
    np.testing.assert_allclose(embs[0], dummy_embedding, atol=1e-6)


# ── List / Delete ───────────────────────────────────────────────────────

def test_list_users(store, dummy_embedding):
    store.enroll("user_a", "signature", "siamese", dummy_embedding)
    store.enroll("user_b", "face", "prototypical", dummy_embedding)
    users = store.list_users()
    ids = [u["user_id"] for u in users]
    assert "user_a" in ids
    assert "user_b" in ids


def test_delete_user(store, dummy_embedding):
    store.enroll("temp_user", "signature", "siamese", dummy_embedding)
    assert store.delete_user("temp_user") is True
    assert store.get_user("temp_user") is None


def test_delete_nonexistent_user(store):
    assert store.delete_user("ghost") is False


# ── Persistence ─────────────────────────────────────────────────────────

def test_persistence(tmp_path, dummy_embedding):
    path = os.path.join(str(tmp_path), "persist_test.json")
    store1 = EnrollmentStore(store_path=path)
    store1.enroll("persist_user", "signature", "siamese", dummy_embedding)

    # Create new instance with same path
    store2 = EnrollmentStore(store_path=path)
    user = store2.get_user("persist_user")
    assert user is not None
    assert user["sample_count"] == 1


# ── Clear ───────────────────────────────────────────────────────────────

def test_clear(store, dummy_embedding):
    store.enroll("cleared", "signature", "siamese", dummy_embedding)
    store.clear()
    assert len(store.list_users()) == 0


# ── User ID Validation ──────────────────────────────────────────────────

def test_invalid_user_id_empty(store, dummy_embedding):
    with pytest.raises(ValueError, match="non-empty"):
        store.enroll("", "signature", "siamese", dummy_embedding)


def test_invalid_user_id_special_chars(store, dummy_embedding):
    with pytest.raises(ValueError, match="letters, numbers"):
        store.enroll("bad user!", "signature", "siamese", dummy_embedding)


def test_valid_user_id_with_dots_and_hyphens(store, dummy_embedding):
    result = store.enroll("user.name-1", "signature", "siamese", dummy_embedding)
    assert result["user_id"] == "user.name-1"
