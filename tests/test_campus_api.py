"""
Integration tests for the FaceVerify Campus API layer.

The tests patch the model loader with a deterministic fake face engine so the
campus workflow can be checked without loading checkpoint files.
"""

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class FakeFaceEngine:
    def __init__(self, score=0.93):
        self.score = score

    def extract_embedding(self, image_input, validate=True):
        return np.ones(128, dtype=np.float32)

    def verify_against_prototype(self, query_input, enrolled_embeddings, validate=True):
        return {
            "match": self.score > 0.5,
            "score": self.score,
            "threshold": 0.5,
        }

    def compare(self, image1_input, image2_input):
        return {
            "match": True,
            "score": self.score,
            "threshold": 0.5,
            "validation": {},
        }


def image_bytes(seed=1):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    if not HAS_FASTAPI:
        pytest.skip("FastAPI not installed")

    import inference.api as api_module
    from inference.campus_store import CampusStore
    from inference.enrollment_store import EnrollmentStore

    api_module._store = EnrollmentStore(
        store_path=str(tmp_path / "enrollments.json")
    )
    api_module._campus_store = CampusStore(
        store_path=str(tmp_path / "campus_demo.json")
    )
    api_module._campus_store.reset_demo()

    fake_engine = FakeFaceEngine()
    monkeypatch.setattr(
        api_module,
        "_get_engine",
        lambda modality, model_type: fake_engine,
    )

    return TestClient(api_module.app)


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_campus_status_shows_face_models(client):
    resp = client.get("/campus/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["product"] == "FaceVerify Campus"
    assert "siamese" in data["face_models_available"]
    assert "hybrid" in data["face_models_available"]
    assert data["students"] >= 20


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_roster_import_accepts_valid_rows_and_reports_bad_rows(client):
    csv_bytes = (
        b"student_id,name,email\n"
        b"NB-TEST-1,Test Student,test.student@northbridge.edu\n"
        b"NB-BAD-1,,missing-name@northbridge.edu\n"
    )
    resp = client.post(
        "/campus/roster/import",
        data={"course_id": "CS204-2026S"},
        files={"roster": ("roster.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["imported"] == ["NB-TEST-1"]
    assert len(data["rejected"]) == 1


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_malformed_roster_is_rejected(client):
    resp = client.post(
        "/campus/roster/import",
        data={"course_id": "CS204-2026S"},
        files={"roster": ("roster.csv", b"id,full_name\n1,Ada\n", "text/csv")},
    )
    assert resp.status_code == 400
    assert "student_id,name,email" in resp.json()["detail"]


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_student_without_enrollment_cannot_verify(client):
    resp = client.post(
        "/campus/exams/CS204-MIDTERM-1/verify",
        data={"student_id": "NB-2026-1042"},
        files={"image": ("selfie.png", image_bytes(), "image/png")},
    )
    assert resp.status_code == 409
    assert "enrollment" in resp.json()["detail"].lower()


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_enroll_verify_manual_review_and_audit(client):
    enroll_resp = client.post(
        "/campus/students/NB-2026-1042/enroll",
        data={"model_type": "siamese"},
        files=[
            ("images", ("face1.png", image_bytes(1), "image/png")),
            ("images", ("face2.png", image_bytes(2), "image/png")),
        ],
    )
    assert enroll_resp.status_code == 200
    assert enroll_resp.json()["sample_count"] == 2

    exam_resp = client.post(
        "/campus/exams",
        data={
            "exam_id": "CS204-STRICT",
            "course_id": "CS204-2026S",
            "name": "Strict Threshold Demo",
            "start_time": "2026-05-12T10:00",
            "end_time": "2026-05-12T11:30",
            "threshold": "0.99",
            "model_type": "siamese",
        },
    )
    assert exam_resp.status_code == 200

    verify_resp = client.post(
        "/campus/exams/CS204-STRICT/verify",
        data={"student_id": "NB-2026-1042"},
        files={"image": ("selfie.png", image_bytes(3), "image/png")},
    )
    assert verify_resp.status_code == 200
    attempt = verify_resp.json()["attempt"]
    assert attempt["decision"] == "manual_review"
    assert attempt["final_status"] == "Manual Review"

    review_resp = client.post(
        f"/campus/attempts/{attempt['attempt_id']}/review",
        data={
            "reviewer": "Proctor Lee",
            "action": "approve",
            "reason": "Manual ID check completed.",
        },
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["final_status"] == "Approved by Proctor"

    audit_resp = client.get("/campus/audit.csv")
    assert audit_resp.status_code == 200
    assert "manual_review_completed" in audit_resp.text


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_empty_upload_is_rejected(client):
    resp = client.post(
        "/campus/students/NB-2026-1042/enroll",
        data={"model_type": "siamese"},
        files={"images": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400
    assert "empty file" in resp.json()["detail"].lower()


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_hybrid_enrollment_and_verification_path(client):
    exam_resp = client.post(
        "/campus/exams",
        data={
            "exam_id": "CS204-HYBRID",
            "course_id": "CS204-2026S",
            "name": "Hybrid Model Demo",
            "start_time": "2026-05-12T10:00",
            "end_time": "2026-05-12T11:30",
            "threshold": "0.30",
            "model_type": "hybrid",
        },
    )
    assert exam_resp.status_code == 200
    assert exam_resp.json()["model_type"] == "hybrid"

    enroll_resp = client.post(
        "/campus/students/NB-2026-1042/enroll",
        data={"model_type": "hybrid"},
        files=[("images", ("face1.png", image_bytes(10), "image/png"))],
    )
    assert enroll_resp.status_code == 200
    assert enroll_resp.json()["sample_count"] == 1

    verify_resp = client.post(
        "/campus/exams/CS204-HYBRID/verify",
        data={"student_id": "NB-2026-1042"},
        files={"image": ("selfie.png", image_bytes(11), "image/png")},
    )
    assert verify_resp.status_code == 200
    attempt = verify_resp.json()["attempt"]
    assert attempt["model_type"] == "hybrid"
    assert attempt["score"] >= attempt["threshold"]
