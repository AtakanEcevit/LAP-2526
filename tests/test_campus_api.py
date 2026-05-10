"""
Integration tests for the FaceVerify Campus API layer.

The tests patch the model loader with a deterministic fake face engine so the
campus workflow can be checked without loading checkpoint files.
"""

import io
import json
import os
import sys
import zipfile
from pathlib import Path

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
    api_module.DEFAULT_FLUX_TEST_EXPORT_DIR = (
        tmp_path / "flux_test_uploads" / "current"
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
    assert "facenet_proto" in data["face_models_available"]
    assert "facenet_contrastive_proto" in data["face_models_available"]
    assert data["face_model_defaults"]["hybrid"]["threshold"] == pytest.approx(0.3000000119)
    assert data["face_model_defaults"]["facenet_proto"]["threshold"] == pytest.approx(0.47)
    assert data["face_model_defaults"]["facenet_contrastive_proto"]["threshold"] == pytest.approx(0.800884)
    assert data["students"] >= 20
    default_exam = client.get("/campus").json()["exams"][0]
    assert default_exam["model_type"] == "facenet_contrastive_proto"
    assert default_exam["threshold"] == pytest.approx(0.800884)


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_flux_status_reports_present_and_missing_dataset(client, tmp_path):
    flux_dir = tmp_path / "FLUXSynID" / "FLUXSynID" / "FLUXSynID"
    _write_flux_identity(flux_dir, "flux_001", age="20-29")

    resp = client.get(
        "/campus/flux/status",
        params={"dataset_dir": str(tmp_path / "FLUXSynID")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["eligible_identity_count"] == 1
    assert data["normalized_path"].endswith(os.path.join("FLUXSynID", "FLUXSynID"))

    missing = client.get(
        "/campus/flux/status",
        params={"dataset_dir": str(tmp_path / "missing")},
    )
    assert missing.status_code == 200
    assert missing.json()["available"] is False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_flux_preupload_enrolls_students_idempotently(client, tmp_path):
    flux_dir = tmp_path / "flux"
    for idx in range(4):
        _write_flux_identity(flux_dir, f"flux_{idx:03d}", age="20-29")

    for _ in range(2):
        resp = client.post(
            "/campus/flux/preupload",
            data={
                "dataset_dir": str(flux_dir),
                "count": "3",
                "seed": "42",
                "model_type": "facenet_contrastive_proto",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["imported_count"] == 3
        assert data["skipped"] == []
        assert data["export"]["image_count"] == 3
        assert data["export"]["skipped"] == []

    roster = client.get("/campus/exams/CS204-MIDTERM-1/roster").json()["roster"]
    preuploaded = [row for row in roster if row.get("face_source") == "flux_synid"]
    assert len(preuploaded) == 3
    assert all(row["sample_count"] == 3 for row in preuploaded)
    assert all(row["reference_preview"].startswith("data:image/jpeg;base64,") for row in preuploaded)

    users = client.get("/users").json()
    campus_users = [user for user in users if user["user_id"].startswith("campus_")]
    assert len(campus_users) == 3
    assert all(user["sample_count"] == 3 for user in campus_users)

    test_set = client.get("/campus/flux/test-set")
    assert test_set.status_code == 200
    test_data = test_set.json()
    assert test_data["image_count"] == 3
    first_entry = test_data["manifest"][0]
    assert Path(first_entry["exported_path"]).is_file()

    per_student = client.get(
        f"/campus/students/{first_entry['student_id']}/flux/test-image"
    )
    assert per_student.status_code == 200
    assert per_student.content == Path(first_entry["exported_path"]).read_bytes()

    zip_resp = client.get("/campus/flux/test-set.zip")
    assert zip_resp.status_code == 200
    zip_path = tmp_path / "flux_test_set.zip"
    zip_path.write_bytes(zip_resp.content)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "manifest.json" in names
    assert first_entry["exported_filename"] in names

    verify = client.post(
        "/campus/exams/CS204-MIDTERM-1/verify",
        data={"student_id": first_entry["student_id"]},
        files={
            "image": (
                first_entry["exported_filename"],
                Path(first_entry["exported_path"]).read_bytes(),
                "image/jpeg",
            )
        },
    )
    assert verify.status_code == 200
    assert verify.json()["attempt"]["model_type"] == "facenet_contrastive_proto"
    assert verify.json()["attempt"]["attempt_source"] == "upload"


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_failed_flux_rerun_preserves_existing_enrollment(client, tmp_path):
    good_flux = tmp_path / "good"
    bad_flux = tmp_path / "bad"
    _write_flux_identity(good_flux, "flux_good", age="20-29")
    _write_tiny_flux_identity(bad_flux, "flux_bad", age="20-29")

    good = client.post(
        "/campus/flux/preupload",
        data={
            "dataset_dir": str(good_flux),
            "count": "1",
            "seed": "1",
            "model_type": "facenet_contrastive_proto",
        },
    )
    assert good.status_code == 200
    student_id = good.json()["imported"][0]["student_id"]

    bad = client.post(
        "/campus/flux/preupload",
        data={
            "dataset_dir": str(bad_flux),
            "count": "1",
            "seed": "1",
            "model_type": "facenet_contrastive_proto",
        },
    )
    assert bad.status_code == 200
    assert bad.json()["imported_count"] == 0
    assert len(bad.json()["skipped"]) == 1

    users = client.get("/users").json()
    user = next(item for item in users if item["user_id"] == f"campus_{student_id}")
    assert user["sample_count"] == 3

    verify = client.post(
        "/campus/exams/CS204-MIDTERM-1/verify-preloaded",
        data={"student_id": student_id, "scenario": "matching", "confirmed": "true"},
    )
    assert verify.status_code == 200


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_verify_preloaded_matching_and_model_mismatch(client, tmp_path):
    flux_dir = tmp_path / "flux"
    for idx in range(2):
        _write_flux_identity(flux_dir, f"flux_{idx:03d}", age="20-29")

    preupload = client.post(
        "/campus/flux/preupload",
        data={
            "dataset_dir": str(flux_dir),
            "count": "2",
            "seed": "7",
            "model_type": "facenet_contrastive_proto",
        },
    )
    assert preupload.status_code == 200
    student_id = preupload.json()["imported"][0]["student_id"]

    unconfirmed = client.post(
        "/campus/exams/CS204-MIDTERM-1/verify-preloaded",
        data={"student_id": student_id, "scenario": "matching"},
    )
    assert unconfirmed.status_code == 400
    repeated_unconfirmed = client.post(
        "/campus/exams/CS204-MIDTERM-1/verify-preloaded",
        data={"student_id": student_id, "scenario": "matching", "confirmed": "false"},
    )
    assert repeated_unconfirmed.status_code == 400
    assert client.get(
        "/campus/attempts",
        params={"exam_id": "CS204-MIDTERM-1", "student_id": student_id},
    ).json() == []

    verify = client.post(
        "/campus/exams/CS204-MIDTERM-1/verify-preloaded",
        data={"student_id": student_id, "scenario": "matching", "confirmed": "true"},
    )
    assert verify.status_code == 200
    attempt = verify.json()["attempt"]
    assert attempt["model_type"] == "facenet_contrastive_proto"
    assert attempt["attempt_source"] == "preloaded_demo"
    assert attempt["scenario"] == "matching"
    assert attempt["query_preview"].startswith("data:image/jpeg;base64,")

    client.post(
        "/campus/exams",
        data={
            "exam_id": "CS204-SIAMESE",
            "course_id": "CS204-2026S",
            "name": "Model Mismatch",
            "start_time": "2026-05-12T10:00",
            "end_time": "2026-05-12T11:30",
            "threshold": "0.65",
            "model_type": "siamese",
        },
    )
    mismatch = client.post(
        "/campus/exams/CS204-SIAMESE/verify-preloaded",
        data={"student_id": student_id, "scenario": "matching", "confirmed": "true"},
    )
    assert mismatch.status_code == 409
    assert "requires siamese" in mismatch.json()["detail"]


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_flux_export_reports_missing_source_and_non_flux_download(client, tmp_path):
    flux_dir = tmp_path / "flux"
    _write_flux_identity(flux_dir, "flux_001", age="20-29")

    preupload = client.post(
        "/campus/flux/preupload",
        data={
            "dataset_dir": str(flux_dir),
            "count": "1",
            "seed": "7",
            "model_type": "facenet_contrastive_proto",
        },
    )
    assert preupload.status_code == 200
    student_id = preupload.json()["imported"][0]["student_id"]

    student = next(
        item for item in client.get("/campus/students").json()
        if item["student_id"] == student_id
    )
    Path(student["face_query_image"]).unlink()

    export = client.post("/campus/flux/export-test-set")
    assert export.status_code == 200
    data = export.json()
    assert data["image_count"] == 0
    assert data["skipped"][0]["student_id"] == student_id
    assert "missing" in data["skipped"][0]["reason"]

    missing_download = client.get(f"/campus/students/{student_id}/flux/test-image")
    assert missing_download.status_code == 404

    non_flux_download = client.get("/campus/students/NB-2026-1043/flux/test-image")
    assert non_flux_download.status_code == 409


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
def test_review_actions_are_restricted_to_reviewable_attempts(client):
    enroll_resp = client.post(
        "/campus/students/NB-2026-1042/enroll",
        data={"model_type": "siamese"},
        files=[
            ("images", ("face1.png", image_bytes(1), "image/png")),
            ("images", ("face2.png", image_bytes(2), "image/png")),
        ],
    )
    assert enroll_resp.status_code == 200

    verified_exam = client.post(
        "/campus/exams",
        data={
            "exam_id": "CS204-VERIFIED-REVIEW-GUARD",
            "course_id": "CS204-2026S",
            "name": "Verified Review Guard",
            "start_time": "2026-05-12T10:00",
            "end_time": "2026-05-12T11:30",
            "threshold": "0.50",
            "model_type": "siamese",
        },
    )
    assert verified_exam.status_code == 200
    verified_resp = client.post(
        "/campus/exams/CS204-VERIFIED-REVIEW-GUARD/verify",
        data={"student_id": "NB-2026-1042"},
        files={"image": ("selfie.png", image_bytes(3), "image/png")},
    )
    assert verified_resp.status_code == 200
    verified_attempt = verified_resp.json()["attempt"]
    assert verified_attempt["final_status"] == "Verified"

    forbidden = client.post(
        f"/campus/attempts/{verified_attempt['attempt_id']}/review",
        data={"reviewer": "Proctor Lee", "action": "deny", "reason": "Should fail."},
    )
    assert forbidden.status_code == 400
    assert "not eligible" in forbidden.json()["detail"]

    strict_exam = client.post(
        "/campus/exams",
        data={
            "exam_id": "CS204-REVIEW-GUARD",
            "course_id": "CS204-2026S",
            "name": "Review Guard",
            "start_time": "2026-05-12T10:00",
            "end_time": "2026-05-12T11:30",
            "threshold": "0.99",
            "model_type": "siamese",
        },
    )
    assert strict_exam.status_code == 200
    manual_resp = client.post(
        "/campus/exams/CS204-REVIEW-GUARD/verify",
        data={"student_id": "NB-2026-1042"},
        files={"image": ("selfie.png", image_bytes(4), "image/png")},
    )
    assert manual_resp.status_code == 200
    manual_attempt = manual_resp.json()["attempt"]
    assert manual_attempt["final_status"] == "Manual Review"

    approve_resp = client.post(
        f"/campus/attempts/{manual_attempt['attempt_id']}/review",
        data={
            "reviewer": "Proctor Lee",
            "action": "approve",
            "reason": "Manual ID check completed.",
        },
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["final_status"] == "Approved by Proctor"

    approved_retry = client.post(
        f"/campus/attempts/{manual_attempt['attempt_id']}/review",
        data={"reviewer": "Proctor Lee", "action": "deny", "reason": "Should fail."},
    )
    assert approved_retry.status_code == 400
    assert "not eligible" in approved_retry.json()["detail"]

    fallback_resp = client.post(
        "/campus/exams/CS204-REVIEW-GUARD/verify",
        data={"student_id": "NB-2026-1042"},
        files={"image": ("selfie.png", image_bytes(5), "image/png")},
    )
    assert fallback_resp.status_code == 200
    fallback_attempt = fallback_resp.json()["attempt"]
    fallback_mark = client.post(
        f"/campus/attempts/{fallback_attempt['attempt_id']}/review",
        data={"reviewer": "Proctor Lee", "action": "fallback", "reason": "Needs ID desk."},
    )
    assert fallback_mark.status_code == 200
    assert fallback_mark.json()["final_status"] == "Fallback Requested"

    fallback_approve = client.post(
        f"/campus/attempts/{fallback_attempt['attempt_id']}/review",
        data={
            "reviewer": "Proctor Lee",
            "action": "approve",
            "reason": "Fallback ID check completed.",
        },
    )
    assert fallback_approve.status_code == 200
    assert fallback_approve.json()["final_status"] == "Approved by Proctor"


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
    assert attempt["attempt_source"] == "upload"


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_facenet_proto_enrollment_verification_flux_and_lab_paths(client, tmp_path):
    exam_resp = client.post(
        "/campus/exams",
        data={
            "exam_id": "CS204-FACENET-PROTO",
            "course_id": "CS204-2026S",
            "name": "FaceNet Proto Demo",
            "start_time": "2026-05-12T10:00",
            "end_time": "2026-05-12T11:30",
            "threshold": "0.47",
            "model_type": "facenet_proto",
        },
    )
    assert exam_resp.status_code == 200
    assert exam_resp.json()["model_type"] == "facenet_proto"

    enroll_resp = client.post(
        "/campus/students/NB-2026-1042/enroll",
        data={"model_type": "facenet_proto"},
        files=[("images", ("face1.png", image_bytes(20), "image/png"))],
    )
    assert enroll_resp.status_code == 200
    assert enroll_resp.json()["sample_count"] == 1

    verify_resp = client.post(
        "/campus/exams/CS204-FACENET-PROTO/verify",
        data={"student_id": "NB-2026-1042"},
        files={"image": ("selfie.png", image_bytes(21), "image/png")},
    )
    assert verify_resp.status_code == 200
    attempt = verify_resp.json()["attempt"]
    assert attempt["model_type"] == "facenet_proto"
    assert attempt["threshold"] == pytest.approx(0.47)
    assert attempt["score"] >= attempt["threshold"]

    flux_dir = tmp_path / "facenet_proto_flux"
    _write_flux_identity(flux_dir, "flux_facenet", age="20-29")
    preupload = client.post(
        "/campus/flux/preupload",
        data={
            "dataset_dir": str(flux_dir),
            "count": "1",
            "seed": "3",
            "model_type": "facenet_proto",
        },
    )
    assert preupload.status_code == 200
    assert preupload.json()["imported_count"] == 1
    assert preupload.json()["imported"][0]["sample_count"] == 3

    users = client.get("/users").json()
    assert any(
        user["model_type"] == "facenet_proto"
        for user in users
        if user["user_id"].startswith("campus_")
    )

    lab_resp = client.post(
        "/campus/model-lab/compare",
        data={"model_type": "facenet_proto"},
        files=[
            ("image1", ("face-a.png", image_bytes(30), "image/png")),
            ("image2", ("face-b.png", image_bytes(31), "image/png")),
        ],
    )
    assert lab_resp.status_code == 200
    assert lab_resp.json()["match"] is True


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_facenet_contrastive_proto_enrollment_verification_flux_and_lab_paths(
    client,
    tmp_path,
):
    exam_resp = client.post(
        "/campus/exams",
        data={
            "exam_id": "CS204-FACENET-CONTRASTIVE-PROTO",
            "course_id": "CS204-2026S",
            "name": "FaceNet Contrastive Proto Demo",
            "start_time": "2026-05-12T10:00",
            "end_time": "2026-05-12T11:30",
            "threshold": "0.800884",
            "model_type": "facenet_contrastive_proto",
        },
    )
    assert exam_resp.status_code == 200
    assert exam_resp.json()["model_type"] == "facenet_contrastive_proto"
    assert exam_resp.json()["threshold"] == pytest.approx(0.800884)

    enroll_resp = client.post(
        "/campus/students/NB-2026-1042/enroll",
        data={"model_type": "facenet_contrastive_proto"},
        files=[("images", ("face1.png", image_bytes(40), "image/png"))],
    )
    assert enroll_resp.status_code == 200
    assert enroll_resp.json()["sample_count"] == 1

    verify_resp = client.post(
        "/campus/exams/CS204-FACENET-CONTRASTIVE-PROTO/verify",
        data={"student_id": "NB-2026-1042"},
        files={"image": ("selfie.png", image_bytes(41), "image/png")},
    )
    assert verify_resp.status_code == 200
    attempt = verify_resp.json()["attempt"]
    assert attempt["model_type"] == "facenet_contrastive_proto"
    assert attempt["threshold"] == pytest.approx(0.800884)
    assert attempt["score"] >= attempt["threshold"]

    flux_dir = tmp_path / "facenet_contrastive_proto_flux"
    _write_flux_identity(flux_dir, "flux_facenet_contrastive", age="20-29")
    preupload = client.post(
        "/campus/flux/preupload",
        data={
            "dataset_dir": str(flux_dir),
            "count": "1",
            "seed": "4",
            "model_type": "facenet_contrastive_proto",
        },
    )
    assert preupload.status_code == 200
    assert preupload.json()["imported_count"] == 1
    assert preupload.json()["imported"][0]["sample_count"] == 3

    users = client.get("/users").json()
    assert any(
        user["model_type"] == "facenet_contrastive_proto"
        for user in users
        if user["user_id"].startswith("campus_")
    )

    lab_resp = client.post(
        "/campus/model-lab/compare",
        data={"model_type": "facenet_contrastive_proto"},
        files=[
            ("image1", ("face-a.png", image_bytes(50), "image/png")),
            ("image2", ("face-b.png", image_bytes(51), "image/png")),
        ],
    )
    assert lab_resp.status_code == 200
    assert lab_resp.json()["match"] is True


def _write_flux_identity(root, identity, age="20-29"):
    identity_dir = root / identity
    identity_dir.mkdir(parents=True, exist_ok=True)
    (identity_dir / f"{identity}_f_doc.jpg").write_bytes(image_bytes(101))
    (identity_dir / f"{identity}_f_live_0_a_d1.jpg").write_bytes(image_bytes(102))
    (identity_dir / f"{identity}_f_live_0_e_d1.jpg").write_bytes(image_bytes(103))
    (identity_dir / f"{identity}_f_live_0_p_d1.jpg").write_bytes(image_bytes(104))
    (identity_dir / f"{identity}_f.json").write_text(
        json.dumps({"attributes": {"ages.txt": age}}),
        encoding="utf-8",
    )


def _write_tiny_flux_identity(root, identity, age="20-29"):
    identity_dir = root / identity
    identity_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("doc", "live_0_a_d1", "live_0_e_d1", "live_0_p_d1"):
        out = io.BytesIO()
        Image.new("RGB", (8, 8), (120, 120, 120)).save(out, format="JPEG")
        (identity_dir / f"{identity}_f_{suffix}.jpg").write_bytes(out.getvalue())
    (identity_dir / f"{identity}_f.json").write_text(
        json.dumps({"attributes": {"ages.txt": age}}),
        encoding="utf-8",
    )
