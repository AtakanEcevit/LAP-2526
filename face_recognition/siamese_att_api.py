"""
InsightFace Siamese ATT — FastAPI verification server.

Uses the buffalo_l backbone (SCRFD detection + ArcFace w600k_r50) with a
trained SiameseHead projection (512 → 128-d, L2-norm) fine-tuned on AT&T faces.

Threshold: EER-calibrated cosine similarity = 0.58 (vs saf buffalo_l's 0.35).
           Raise for stricter (fewer false accepts), lower for more recall.

Endpoints:
    GET  /health                 system info + calibration stats
    POST /enroll                 register user with 1+ face images
    POST /verify                 verify query image against enrolled user
    POST /compare                compare two images directly (no enrollment)
    GET  /users                  list all enrolled users
    DELETE /users/{user_id}      remove a user

Run:
    uvicorn face_recognition.siamese_att_api:app --host 0.0.0.0 --port 8002 --reload

The model is loaded lazily on the first request.
Enrollments are stored in data/enrollments_siamese_att.json.
"""

import os
import threading
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from face_recognition.insightface_siamese import (
    InsightFaceSiamese,
    NoFaceDetectedError,
    CALIBRATED_THRESHOLD,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINT = os.path.join(
    _PROJECT_ROOT, "results", "insightface_siamese_att", "checkpoints", "best.pth"
)
_ENROLLMENT_PATH = os.path.join(_PROJECT_ROOT, "data", "enrollments_siamese_att.json")
_MODALITY   = "face"
_MODEL_TYPE = "insightface_siamese"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InsightFace Siamese ATT Verification API",
    description=(
        "Face verification using InsightFace buffalo_l backbone + trained "
        "SiameseHead (AT&T fine-tuned). "
        "Cosine similarity threshold calibrated to EER=0.123 on AT&T validation set."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────

_model: InsightFaceSiamese = None
_model_lock = threading.Lock()


def _get_model() -> InsightFaceSiamese:
    """Lazy-load the model (thread-safe double-checked locking)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(_CHECKPOINT):
                    raise RuntimeError(
                        f"Checkpoint not found: {_CHECKPOINT}\n"
                        "Train the model first:\n"
                        "  python -m face_recognition.train_att_siamese"
                    )
                _model = InsightFaceSiamese.from_checkpoint(_CHECKPOINT)
                _model._ensure_if_app()   # pre-load InsightFace detector
    return _model


# Enrollment store — reuse the existing implementation
from inference.enrollment_store import EnrollmentStore

_store = EnrollmentStore(store_path=_ENROLLMENT_PATH)

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 15 MB limit.")
    return data


def _embed_safe(
    model: InsightFaceSiamese, image_bytes: bytes, filename: str, detect: bool = True
) -> np.ndarray:
    """Extract embedding; convert engine errors to HTTP exceptions."""
    try:
        return model.extract_embedding(image_bytes, detect=detect)
    except NoFaceDetectedError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No face detected in '{filename}'. "
                "Upload a clear, front-facing photo with a visible face, "
                "or set detect=false for pre-cropped face images."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    """System status, model info, and calibration details."""
    try:
        model = _get_model()
        info = model.info()
    except Exception as e:
        info = {"error": str(e)}
    return {
        "status": "ok",
        "model": info,
        "calibration": {
            "threshold": CALIBRATED_THRESHOLD,
            "method": "EER on AT&T validation set (val_ratio=0.2, seed=0)",
            "eer": 0.1231,
            "accuracy": 0.8829,
            "genuine_mean_cos": 0.7621,
            "impostor_mean_cos": 0.0578,
        },
        "enrolled_users": len(_store.list_users()),
    }


@app.post("/enroll", summary="Enroll a user")
async def enroll(
    user_id: str = Form(..., description="Unique user identifier (alphanumeric, -, _)"),
    images: List[UploadFile] = File(..., description="One or more face images"),
    threshold: float = Form(
        CALIBRATED_THRESHOLD,
        description="Cosine similarity threshold (default: EER-calibrated 0.58)",
    ),
    detect: bool = Form(
        True,
        description=(
            "Use InsightFace face detection (True = recommended for real photos). "
            "Set False for pre-cropped face images (AT&T, LFW crops, passport photos)."
        ),
    ),
):
    """
    Register a user with one or more face images.

    Re-submitting the same user_id appends additional samples, improving
    prototype quality. For real photos set detect=true (default); for
    pre-cropped face images (e.g. AT&T dataset) set detect=false.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    model = _get_model()
    enrolled_count = 0

    for img_file in images:
        data = await _read_upload(img_file)
        emb = _embed_safe(model, data, img_file.filename or "image", detect=detect)
        try:
            result = _store.enroll(user_id, _MODALITY, _MODEL_TYPE, emb)
            enrolled_count = result["sample_count"]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {
        "user_id": user_id,
        "sample_count": enrolled_count,
        "message": f"Enrolled {len(images)} image(s) for user '{user_id}'.",
    }


@app.post("/verify", summary="Verify a query image against an enrolled user")
async def verify(
    user_id: str = Form(..., description="Enrolled user ID"),
    image: UploadFile = File(..., description="Query face image"),
    threshold: float = Form(
        CALIBRATED_THRESHOLD,
        description="Cosine similarity threshold (default: EER-calibrated 0.58)",
    ),
    detect: bool = Form(
        True,
        description="Use InsightFace face detection. Set False for pre-cropped images.",
    ),
):
    """
    Verify whether the uploaded face belongs to an enrolled user.

    Returns a match decision, cosine similarity score, and the threshold used.
    Raise threshold for stricter verification (fewer false accepts, more false rejects).
    """
    user_info = _store.get_user(user_id)
    if user_info is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' is not enrolled.")

    if user_info.get("model_type") != _MODEL_TYPE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"User '{user_id}' was enrolled with model '{user_info['model_type']}', "
                f"not '{_MODEL_TYPE}'."
            ),
        )

    model = _get_model()
    data = await _read_upload(image)
    query_emb = _embed_safe(model, data, image.filename or "query", detect=detect)

    enrolled_embs = _store.get_embeddings(user_id)  # (N, 128)
    result = model.verify(query_emb, enrolled_embs, threshold=threshold, detect=False)

    return {
        "user_id": user_id,
        "match": result["match"],
        "score": result["score"],
        "threshold": result["threshold"],
        "enrolled_samples": int(enrolled_embs.shape[0]),
    }


@app.post("/compare", summary="Compare two face images directly")
async def compare(
    image1: UploadFile = File(..., description="First face image"),
    image2: UploadFile = File(..., description="Second face image"),
    threshold: float = Form(
        CALIBRATED_THRESHOLD,
        description="Cosine similarity threshold (default: EER-calibrated 0.58)",
    ),
    detect: bool = Form(
        True,
        description="Use InsightFace face detection. Set False for pre-cropped images.",
    ),
):
    """
    Compare two face images without enrollment.

    Returns a match decision and cosine similarity score.
    """
    model = _get_model()
    data1 = await _read_upload(image1)
    data2 = await _read_upload(image2)

    emb1 = _embed_safe(model, data1, image1.filename or "image1", detect=detect)
    emb2 = _embed_safe(model, data2, image2.filename or "image2", detect=detect)

    result = model.verify(emb1, np.array([emb2]), threshold=threshold, detect=False)

    return {
        "match": result["match"],
        "score": result["score"],
        "threshold": threshold,
    }


@app.get("/users", summary="List enrolled users")
async def list_users():
    """Return summary info for all enrolled users."""
    return _store.list_users()


@app.delete("/users/{user_id}", summary="Delete an enrolled user")
async def delete_user(user_id: str):
    """Remove a user and all their enrolled embeddings."""
    deleted = _store.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    return {"deleted": True, "user_id": user_id}
