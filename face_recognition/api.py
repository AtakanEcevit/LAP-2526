"""
InsightFace ArcFace — FastAPI verification server.

Endpoints:
    GET  /health              system info
    POST /enroll              register user with 1+ face images
    POST /verify              verify query image against enrolled user
    POST /compare             compare two images directly (no enrollment)
    GET  /users               list all enrolled users
    DELETE /users/{user_id}   remove a user

Run:
    uvicorn face_recognition.api:app --host 0.0.0.0 --port 8001 --reload

The engine is lazily loaded on the first request.
Enrollments are stored in data/enrollments_insightface.json.
"""

import os
import threading
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from face_recognition.engine import InsightFaceEngine, NoFaceDetectedError, DEFAULT_THRESHOLD
from inference.enrollment_store import EnrollmentStore

# ── Constants ─────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENROLLMENT_PATH = os.path.join(_PROJECT_ROOT, "data", "enrollments_insightface.json")
_MODEL_TYPE = "insightface"
_MODALITY = "face"

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InsightFace ArcFace Verification API",
    description=(
        "Face verification using InsightFace buffalo_l (SCRFD detection + "
        "ArcFace w600k_r50). 512-d L2-normalised embeddings, cosine similarity."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────

_engine = InsightFaceEngine()
_engine_lock = threading.Lock()
_store = EnrollmentStore(store_path=_ENROLLMENT_PATH)


def _get_engine() -> InsightFaceEngine:
    """Ensure model is loaded (thread-safe, lazy)."""
    if _engine._app is None:
        with _engine_lock:
            if _engine._app is None:
                _engine._load()
    return _engine


# ── Helpers ───────────────────────────────────────────────────────────────

async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 15 MB limit.")
    return data


def _extract_safe(engine: InsightFaceEngine, image_bytes: bytes, filename: str) -> np.ndarray:
    """Extract embedding; convert engine errors to HTTP exceptions."""
    try:
        return engine.extract_embedding(image_bytes)
    except NoFaceDetectedError:
        raise HTTPException(
            status_code=422,
            detail=f"No face detected in '{filename}'. "
                   "Please upload a clear, front-facing photo.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    """Returns system status and model info."""
    return {
        "status": "ok",
        "engine": _engine.info(),
        "enrolled_users": len(_store.list_users()),
    }


@app.post("/enroll", summary="Enroll a user")
async def enroll(
    user_id: str = Form(..., description="Unique user identifier (alphanumeric, -, _)"),
    images: List[UploadFile] = File(..., description="One or more face images"),
    threshold: float = Form(DEFAULT_THRESHOLD, description="Cosine similarity threshold"),
):
    """
    Register a user with one or more face images.

    Each image must contain exactly one detectable face.
    Re-submitting with the same `user_id` appends new samples.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    engine = _get_engine()
    enrolled_count = 0

    for img_file in images:
        data = await _read_upload(img_file)
        emb = _extract_safe(engine, data, img_file.filename or "image")

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
    threshold: float = Form(DEFAULT_THRESHOLD, description="Cosine similarity threshold"),
):
    """
    Verify whether the uploaded face belongs to an enrolled user.

    Returns a match decision, cosine similarity score, and the threshold used.
    """
    user_info = _store.get_user(user_id)
    if user_info is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' is not enrolled.")

    if user_info.get("model_type") != _MODEL_TYPE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"User '{user_id}' was enrolled with model_type "
                f"'{user_info['model_type']}', not '{_MODEL_TYPE}'."
            ),
        )

    engine = _get_engine()
    data = await _read_upload(image)
    query_emb = _extract_safe(engine, data, image.filename or "query")

    enrolled_embs = _store.get_embeddings(user_id)  # (N, 512)

    try:
        result = engine.verify(query_emb, enrolled_embs, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {e}")

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
    threshold: float = Form(DEFAULT_THRESHOLD, description="Cosine similarity threshold"),
):
    """
    Compare two images without enrollment.

    Returns a match decision and cosine similarity score.
    """
    engine = _get_engine()
    data1 = await _read_upload(image1)
    data2 = await _read_upload(image2)

    emb1 = _extract_safe(engine, data1, image1.filename or "image1")
    emb2 = _extract_safe(engine, data2, image2.filename or "image2")

    try:
        result = engine.compare(emb1, emb2, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {e}")

    return {
        "match": result["match"],
        "score": result["score"],
        "threshold": result["threshold"],
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
