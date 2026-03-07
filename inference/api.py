"""
FastAPI REST server for biometric verification.

Exposes the inference engine via HTTP endpoints with support for
enrollment, verification, and direct pair comparison.
Also serves the web UI from the /static path.

Usage:
    uvicorn inference.api:app --host 127.0.0.1 --port 8000
    # Then open http://127.0.0.1:8000 for the web UI
    # or http://127.0.0.1:8000/docs for the Swagger API docs
"""

import os
import threading
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference.engine import VerificationEngine
from inference.enrollment_store import EnrollmentStore
from inference.config import (
    MODEL_REGISTRY, VALID_MODALITIES, VALID_MODEL_TYPES,
    DEFAULT_ENROLLMENT_PATH,
)
from inference.validation import validate_image


# ── App Setup ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biometric Few-Shot Verification API",
    description=(
        "REST API for biometric verification using Siamese and "
        "Prototypical Networks. Supports signature, face, and "
        "fingerprint modalities."
    ),
    version="1.1.0",
)

# CORS — allow the web UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web UI static files (css/js subdirs at root-relative paths)
_UI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"
)
if os.path.isdir(_UI_DIR):
    _CSS_DIR = os.path.join(_UI_DIR, "css")
    _JS_DIR = os.path.join(_UI_DIR, "js")
    if os.path.isdir(_CSS_DIR):
        app.mount("/css", StaticFiles(directory=_CSS_DIR), name="css")
    if os.path.isdir(_JS_DIR):
        app.mount("/js", StaticFiles(directory=_JS_DIR), name="js")

# Shared enrollment store
_store = EnrollmentStore()

# Lazy-loaded engine cache: (modality, model_type) -> VerificationEngine
_engines = {}
_engine_locks = {}

# Create a lock per model slot to prevent double-loading
for _key in MODEL_REGISTRY:
    _engine_locks[_key] = threading.Lock()


def _get_engine(modality: str, model_type: str) -> VerificationEngine:
    """Get or lazily load an engine for the given modality and model_type."""
    key = (modality, model_type)

    if key not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"No model registered for {modality}/{model_type}. "
                   f"Valid modalities: {sorted(VALID_MODALITIES)}, "
                   f"Valid models: {sorted(VALID_MODEL_TYPES)}",
        )

    if key not in _engines:
        with _engine_locks[key]:
            # Double-check after acquiring lock
            if key not in _engines:
                engine = VerificationEngine()
                engine.load(modality, model_type)
                _engines[key] = engine

    return _engines[key]


async def _read_image_bytes(file: UploadFile) -> bytes:
    """Read and validate uploaded image file."""
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")
    return contents


def _validate_modality(modality: str):
    if modality not in VALID_MODALITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modality '{modality}'. "
                   f"Must be one of: {sorted(VALID_MODALITIES)}",
        )


def _validate_model_type(model_type: str):
    if model_type not in VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type '{model_type}'. "
                   f"Must be one of: {sorted(VALID_MODEL_TYPES)}",
        )


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check and system info."""
    return {
        "status": "ok",
        "models_available": [
            f"{m}/{t}" for (m, t) in MODEL_REGISTRY.keys()
        ],
        "models_loaded": [
            f"{m}/{t}" for (m, t) in _engines.keys()
        ],
        "enrollment_count": len(_store.list_users()),
    }


@app.post("/enroll")
async def enroll(
    user_id: str = Form(..., description="Unique user identifier"),
    modality: str = Form(..., description="signature, face, or fingerprint"),
    model: str = Form(..., description="siamese or prototypical"),
    images: List[UploadFile] = File(..., description="Reference image(s)"),
):
    """
    Enroll a user with one or more reference images.

    The embeddings are extracted and stored. Additional images can be
    added later by calling this endpoint again with the same user_id.
    """
    _validate_modality(modality)
    _validate_model_type(model)

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="At least one image required.")

    engine = _get_engine(modality, model)

    total_enrolled = 0
    all_warnings = []
    for img_file in images:
        image_bytes = await _read_image_bytes(img_file)

        # Validate image before processing
        val = validate_image(image_bytes, modality)
        if not val.passed:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image '{img_file.filename}' failed validation: "
                    f"{'; '.join(val.warnings)}"
                ),
            )
        if val.warnings:
            all_warnings.extend(
                f"{img_file.filename}: {w}" for w in val.warnings
            )

        try:
            embedding = engine.extract_embedding(image_bytes, validate=False)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process image '{img_file.filename}': {str(e)}",
            )

        try:
            result = _store.enroll(user_id, modality, model, embedding)
            total_enrolled = result["sample_count"]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {
        "user_id": user_id,
        "sample_count": total_enrolled,
        "message": f"Enrolled {len(images)} sample(s) for user '{user_id}'.",
        "validation_warnings": all_warnings,
    }


@app.post("/verify")
async def verify(
    user_id: str = Form(..., description="Enrolled user ID to verify against"),
    image: UploadFile = File(..., description="Query image to verify"),
):
    """
    Verify a query image against an enrolled user's prototype.

    The user must have been previously enrolled via /enroll.
    """
    # Look up user
    user_info = _store.get_user(user_id)
    if user_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' is not enrolled.",
        )

    modality = user_info["modality"]
    model_type = user_info["model_type"]

    engine = _get_engine(modality, model_type)
    image_bytes = await _read_image_bytes(image)

    try:
        enrolled_embeddings = _store.get_embeddings(user_id)
        result = engine.verify_against_prototype(image_bytes, enrolled_embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

    return {
        "user_id": user_id,
        "match": result["match"],
        "score": result["score"],
        "threshold": result["threshold"],
        "validation": result.get("validation"),
    }


@app.post("/compare")
async def compare(
    modality: str = Form(..., description="signature, face, or fingerprint"),
    model: str = Form(..., description="siamese or prototypical"),
    image1: UploadFile = File(..., description="First image"),
    image2: UploadFile = File(..., description="Second image"),
):
    """
    Compare two images directly without enrollment.

    Returns a similarity score and match decision.
    """
    _validate_modality(modality)
    _validate_model_type(model)

    engine = _get_engine(modality, model)

    bytes1 = await _read_image_bytes(image1)
    bytes2 = await _read_image_bytes(image2)

    try:
        result = engine.compare(bytes1, bytes2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

    return {
        "match": result["match"],
        "score": result["score"],
        "threshold": result["threshold"],
        "validation": result.get("validation"),
    }


@app.get("/users")
async def list_users():
    """List all enrolled users."""
    return _store.list_users()


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Delete an enrolled user and their embeddings."""
    deleted = _store.delete_user(user_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found.",
        )
    return {"deleted": True, "user_id": user_id}


# ── Root: serve the UI ───────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI."""
    index_path = os.path.join(_UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "API is running. Visit /docs for Swagger UI."})
