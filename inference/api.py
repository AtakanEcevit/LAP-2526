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

import base64
import io
import os
import threading
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference.campus_store import CampusStore
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
    version="1.4.0",
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
    _ASSETS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs",
        "assets",
    )
    if os.path.isdir(_CSS_DIR):
        app.mount("/css", StaticFiles(directory=_CSS_DIR), name="css")
    if os.path.isdir(_JS_DIR):
        app.mount("/js", StaticFiles(directory=_JS_DIR), name="js")
    if os.path.isdir(_ASSETS_DIR):
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

# Shared enrollment store
_store = EnrollmentStore()
_campus_store = CampusStore()

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


def _campus_user_id(student_id: str) -> str:
    """Namespace campus enrollments inside the existing embedding store."""
    return f"campus_{student_id}"


def _make_image_preview(image_bytes: bytes) -> Optional[str]:
    """Create a compact data-URL preview for demo review screens."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((360, 360))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=78, optimize=True)
        encoded = base64.b64encode(out.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


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


@app.get("/campus/status")
async def campus_status():
    """Health/status for the FaceVerify Campus showcase."""
    snapshot = _campus_store.snapshot()
    return {
        "status": "ok",
        "product": "FaceVerify Campus",
        "university": "Northbridge University",
        "face_models_available": [
            model_type
            for (modality, model_type) in MODEL_REGISTRY.keys()
            if modality == "face"
        ],
        "courses": len(snapshot["courses"]),
        "exams": len(snapshot["exams"]),
        "students": len(snapshot["students"]),
        "attempts": len(snapshot["attempts"]),
        "models_loaded": [
            f"{m}/{t}" for (m, t) in _engines.keys()
        ],
    }


@app.get("/campus")
async def campus_snapshot():
    """Return all demo state for dashboards."""
    return _campus_store.snapshot()


@app.post("/campus/reset")
async def campus_reset():
    """Reset the campus showcase to the Northbridge demo scenario."""
    ids_to_delete = set(_campus_store.student_ids())
    snapshot = _campus_store.reset_demo()
    ids_to_delete.update(student["student_id"] for student in snapshot["students"])
    for student_id in ids_to_delete:
        _store.delete_user(_campus_user_id(student_id))
    return snapshot


@app.get("/campus/courses")
async def campus_courses():
    return _campus_store.list_courses()


@app.post("/campus/courses")
async def campus_create_course(
    course_id: str = Form(...),
    name: str = Form(...),
    instructor: str = Form(""),
    term: str = Form(""),
):
    try:
        return _campus_store.create_course(course_id, name, instructor, term)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/campus/exams")
async def campus_exams():
    return _campus_store.list_exams()


@app.post("/campus/exams")
async def campus_create_exam(
    exam_id: str = Form(...),
    course_id: str = Form(...),
    name: str = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    threshold: float = Form(0.65),
    model_type: str = Form("siamese"),
):
    try:
        return _campus_store.create_exam(
            exam_id=exam_id,
            course_id=course_id,
            name=name,
            start_time=start_time,
            end_time=end_time,
            threshold=threshold,
            model_type=model_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/campus/students")
async def campus_students(course_id: Optional[str] = None):
    return _campus_store.list_students(course_id=course_id)


@app.post("/campus/roster/import")
async def campus_import_roster(
    course_id: str = Form(...),
    roster: UploadFile = File(...),
):
    contents = await roster.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty roster uploaded.")
    try:
        csv_text = contents.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Roster must be UTF-8 CSV.")
    try:
        return _campus_store.import_roster(course_id, csv_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/campus/exams/{exam_id}/roster")
async def campus_exam_roster(exam_id: str):
    try:
        return _campus_store.exam_roster(exam_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/campus/students/{student_id}/enroll")
async def campus_enroll_student(
    student_id: str,
    model_type: str = Form("siamese"),
    images: List[UploadFile] = File(...),
):
    student = _campus_store.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"Unknown student '{student_id}'.")
    _validate_model_type(model_type)
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="At least one image required.")
    if len(images) > 5:
        raise HTTPException(status_code=400, detail="Upload at most 5 face samples.")

    campus_user = _campus_user_id(student_id)
    existing = _store.get_user(campus_user)
    existing_count = existing["sample_count"] if existing else 0
    if existing_count + len(images) > 5:
        raise HTTPException(
            status_code=400,
            detail="Campus demo enrollment is limited to 5 total face samples.",
        )

    engine = _get_engine("face", model_type)
    total_enrolled = existing_count
    all_warnings = []
    preview = None

    for img_file in images:
        image_bytes = await _read_image_bytes(img_file)
        val = validate_image(image_bytes, "face")
        if not val.passed:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image '{img_file.filename}' failed validation: "
                    f"{'; '.join(val.warnings)}"
                ),
            )
        if val.warnings:
            all_warnings.extend(f"{img_file.filename}: {w}" for w in val.warnings)
        if preview is None:
            preview = _make_image_preview(image_bytes)
        try:
            embedding = engine.extract_embedding(image_bytes, validate=False)
            result = _store.enroll(campus_user, "face", model_type, embedding)
            total_enrolled = result["sample_count"]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Enrollment processing error: {str(e)}",
            )

    updated = _campus_store.record_enrollment(
        student_id,
        total_enrolled,
        reference_preview=preview,
    )
    return {
        "student": updated,
        "sample_count": total_enrolled,
        "validation_warnings": all_warnings,
    }


@app.post("/campus/exams/{exam_id}/verify")
async def campus_verify_exam_attempt(
    exam_id: str,
    student_id: str = Form(...),
    image: UploadFile = File(...),
):
    exam = _campus_store.get_exam(exam_id)
    if exam is None:
        raise HTTPException(status_code=404, detail=f"Unknown exam '{exam_id}'.")
    student = _campus_store.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"Unknown student '{student_id}'.")
    if exam["course_id"] not in student.get("course_ids", []):
        raise HTTPException(
            status_code=403,
            detail=f"Student '{student_id}' is not on this exam roster.",
        )

    campus_user = _campus_user_id(student_id)
    user_info = _store.get_user(campus_user)
    if user_info is None:
        raise HTTPException(
            status_code=409,
            detail="Student must complete face enrollment before exam verification.",
        )
    if user_info["model_type"] != exam["model_type"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Student is enrolled with {user_info['model_type']}; "
                f"this exam requires {exam['model_type']}."
            ),
        )

    image_bytes = await _read_image_bytes(image)
    val = validate_image(image_bytes, "face")
    if not val.passed:
        raise HTTPException(
            status_code=400,
            detail=f"Image failed validation: {'; '.join(val.warnings)}",
        )

    engine = _get_engine("face", exam["model_type"])
    try:
        enrolled_embeddings = _store.get_embeddings(campus_user)
        result = engine.verify_against_prototype(
            image_bytes,
            enrolled_embeddings,
            validate=False,
        )
        attempt = _campus_store.record_attempt(
            exam_id=exam_id,
            student_id=student_id,
            score=result["score"],
            threshold=exam["threshold"],
            model_type=exam["model_type"],
            validation=val.to_dict(),
            query_preview=_make_image_preview(image_bytes),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

    return {
        "attempt": attempt,
        "student": _campus_store.get_student(student_id),
        "exam": exam,
        "message": _campus_result_message(attempt["decision"]),
    }


@app.get("/campus/attempts")
async def campus_attempts(
    exam_id: Optional[str] = None,
    student_id: Optional[str] = None,
):
    return _campus_store.list_attempts(exam_id=exam_id, student_id=student_id)


@app.get("/campus/attempts/{attempt_id}")
async def campus_attempt(attempt_id: str):
    attempt = _campus_store.get_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail=f"Unknown attempt '{attempt_id}'.")
    student = _campus_store.get_student(attempt["student_id"])
    return {"attempt": attempt, "student": student}


@app.post("/campus/attempts/{attempt_id}/review")
async def campus_review_attempt(
    attempt_id: str,
    reviewer: str = Form("Proctor"),
    action: str = Form(...),
    reason: str = Form(""),
):
    try:
        return _campus_store.review_attempt(attempt_id, reviewer, action, reason)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/campus/audit")
async def campus_audit():
    return _campus_store.audit_log()


@app.get("/campus/audit.csv")
async def campus_audit_csv():
    return Response(
        content=_campus_store.audit_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=faceverify_audit.csv"},
    )


@app.post("/campus/model-lab/compare")
async def campus_model_lab_compare(
    model_type: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    _validate_model_type(model_type)
    engine = _get_engine("face", model_type)
    bytes1 = await _read_image_bytes(image1)
    bytes2 = await _read_image_bytes(image2)
    try:
        return engine.compare(bytes1, bytes2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model lab error: {str(e)}")


def _campus_result_message(decision: str) -> str:
    if decision == "verified":
        return "Verified - Exam Access Granted"
    if decision == "manual_review":
        return "Manual Review Required"
    return "Rejected - Use fallback verification"


# ── Root: serve the UI ───────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI."""
    index_path = os.path.join(_UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "API is running. Visit /docs for Swagger UI."})
