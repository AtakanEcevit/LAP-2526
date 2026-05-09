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

from inference.campus_store import DEFAULT_FACE_MODEL, CampusStore
from inference.engine import VerificationEngine
from inference.enrollment_store import EnrollmentStore
from inference.flux_preupload import (
    DEFAULT_FLUX_COUNT,
    DEFAULT_FLUX_ROOT,
    DEFAULT_FLUX_SEED,
    FluxPreuploadError,
    count_flux_identity_dirs,
    normalize_flux_dataset_dir,
    select_flux_identities_from_dataset,
)
from inference.flux_test_export import (
    DEFAULT_FLUX_TEST_EXPORT_DIR,
    build_test_set_zip_bytes,
    export_flux_test_set,
    is_path_within,
    load_manifest,
)
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


def _read_local_image_bytes(path: str) -> bytes:
    """Read a local FLUXSynID image without exposing the raw file through HTTP."""
    with open(path, "rb") as f:
        contents = f.read()
    if len(contents) == 0:
        raise ValueError(f"Empty image file: {path}")
    if len(contents) > 10 * 1024 * 1024:
        raise ValueError(f"Image file too large (max 10MB): {path}")
    return contents


def _flux_status_payload(dataset_dir: Optional[str] = None) -> dict:
    configured_path = str(dataset_dir or DEFAULT_FLUX_ROOT)
    summary = _campus_store.flux_summary()
    payload = {
        "configured_path": configured_path,
        "normalized_path": None,
        "available": False,
        "eligible_identity_count": 0,
        "preuploaded_students": summary["preuploaded_students"],
        "student_ids": summary["student_ids"],
        "error": None,
    }
    try:
        normalized = normalize_flux_dataset_dir(dataset_dir)
        payload.update({
            "normalized_path": str(normalized),
            "available": True,
            "eligible_identity_count": count_flux_identity_dirs(normalized),
        })
    except FluxPreuploadError as exc:
        payload["error"] = str(exc)
    return payload


def _flux_export_allowed_roots(students: Optional[List[dict]] = None) -> List[str]:
    students = students or _campus_store.list_students()
    roots = [
        student.get("face_dataset_dir")
        for student in students
        if student.get("face_dataset_dir")
    ]
    try:
        roots.append(str(normalize_flux_dataset_dir(None)))
    except FluxPreuploadError:
        pass
    roots.append(str(DEFAULT_FLUX_TEST_EXPORT_DIR))
    return roots


def _export_flux_test_set_payload() -> dict:
    students = _campus_store.list_students()
    return export_flux_test_set(
        students,
        output_dir=DEFAULT_FLUX_TEST_EXPORT_DIR,
        allowed_roots=_flux_export_allowed_roots(students),
    )


def _flux_test_set_payload() -> dict:
    payload = load_manifest(DEFAULT_FLUX_TEST_EXPORT_DIR)
    summary = _campus_store.flux_summary()
    payload["preuploaded_students"] = summary["preuploaded_students"]
    payload["student_ids"] = summary["student_ids"]
    return payload


def _preupload_flux_students(
    dataset_dir: Optional[str],
    count: int,
    seed: int,
    model_type: str,
) -> dict:
    if count < 1:
        raise FluxPreuploadError("count must be at least 1.")
    normalized = normalize_flux_dataset_dir(dataset_dir)
    selected = select_flux_identities_from_dataset(normalized, count=count, seed=seed)
    students = _campus_store.list_students()[:count]
    engine = _get_engine("face", model_type)

    imported = []
    skipped = []
    warnings = []
    pair_count = min(len(students), len(selected))
    if len(students) < count:
        warnings.append(f"Only {len(students)} campus students are available.")
    if len(selected) < count:
        warnings.append(f"Only {len(selected)} eligible FLUXSynID identities are available.")

    for student, identity in zip(students[:pair_count], selected[:pair_count]):
        student_id = student["student_id"]
        campus_user = _campus_user_id(student_id)
        total_enrolled = 0
        preview = None
        replacement_started = False
        try:
            prepared_embeddings = []
            for image_path in identity.enrollment_images:
                image_bytes = _read_local_image_bytes(str(image_path))
                val = validate_image(image_bytes, "face")
                if not val.passed:
                    raise ValueError(
                        f"{image_path.name} failed validation: {'; '.join(val.warnings)}"
                    )
                if val.warnings:
                    warnings.extend(f"{student_id}/{image_path.name}: {w}" for w in val.warnings)
                if preview is None:
                    preview = _make_image_preview(image_bytes)
                embedding = engine.extract_embedding(image_bytes, validate=False)
                prepared_embeddings.append(embedding)

            _store.delete_user(campus_user)
            replacement_started = True
            for embedding in prepared_embeddings:
                result = _store.enroll(campus_user, "face", model_type, embedding)
                total_enrolled = result["sample_count"]

            updated = _campus_store.record_flux_enrollment(
                student_id=student_id,
                sample_count=total_enrolled,
                model_type=model_type,
                face_identity=identity.identity,
                enrollment_images=[str(path) for path in identity.enrollment_images],
                query_image=str(identity.query_image),
                dataset_dir=str(normalized),
                reference_preview=preview,
            )
            imported.append({
                "student_id": student_id,
                "face_identity": identity.identity,
                "sample_count": updated["sample_count"],
            })
        except Exception as exc:
            if replacement_started:
                _store.delete_user(campus_user)
            skipped.append({
                "student_id": student_id,
                "face_identity": identity.identity,
                "reason": str(exc),
            })

    status = _flux_status_payload(str(normalized))
    try:
        export_status = _export_flux_test_set_payload()
    except Exception as exc:
        export_status = {
            "available": False,
            "output_dir": str(DEFAULT_FLUX_TEST_EXPORT_DIR),
            "image_count": 0,
            "manifest": [],
            "skipped": [],
            "error": str(exc),
        }
        warnings.append(f"FLUXSynID test image export failed: {exc}")
    return {
        "dataset_dir": str(normalized),
        "requested_count": count,
        "imported_count": len(imported),
        "imported": imported,
        "skipped": skipped,
        "warnings": warnings,
        "status": status,
        "export": export_status,
    }


def _preloaded_query_image_for(student: dict, scenario: str) -> Optional[str]:
    if scenario == "matching":
        return student.get("face_query_image")
    if scenario != "impostor":
        raise ValueError("scenario must be matching or impostor.")
    for candidate in _campus_store.list_students():
        if (
            candidate.get("student_id") != student.get("student_id")
            and candidate.get("face_source") == "flux_synid"
            and candidate.get("face_query_image")
        ):
            return candidate["face_query_image"]
    raise ValueError("No other preuploaded FLUXSynID identity is available.")


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
    flux_result = None
    if _flux_status_payload()["available"]:
        try:
            flux_result = _preupload_flux_students(
                dataset_dir=None,
                count=DEFAULT_FLUX_COUNT,
                seed=DEFAULT_FLUX_SEED,
                model_type=DEFAULT_FACE_MODEL,
            )
            snapshot = _campus_store.snapshot()
        except Exception as exc:
            flux_result = {"error": str(exc)}
    snapshot["flux_preupload"] = flux_result
    return snapshot


@app.get("/campus/flux/status")
async def campus_flux_status(dataset_dir: Optional[str] = None):
    """Report FLUXSynID availability and current campus preupload state."""
    return _flux_status_payload(dataset_dir)


@app.post("/campus/flux/preupload")
async def campus_flux_preupload(
    dataset_dir: Optional[str] = Form(None),
    count: int = Form(DEFAULT_FLUX_COUNT),
    seed: int = Form(DEFAULT_FLUX_SEED),
    model_type: str = Form(DEFAULT_FACE_MODEL),
):
    _validate_model_type(model_type)
    try:
        return _preupload_flux_students(dataset_dir, count, seed, model_type)
    except FluxPreuploadError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/campus/flux/export-test-set")
async def campus_flux_export_test_set():
    """Copy current preuploaded FLUXSynID query selfies into the runtime test kit."""
    try:
        return _export_flux_test_set_payload()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not export test selfies: {e}")


@app.get("/campus/flux/test-set")
async def campus_flux_test_set():
    """Report the current exported FLUXSynID manual test kit."""
    return _flux_test_set_payload()


@app.get("/campus/flux/test-set.zip")
async def campus_flux_test_set_zip():
    try:
        contents = build_test_set_zip_bytes(DEFAULT_FLUX_TEST_EXPORT_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(
        content=contents,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=flux_test_uploads.zip"},
    )


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


@app.get("/campus/students/{student_id}/flux/test-image")
async def campus_student_flux_test_image(student_id: str):
    student = _campus_store.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"Unknown student '{student_id}'.")
    if student.get("face_source") != "flux_synid":
        raise HTTPException(
            status_code=409,
            detail="Student does not have a preuploaded FLUXSynID test selfie.",
        )

    manifest = load_manifest(DEFAULT_FLUX_TEST_EXPORT_DIR)
    entry = next(
        (
            item for item in manifest.get("manifest", [])
            if item.get("student_id") == student_id
        ),
        None,
    )
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="No exported selfie found. Run FLUXSynID test set export first.",
        )

    image_path = entry.get("exported_path")
    if not image_path or not is_path_within(image_path, [DEFAULT_FLUX_TEST_EXPORT_DIR]):
        raise HTTPException(status_code=409, detail="Exported selfie path is not trusted.")
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Exported selfie file is missing.")

    return FileResponse(image_path, filename=entry.get("exported_filename"))


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


@app.post("/campus/exams/{exam_id}/verify-preloaded")
async def campus_verify_preloaded_exam_attempt(
    exam_id: str,
    student_id: str = Form(...),
    scenario: str = Form("matching"),
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
    if student.get("face_source") != "flux_synid":
        raise HTTPException(
            status_code=409,
            detail="Student does not have a preuploaded FLUXSynID selfie.",
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

    try:
        query_path = _preloaded_query_image_for(student, scenario)
        if not query_path:
            raise ValueError("Student does not have a reserved preloaded selfie.")
        image_bytes = _read_local_image_bytes(query_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=409, detail=f"Could not read preloaded selfie: {e}")

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
        "scenario": scenario,
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
