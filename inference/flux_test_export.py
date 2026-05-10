"""Export FLUXSynID query selfies for manual dashboard verification tests."""

import csv
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLUX_TEST_EXPORT_ROOT = PROJECT_ROOT / "data" / "flux_test_uploads"
DEFAULT_FLUX_TEST_EXPORT_DIR = DEFAULT_FLUX_TEST_EXPORT_ROOT / "current"
MANIFEST_JSON = "manifest.json"
MANIFEST_CSV = "manifest.csv"


def safe_export_filename(student_id: str, source_path: str | Path) -> str:
    """Return a stable, upload-friendly filename named by student id."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(student_id)).strip("._-")
    if not stem:
        stem = "student"
    suffix = Path(source_path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    return f"{stem}_matching_selfie{suffix}"


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def is_path_within(path: str | Path, roots: Iterable[str | Path]) -> bool:
    resolved = _resolve(path)
    for root in roots:
        try:
            resolved.relative_to(_resolve(root))
            return True
        except ValueError:
            continue
    return False


def load_manifest(output_dir: str | Path = DEFAULT_FLUX_TEST_EXPORT_DIR) -> dict:
    output_path = Path(output_dir)
    manifest_path = output_path / MANIFEST_JSON
    if not manifest_path.exists():
        return {
            "output_dir": str(output_path),
            "image_count": 0,
            "manifest": [],
            "skipped": [],
            "manifest_json": None,
            "manifest_csv": None,
            "available": False,
        }
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["available"] = bool(data.get("image_count"))
    return data


def clear_flux_test_set(output_dir: str | Path = DEFAULT_FLUX_TEST_EXPORT_DIR) -> dict:
    """Remove generated FLUXSynID test-kit artifacts and return an empty manifest."""
    output_path = Path(output_dir)
    resolved = output_path.expanduser().resolve(strict=False)
    if (
        resolved.name != DEFAULT_FLUX_TEST_EXPORT_DIR.name
        or resolved.parent.name != DEFAULT_FLUX_TEST_EXPORT_ROOT.name
    ):
        raise ValueError(f"Refusing to clear unexpected FLUX test-kit path: {output_path}")
    if output_path.is_dir():
        shutil.rmtree(output_path)
    elif output_path.exists():
        output_path.unlink()
    return load_manifest(output_path)


def export_flux_test_set(
    students: List[dict],
    output_dir: str | Path = DEFAULT_FLUX_TEST_EXPORT_DIR,
    allowed_roots: Optional[Iterable[str | Path]] = None,
    replace: bool = True,
) -> dict:
    """Copy current FLUXSynID query selfies into a runtime test kit."""
    output_path = Path(output_dir)
    allowed = list(allowed_roots or [])
    allowed.append(DEFAULT_FLUX_TEST_EXPORT_ROOT)
    allowed.append(output_path)

    if replace and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped = []
    for student in students:
        if student.get("face_source") != "flux_synid":
            continue

        student_id = student.get("student_id") or ""
        source_image = student.get("face_query_image")
        if not source_image:
            skipped.append({
                "student_id": student_id,
                "reason": "No reserved FLUXSynID query image is recorded.",
            })
            continue

        source_path = Path(source_image)
        if not is_path_within(source_path, allowed):
            skipped.append({
                "student_id": student_id,
                "source_query_image": str(source_path),
                "reason": "Reserved query image is outside allowed FLUXSynID/export roots.",
            })
            continue

        if not source_path.is_file():
            skipped.append({
                "student_id": student_id,
                "source_query_image": str(source_path),
                "reason": "Reserved query image file is missing.",
            })
            continue

        exported_filename = safe_export_filename(student_id, source_path)
        destination = output_path / exported_filename
        shutil.copy2(source_path, destination)
        manifest.append({
            "student_id": student_id,
            "name": student.get("name") or "",
            "face_identity": student.get("face_identity") or "",
            "source_query_image": str(source_path),
            "exported_filename": exported_filename,
            "exported_path": str(destination),
            "model_type": student.get("face_model_type") or "",
            "sample_count": student.get("sample_count") or 0,
        })

    payload = {
        "output_dir": str(output_path),
        "image_count": len(manifest),
        "manifest": manifest,
        "skipped": skipped,
        "manifest_json": str(output_path / MANIFEST_JSON),
        "manifest_csv": str(output_path / MANIFEST_CSV),
        "available": bool(manifest),
    }

    _write_manifest_json(output_path / MANIFEST_JSON, payload)
    _write_manifest_csv(output_path / MANIFEST_CSV, manifest)
    return payload


def build_test_set_zip_bytes(output_dir: str | Path = DEFAULT_FLUX_TEST_EXPORT_DIR) -> bytes:
    output_path = Path(output_dir)
    manifest = load_manifest(output_path)
    if not manifest.get("image_count"):
        raise FileNotFoundError("No FLUXSynID test images have been exported.")

    temp = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024)
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in (MANIFEST_JSON, MANIFEST_CSV):
            path = output_path / filename
            if path.exists():
                archive.write(path, arcname=filename)
        for entry in manifest.get("manifest", []):
            path = Path(entry.get("exported_path", ""))
            if path.is_file() and is_path_within(path, [output_path]):
                archive.write(path, arcname=entry["exported_filename"])
    temp.seek(0)
    return temp.read()


def _write_manifest_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_manifest_csv(path: Path, manifest: List[dict]) -> None:
    fields = [
        "student_id",
        "name",
        "face_identity",
        "source_query_image",
        "exported_filename",
        "exported_path",
        "model_type",
        "sample_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow({field: row.get(field, "") for field in fields})
