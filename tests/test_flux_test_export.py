import json
import zipfile

from inference.flux_test_export import (
    build_test_set_zip_bytes,
    export_flux_test_set,
    is_path_within,
    load_manifest,
    safe_export_filename,
)


def test_safe_export_filename_uses_student_id_and_image_suffix():
    assert safe_export_filename("NB-2026-1042", "live_p.JPG") == (
        "NB-2026-1042_matching_selfie.jpg"
    )
    assert safe_export_filename("bad id/with spaces", "image.tiff") == (
        "bad_id_with_spaces_matching_selfie.jpg"
    )


def test_export_flux_test_set_writes_files_and_manifests(tmp_path):
    flux_root = tmp_path / "flux"
    source = flux_root / "person_001" / "person_001_f_live_0_p_d1.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fake-image")
    output = tmp_path / "exports" / "current"

    result = export_flux_test_set(
        [
            {
                "student_id": "NB-2026-1042",
                "name": "Aylin Kaya",
                "face_source": "flux_synid",
                "face_identity": "person_001",
                "face_query_image": str(source),
                "face_model_type": "hybrid",
                "sample_count": 3,
            }
        ],
        output_dir=output,
        allowed_roots=[flux_root],
    )

    assert result["image_count"] == 1
    assert result["skipped"] == []
    assert (output / "NB-2026-1042_matching_selfie.jpg").read_bytes() == b"fake-image"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["image_count"] == 1
    assert "NB-2026-1042" in (output / "manifest.csv").read_text(encoding="utf-8")
    assert load_manifest(output)["available"] is True


def test_export_flux_test_set_rejects_untrusted_source(tmp_path):
    trusted_root = tmp_path / "trusted"
    outside = tmp_path / "outside" / "live_p.jpg"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"fake-image")

    result = export_flux_test_set(
        [
            {
                "student_id": "NB-2026-1042",
                "face_source": "flux_synid",
                "face_query_image": str(outside),
            }
        ],
        output_dir=tmp_path / "exports",
        allowed_roots=[trusted_root],
    )

    assert result["image_count"] == 0
    assert "outside allowed" in result["skipped"][0]["reason"]
    assert is_path_within(outside, [trusted_root]) is False


def test_build_test_set_zip_contains_manifest_and_images(tmp_path):
    flux_root = tmp_path / "flux"
    source = flux_root / "person_001" / "live_p.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fake-image")
    output = tmp_path / "exports"
    export_flux_test_set(
        [
            {
                "student_id": "NB-2026-1042",
                "name": "Aylin Kaya",
                "face_source": "flux_synid",
                "face_query_image": str(source),
            }
        ],
        output_dir=output,
        allowed_roots=[flux_root],
    )

    zip_bytes = build_test_set_zip_bytes(output)
    zip_path = tmp_path / "kit.zip"
    zip_path.write_bytes(zip_bytes)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())

    assert "manifest.json" in names
    assert "manifest.csv" in names
    assert "NB-2026-1042_matching_selfie.jpg" in names
