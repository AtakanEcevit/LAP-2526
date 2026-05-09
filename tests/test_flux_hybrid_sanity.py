import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.flux_hybrid_sanity import (  # noqa: E402
    FluxSanityConfig,
    FluxSanityError,
    discover_flux_identities,
    run_flux_sanity,
    select_flux_identities,
    threshold_sweep,
)


def test_discover_flux_identities_requires_doc_and_three_live_images(tmp_path):
    _write_flux_identity(tmp_path, "eligible_001")
    _write_image(tmp_path / "too_small" / "too_small_f_doc.jpg")
    _write_image(tmp_path / "too_small" / "too_small_f_live_0_a_d1.jpg")

    identities = discover_flux_identities(tmp_path)

    assert list(identities) == ["eligible_001"]
    item = identities["eligible_001"]
    assert item.doc_image.name == "eligible_001_f_doc.jpg"
    assert len(item.live_images) == 3
    assert item.query_image.name == "eligible_001_f_live_0_p_d1.jpg"


def test_select_flux_identities_is_deterministic(tmp_path):
    for idx in range(8):
        _write_flux_identity(tmp_path, f"person_{idx:03d}")
    identities = discover_flux_identities(tmp_path)

    first = [item.identity for item in select_flux_identities(identities, count=4, seed=42)]
    second = [item.identity for item in select_flux_identities(identities, count=4, seed=42)]

    assert first == second
    assert len(first) == 4


def test_select_flux_identities_all_returns_every_eligible_identity(tmp_path):
    for idx in range(5):
        _write_flux_identity(tmp_path, f"person_{idx:03d}")
    identities = discover_flux_identities(tmp_path)

    selected = select_flux_identities(identities, count=None, seed=42)

    assert [item.identity for item in selected] == [f"person_{idx:03d}" for idx in range(5)]


def test_run_flux_sanity_generates_expected_trials_and_outputs(tmp_path):
    dataset_dir = tmp_path / "flux"
    output_dir = tmp_path / "out"
    for idx in range(6):
        _write_flux_identity(dataset_dir, f"person_{idx:03d}")

    metrics = run_flux_sanity(
        FluxSanityConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            identities=4,
            seed=42,
            threshold=0.75,
            impostors_per_identity=2,
        ),
        _fake_embedding,
    )

    assert metrics["identities_tested"] == 4
    assert metrics["genuine"]["trials"] == 4
    assert metrics["impostor"]["trials"] == 8
    rows = _read_csv(output_dir / "results.csv")
    selected_identities = {row["query_identity"] for row in rows if row["trial_type"] == "genuine"}
    assert len(selected_identities) == 4
    for identity in selected_identities:
        genuine_rows = [
            row for row in rows
            if row["trial_type"] == "genuine" and row["query_identity"] == identity
        ]
        assert len(genuine_rows) == 1
        assert genuine_rows[0]["enrolled_identity"] == identity
        assert _path_parent_names(genuine_rows[0]["enrollment_images"]) == {identity}
        assert _path_parent_name(genuine_rows[0]["query_image"]) == identity

        impostor_rows = [
            row for row in rows
            if row["trial_type"] == "impostor" and row["query_identity"] == identity
        ]
        assert len(impostor_rows) == 2
        impostor_enrollments = [row["enrolled_identity"] for row in impostor_rows]
        assert identity not in impostor_enrollments
        assert len(impostor_enrollments) == len(set(impostor_enrollments))
        for row in impostor_rows:
            assert _path_parent_names(row["enrollment_images"]) == {row["enrolled_identity"]}
            assert _path_parent_name(row["query_image"]) == identity
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "failures.csv").exists()
    assert (output_dir / "threshold_sweep.csv").exists()
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Hybrid FaceNet FLUXSynID Sanity Report" in summary
    assert "FaceNet-style FaceVerify model" in summary


def test_run_flux_sanity_records_selected_model_type(tmp_path):
    dataset_dir = tmp_path / "flux"
    output_dir = tmp_path / "out"
    for idx in range(5):
        _write_flux_identity(dataset_dir, f"person_{idx:03d}")

    metrics = run_flux_sanity(
        FluxSanityConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            model_type="facenet_proto",
            identities=4,
            seed=42,
            threshold=0.47,
            impostors_per_identity=2,
        ),
        _fake_embedding,
    )

    assert metrics["model_type"] == "facenet_proto"
    assert metrics["model_label"] == "FaceNet Proto"
    assert "FaceNet Proto FLUXSynID Sanity Report" in (
        output_dir / "summary.md"
    ).read_text(encoding="utf-8")


def test_run_flux_sanity_rejects_impostor_count_equal_to_identity_count(tmp_path):
    for idx in range(3):
        _write_flux_identity(tmp_path, f"person_{idx:03d}")

    with pytest.raises(FluxSanityError):
        run_flux_sanity(
            FluxSanityConfig(
                dataset_dir=tmp_path,
                output_dir=tmp_path / "out",
                identities=3,
                impostors_per_identity=3,
            ),
            _fake_embedding,
        )


def test_threshold_sweep_counts_far_and_frr():
    trials = [
        _trial("genuine", 0.90),
        _trial("genuine", 0.60),
        _trial("impostor", 0.80),
        _trial("impostor", 0.20),
    ]

    rows = threshold_sweep(trials, [0.50, 0.85])

    assert rows[0]["genuine_failed"] == 0
    assert rows[0]["impostor_false_accepts"] == 1
    assert rows[0]["far"] == 0.5
    assert rows[1]["genuine_failed"] == 1
    assert rows[1]["impostor_false_accepts"] == 0
    assert rows[1]["frr"] == 0.5


def _write_flux_identity(root, identity):
    identity_dir = root / identity
    _write_image(identity_dir / f"{identity}_f_doc.jpg")
    _write_image(identity_dir / f"{identity}_f_live_0_a_d1.jpg")
    _write_image(identity_dir / f"{identity}_f_live_0_e_d1.jpg")
    _write_image(identity_dir / f"{identity}_f_live_0_p_d1.jpg")
    (identity_dir / f"{identity}_f.json").write_text("{}", encoding="utf-8")


def _write_image(path, color=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (color, color, color)).save(path)


def _fake_embedding(path):
    path = str(path)
    identity_name = os.path.basename(os.path.dirname(path))
    identity_idx = int(identity_name.split("_")[-1])
    vec = np.zeros(16, dtype=np.float32)
    vec[identity_idx] = 1.0
    if "_live_" in os.path.basename(path):
        vec[-1] = 0.03
    return vec


def _read_csv(path):
    import csv

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _path_parent_name(path):
    from pathlib import Path

    return Path(path).parent.name


def _path_parent_names(paths):
    return {_path_parent_name(path) for path in paths.split("|")}


def _trial(trial_type, score):
    return {
        "trial_type": trial_type,
        "score": score,
        "passed": score >= 0.5,
    }
