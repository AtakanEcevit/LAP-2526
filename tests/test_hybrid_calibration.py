import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.hybrid_calibration import (  # noqa: E402
    CalibrationConfig,
    discover_identity_images,
    run_calibration,
    runtime_similarity_score,
    threshold_metrics,
)


def test_runtime_similarity_score_direction():
    assert runtime_similarity_score(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1.0)
    assert runtime_similarity_score(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(0.0)
    assert runtime_similarity_score(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.5)


def test_threshold_metrics_counts():
    metrics = threshold_metrics(
        genuine_scores=[0.95, 0.80, 0.40],
        impostor_scores=[0.70, 0.30],
        threshold=0.75,
    )

    assert metrics["far"] == 0.0
    assert metrics["frr"] == pytest.approx(1 / 3)
    assert metrics["accepted_genuine"] == 2
    assert metrics["accepted_impostor"] == 0


def test_discover_identity_images_filters_small_identities(tmp_path):
    _write_image(tmp_path / "person_001" / "1.jpg")
    for i in range(4):
        _write_image(tmp_path / "person_002" / f"{i}.jpg")

    identities = discover_identity_images(tmp_path, min_images=4)

    assert list(identities) == ["person_002"]
    assert len(identities["person_002"]) == 4


def test_run_calibration_writes_report_artifacts(tmp_path):
    dataset_dir = tmp_path / "target_faces"
    output_dir = tmp_path / "calibration"
    for identity_idx in range(6):
        identity_dir = dataset_dir / f"person_{identity_idx:03d}"
        for image_idx in range(4):
            _write_image(identity_dir / f"{image_idx}.jpg", color=80 + identity_idx)

    metrics = run_calibration(
        CalibrationConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            bootstrap_iterations=10,
        ),
        _fake_embedding,
    )

    assert metrics["dataset"]["eligible_identities"] == 6
    assert metrics["verdict"]["level"] in {"Green", "Yellow", "Red"}
    assert (output_dir / "report.md").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "scores.csv").exists()
    assert (output_dir / "score_distribution.png").exists()
    assert (output_dir / "roc_curve.png").exists()
    assert (output_dir / "det_curve.png").exists()

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "did not modify production thresholds" in report
    assert "Current Threshold 0.300" in report

    parsed = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert parsed["holdout_thresholds"]["current_threshold"]["threshold"] == pytest.approx(
        0.3000000119
    )


def _write_image(path, color=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (color, color, color)).save(path)


def _fake_embedding(path):
    path = str(path)
    identity = int(os.path.basename(os.path.dirname(path)).split("_")[-1])
    image = int(os.path.splitext(os.path.basename(path))[0])
    vec = np.zeros(16, dtype=np.float32)
    vec[identity] = 1.0
    vec[-1] = image * 0.01
    return vec
