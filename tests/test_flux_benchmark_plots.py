import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.flux_benchmark_plots import (  # noqa: E402
    BenchmarkPlotError,
    generate_benchmark_chart_report,
    load_model_summaries,
    select_best_model,
    select_overlay_models,
)


def test_load_model_summaries_and_select_best_model(tmp_path):
    run_dir = _write_benchmark_fixture(tmp_path)

    summaries = load_model_summaries(run_dir)
    best = select_best_model(summaries)

    assert [item.model_type for item in summaries] == ["hybrid", "facenet_contrastive_proto"]
    assert best.model_type == "hybrid"
    assert best.far == pytest.approx(0.01)
    assert best.frr == pytest.approx(0.002)


def test_select_overlay_models_respects_requested_order(tmp_path):
    run_dir = _write_benchmark_fixture(tmp_path)
    summaries = load_model_summaries(run_dir)

    selected = select_overlay_models(
        summaries,
        ["facenet_contrastive_proto", "hybrid"],
    )

    assert [item.model_type for item in selected] == ["facenet_contrastive_proto", "hybrid"]


def test_generate_benchmark_chart_report_writes_expected_artifacts(tmp_path):
    run_dir = _write_benchmark_fixture(tmp_path)

    artifacts = generate_benchmark_chart_report(
        run_dir,
        title="Fixture Benchmark",
        top_models=["hybrid", "facenet_contrastive_proto"],
    )

    assert artifacts.executive_summary.exists()
    assert artifacts.buyer_summary_table.exists()
    assert artifacts.buyer_summary_table_tr.exists()
    assert (artifacts.output_dir / "far_frr_comparison.png").exists()
    assert (artifacts.output_dir / "false_accepts_false_rejects.png").exists()
    assert (artifacts.output_dir / "score_distribution_hybrid.png").exists()
    assert (artifacts.output_dir / "score_distribution_overlay.png").exists()
    assert (artifacts.output_dir / "threshold_sweep_hybrid.png").exists()
    assert "Hybrid FaceNet accepted" in artifacts.executive_summary.read_text(encoding="utf-8")
    turkish_table = artifacts.buyer_summary_table_tr.read_text(encoding="utf-8-sig")
    assert "model_tipi,model_adi,esik_degeri" in turkish_table
    assert "Hibrit FaceNet" in turkish_table


def test_generate_benchmark_chart_report_reports_missing_comparison(tmp_path):
    with pytest.raises(BenchmarkPlotError, match="comparison.csv"):
        generate_benchmark_chart_report(tmp_path)


def test_generate_benchmark_chart_report_supports_turkish_labels(tmp_path):
    run_dir = _write_benchmark_fixture(tmp_path)

    artifacts = generate_benchmark_chart_report(
        run_dir,
        title="FLUXSynID 3000 Kimlik Benchmark",
        top_models=["hybrid", "facenet_contrastive_proto"],
        language="tr",
    )

    summary = artifacts.executive_summary.read_text(encoding="utf-8")
    assert "Yönetici Özeti" in summary
    assert "Hibrit FaceNet" in summary
    assert (artifacts.output_dir / "far_frr_comparison.png").exists()


def _write_benchmark_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "flux_fixture"
    run_dir.mkdir()
    comparison_rows = [
        {
            "model_type": "hybrid",
            "model_label": "Hybrid FaceNet",
            "status": "completed",
            "checkpoint_available": "True",
            "checkpoint_path": "fake-hybrid.pth",
            "threshold": "0.800884",
            "identities_tested": "10",
            "genuine_trials": "10",
            "genuine_passed": "10",
            "genuine_failed": "0",
            "impostor_trials": "50",
            "impostor_false_accepts": "1",
            "frr": "0.002",
            "far": "0.01",
            "genuine_score_min": "0.8",
            "genuine_score_mean": "0.9",
            "genuine_score_max": "0.98",
            "impostor_score_min": "0.2",
            "impostor_score_mean": "0.5",
            "impostor_score_max": "0.82",
            "best_threshold": "0.800884",
            "best_threshold_far": "0.01",
            "best_threshold_frr": "0.002",
            "best_threshold_note": "within_acceptable_frr",
            "output_dir": str(run_dir / "hybrid"),
            "error": "",
        },
        {
            "model_type": "facenet_contrastive_proto",
            "model_label": "FaceNet Contrastive Proto",
            "status": "completed",
            "checkpoint_available": "True",
            "checkpoint_path": "fake-contrastive.pth",
            "threshold": "0.800884",
            "identities_tested": "10",
            "genuine_trials": "10",
            "genuine_passed": "10",
            "genuine_failed": "0",
            "impostor_trials": "50",
            "impostor_false_accepts": "7",
            "frr": "0.0",
            "far": "0.14",
            "genuine_score_min": "0.82",
            "genuine_score_mean": "0.94",
            "genuine_score_max": "0.99",
            "impostor_score_min": "0.18",
            "impostor_score_mean": "0.62",
            "impostor_score_max": "0.91",
            "best_threshold": "0.800884",
            "best_threshold_far": "0.14",
            "best_threshold_frr": "0",
            "best_threshold_note": "within_acceptable_frr",
            "output_dir": str(run_dir / "facenet_contrastive_proto"),
            "error": "",
        },
    ]
    _write_csv(run_dir / "comparison.csv", comparison_rows)
    for row in comparison_rows:
        model_dir = Path(row["output_dir"])
        model_dir.mkdir()
        _write_csv(model_dir / "results.csv", _trial_rows(row["model_type"], float(row["threshold"])))
        _write_csv(model_dir / "threshold_sweep.csv", _sweep_rows())
    return run_dir


def _trial_rows(model_type: str, threshold: float):
    rows = []
    for idx in range(10):
        identity = f"{model_type}_person_{idx:03d}"
        rows.append({
            "trial_type": "genuine",
            "enrolled_identity": identity,
            "query_identity": identity,
            "score": 0.90 + idx * 0.002,
            "threshold": threshold,
            "passed": True,
            "enrollment_images": f"C:/fixture/{identity}/{identity}_doc.jpg",
            "query_image": f"C:/fixture/{identity}/{identity}_live.jpg",
        })
        for impostor_idx in range(5):
            rows.append({
                "trial_type": "impostor",
                "enrolled_identity": f"{model_type}_other_{idx}_{impostor_idx}",
                "query_identity": identity,
                "score": 0.35 + impostor_idx * 0.05,
                "threshold": threshold,
                "passed": False,
                "enrollment_images": f"C:/fixture/other/{model_type}_other_{impostor_idx}.jpg",
                "query_image": f"C:/fixture/{identity}/{identity}_live.jpg",
            })
    return rows


def _sweep_rows():
    return [
        {"threshold": "0.7", "genuine_passed": "10", "genuine_failed": "0", "impostor_false_accepts": "10", "genuine_trials": "10", "impostor_trials": "50", "frr": "0", "far": "0.2"},
        {"threshold": "0.800884", "genuine_passed": "10", "genuine_failed": "0", "impostor_false_accepts": "1", "genuine_trials": "10", "impostor_trials": "50", "frr": "0", "far": "0.02"},
        {"threshold": "0.9", "genuine_passed": "8", "genuine_failed": "2", "impostor_false_accepts": "0", "genuine_trials": "10", "impostor_trials": "50", "frr": "0.2", "far": "0"},
    ]


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
