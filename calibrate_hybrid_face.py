#!/usr/bin/env python3
"""
Report-only calibration runner for the FaceVerify hybrid face model.

Example:
    python calibrate_hybrid_face.py C:\\path\\to\\target_faces --device cpu

The dataset must use one folder per identity:
    target_faces/person_001/img1.jpg
    target_faces/person_001/img2.jpg
    target_faces/person_002/img1.jpg

Outputs are written to results/hybrid_face/calibration by default. The script
does not modify inference/calibrated_thresholds.json or runtime configuration.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from evaluation.hybrid_calibration import CalibrationConfig, CalibrationError, run_calibration
from inference.engine import VerificationEngine


def _build_extractor(device: str | None, validate: bool):
    engine = VerificationEngine()
    engine.load("face", "hybrid", device=device)

    def extract(image_path: Path) -> np.ndarray:
        return engine.extract_embedding(str(image_path), validate=validate)

    return extract


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate/evaluate FaceVerify hybrid face threshold without changing runtime config."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Folder containing one subfolder per identity, each with face images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "hybrid_face" / "calibration",
        help="Directory for report.md, metrics.json, scores.csv, and plots.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument("--k-shot", type=int, default=3, help="Enrollment images per identity.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split/sampling seed.")
    parser.add_argument("--min-images", type=int, default=4, help="Minimum images per identity.")
    parser.add_argument(
        "--current-threshold",
        type=float,
        default=0.3000000119,
        help="Runtime threshold to evaluate, without modifying it.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Identity-bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--max-identities",
        type=int,
        default=None,
        help="Optional deterministic identity cap for quick sanity runs.",
    )
    parser.add_argument(
        "--max-impostors-per-query",
        type=int,
        default=None,
        help="Optional cap for impostor comparisons per query; default compares all other identities.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip runtime image quality validation before embedding extraction.",
    )
    args = parser.parse_args()

    if args.k_shot < 1:
        parser.error("--k-shot must be at least 1")
    if args.min_images < args.k_shot + 1:
        parser.error("--min-images must be at least --k-shot + 1")

    config = CalibrationConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        k_shot=args.k_shot,
        seed=args.seed,
        min_images=args.min_images,
        current_threshold=args.current_threshold,
        bootstrap_iterations=args.bootstrap_iterations,
        max_identities=args.max_identities,
        max_impostors_per_query=args.max_impostors_per_query,
    )

    try:
        metrics = run_calibration(
            config,
            _build_extractor(args.device, validate=not args.skip_validation),
        )
    except CalibrationError as exc:
        print(f"Calibration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI should fail with a clear message.
        print(f"Hybrid calibration failed: {exc}", file=sys.stderr)
        return 1

    verdict = metrics["verdict"]
    current = metrics["holdout_thresholds"]["current_threshold"]
    print("\nHybrid face calibration complete.")
    print(f"  Verdict: {verdict['level']}")
    print(f"  Current threshold supported: {verdict['current_threshold_supported']}")
    print(f"  Holdout FAR @ current threshold: {current['far']}")
    print(f"  Holdout FRR @ current threshold: {current['frr']}")
    print(f"  Report: {metrics['artifacts']['report']}")
    print("  Production thresholds were not changed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
