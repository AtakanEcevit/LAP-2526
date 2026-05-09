#!/usr/bin/env python3
"""
Run a quick FaceNet-style sanity check against FLUXSynID.

Example:
    python test_flux_hybrid_sanity.py C:\\Users\\USER\\Downloads\\FLUXSynID\\FLUXSynID\\FLUXSynID --device cpu
    python test_flux_hybrid_sanity.py C:\\Users\\USER\\Downloads\\FLUXSynID\\FLUXSynID\\FLUXSynID --model-type facenet_proto --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.flux_hybrid_sanity import (  # noqa: E402
    DEFAULT_MODEL_TYPE,
    FluxSanityConfig,
    FluxSanityError,
    run_flux_sanity,
)
from inference.config import MODEL_REGISTRY  # noqa: E402
from inference.engine import VerificationEngine  # noqa: E402


FACENET_STYLE_MODELS = ("hybrid", "facenet_proto")


def _build_extractor(model_type: str, device: str | None, validate: bool):
    engine = VerificationEngine()
    engine.load("face", model_type, device=device)

    def extract(image_path: Path) -> np.ndarray:
        return engine.extract_embedding(str(image_path), validate=validate)

    return extract


def main() -> int:
    parser = argparse.ArgumentParser(
        description="First-layer FLUXSynID sanity test for FaceVerify FaceNet-style models."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="FLUXSynID identity folder, e.g. ...\\FLUXSynID\\FLUXSynID\\FLUXSynID.",
    )
    parser.add_argument(
        "--model-type",
        choices=FACENET_STYLE_MODELS,
        default=DEFAULT_MODEL_TYPE,
        help="Face model to test. Defaults to hybrid for backward compatibility.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument(
        "--identities",
        default="20",
        help='Number of identities to test, or "all" for every eligible identity.',
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic identity/trial seed.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold to evaluate without modifying runtime config. Defaults to registry threshold.",
    )
    parser.add_argument(
        "--impostors-per-identity",
        type=int,
        default=5,
        help="Number of wrong-identity prototypes to compare per genuine query.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary.md, results.csv, and failures.csv.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip runtime image quality validation before embedding extraction.",
    )
    args = parser.parse_args()
    identities = _parse_identities(args.identities, parser)
    threshold = args.threshold
    if threshold is None:
        threshold = _registry_threshold(args.model_type)
    output_dir = args.output_dir or _default_output_dir(args.model_type)

    config = FluxSanityConfig(
        dataset_dir=args.dataset_dir,
        output_dir=output_dir,
        model_type=args.model_type,
        identities=identities,
        seed=args.seed,
        threshold=threshold,
        impostors_per_identity=args.impostors_per_identity,
    )

    try:
        metrics = run_flux_sanity(
            config,
            _build_extractor(args.model_type, args.device, validate=not args.skip_validation),
            progress_callback=_progress,
        )
    except FluxSanityError as exc:
        print(f"FLUX sanity error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI should fail with a clear message.
        print(f"{args.model_type} FLUX sanity test failed: {exc}", file=sys.stderr)
        return 1

    print_summary(metrics)
    return 0


def print_summary(metrics: dict) -> None:
    genuine = metrics["genuine"]
    impostor = metrics["impostor"]
    print(f"\n{metrics.get('model_label', metrics['model_type'])} FLUXSynID sanity complete.")
    print(f"  Model: {metrics['model_type']}")
    print(f"  Dataset: {metrics['dataset_path']}")
    print(f"  Identities tested: {metrics['identities_tested']}")
    print(f"  Threshold: {metrics['threshold']}")
    print(f"  Genuine pass/fail: {genuine['passed']} / {genuine['failed']}")
    print(f"  Impostor false accepts: {impostor['false_accepts']} of {impostor['trials']}")
    print(f"  Genuine score min/mean/max: {_format_stats(genuine['score_stats'])}")
    print(f"  Impostor score min/mean/max: {_format_stats(impostor['score_stats'])}")
    print("  Worst genuine scores:")
    for row in metrics["worst_genuine"][:5]:
        print(
            f"    {row['query_identity']} score={row['score']:.3f} "
            f"passed={row['passed']}"
        )
    print("  Highest impostor scores:")
    for row in metrics["highest_impostors"][:5]:
        print(
            f"    enrolled={row['enrolled_identity']} query={row['query_identity']} "
            f"score={row['score']:.3f} passed={row['passed']}"
        )
    print(f"  Summary: {metrics['artifacts']['summary']}")
    print(f"  Metrics: {metrics['artifacts']['metrics']}")
    print(f"  Results: {metrics['artifacts']['results']}")
    print(f"  Failures: {metrics['artifacts']['failures']}")
    print(f"  Threshold sweep: {metrics['artifacts']['threshold_sweep']}")
    print("  Production thresholds were not changed.\n")


def _format_stats(stats: dict) -> str:
    return f"{stats['min']} / {stats['mean']} / {stats['max']}"


def _parse_identities(value: str, parser: argparse.ArgumentParser):
    if str(value).lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError:
        parser.error('--identities must be an integer or "all"')
    if parsed < 2:
        parser.error("--identities must be at least 2")
    return parsed


def _registry_threshold(model_type: str) -> float:
    return float(MODEL_REGISTRY[("face", model_type)]["threshold"])


def _default_output_dir(model_type: str) -> Path:
    result_name = {
        "hybrid": "hybrid_face",
        "facenet_proto": "facenet_proto_face",
    }[model_type]
    return Path("results") / result_name / "flux_sanity"


def _progress(done: int, total: int) -> None:
    print(f"[flux-sanity] embedded {done}/{total} identities", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
