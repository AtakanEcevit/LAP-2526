#!/usr/bin/env python3
"""
Run a FLUXSynID benchmark across registered FaceVerify face models.

Example:
    python test_flux_face_models.py C:\\Users\\USER\\Downloads\\FLUXSynID\\FLUXSynID\\FLUXSynID --device cpu --identities 1000 --skip-validation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.flux_face_benchmark import (  # noqa: E402
    ALL_MODELS,
    FaceModelBenchmarkConfig,
    discover_registered_face_models,
    run_face_model_benchmark,
)
from evaluation.flux_hybrid_sanity import FluxSanityError  # noqa: E402


def main() -> int:
    registered_models = discover_registered_face_models()
    parser = argparse.ArgumentParser(
        description="Compare registered FaceVerify face models on FLUXSynID."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="FLUXSynID identity folder, e.g. ...\\FLUXSynID\\FLUXSynID\\FLUXSynID.",
    )
    parser.add_argument(
        "--model-type",
        action="append",
        default=None,
        choices=(ALL_MODELS, *registered_models),
        help="Face model to test. Repeat for multiple models, or use all. Defaults to all.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument(
        "--identities",
        default="1000",
        help='Number of identities to test, or "all" for every eligible identity.',
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic identity/trial seed.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Single-model threshold override. Not allowed with --model-type all.",
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
        help="Directory for aggregate comparison outputs.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip runtime image quality validation before embedding extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images to embed per forward pass. Defaults to 32.",
    )
    parser.add_argument(
        "--parallel-models",
        type=int,
        default=1,
        help="Number of model benchmarks to run concurrently. Defaults to 1.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately for missing checkpoints or per-model errors.",
    )
    args = parser.parse_args()

    identities = _parse_identities(args.identities, parser)
    model_types = tuple(args.model_type or (ALL_MODELS,))
    if args.threshold is not None and (len(model_types) != 1 or model_types[0] == ALL_MODELS):
        parser.error("--threshold can only be used when exactly one explicit --model-type is supplied.")
    threshold_overrides = {}
    if args.threshold is not None:
        threshold_overrides[model_types[0]] = args.threshold

    config = FaceModelBenchmarkConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir or _default_output_dir(identities),
        model_types=model_types,
        identities=identities,
        seed=args.seed,
        threshold_overrides=threshold_overrides,
        impostors_per_identity=args.impostors_per_identity,
        device=args.device,
        validate_images=not args.skip_validation,
        strict=args.strict,
        batch_size=args.batch_size,
        parallel_models=args.parallel_models,
    )

    try:
        result = run_face_model_benchmark(config, progress_callback=_progress)
    except FluxSanityError as exc:
        print(f"FLUX face benchmark error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI should fail with a clear message.
        print(f"FLUX face benchmark failed: {exc}", file=sys.stderr)
        return 1

    print_summary(result)
    return 0


def print_summary(result: dict) -> None:
    print("\nFLUXSynID face model benchmark complete.")
    print(f"  Dataset: {result['dataset_path']}")
    print(f"  Output: {result['output_dir']}")
    print(f"  Batch size: {result.get('batch_size')}")
    print(f"  Parallel models: {result.get('parallel_models')}")
    for row in result["models"]:
        if row["status"] != "completed":
            print(f"  {row['model_type']}: {row['status']} {row.get('error', '')}".rstrip())
            continue
        print(
            "  "
            f"{row['model_type']}: threshold={row['threshold']} "
            f"FRR={row['frr']} FAR={row['far']} "
            f"false_accepts={row['impostor_false_accepts']}/{row['impostor_trials']} "
            f"candidate={row['best_threshold']}"
        )
    print(f"  Comparison: {result['artifacts']['comparison']}")
    print(f"  CSV: {result['artifacts']['comparison_csv']}")
    print("  Production thresholds were not changed.\n")


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


def _default_output_dir(identities) -> Path:
    suffix = "all" if identities is None else str(identities)
    return Path("results") / "face_model_benchmark" / f"flux_{suffix}"


def _progress(model_type: str, done: int, total: int) -> None:
    print(f"[flux-benchmark] {model_type} embedded {done}/{total} identities", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
