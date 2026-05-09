"""
FaceNet-style FLUXSynID sanity checks for FaceVerify.

This module runs a small, report-only first-layer check against FLUXSynID:
doc + two live images become the enrollment prototype, and the remaining live
image is tested as the genuine exam-day query plus impostor comparisons.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.hybrid_calibration import normalize_embedding, runtime_similarity_score


DEFAULT_MODEL_TYPE = "hybrid"
DEFAULT_THRESHOLD = 0.3000000119
DEFAULT_SWEEP_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_LABELS = {
    "hybrid": "Hybrid FaceNet",
    "facenet_proto": "FaceNet Proto",
}


class FluxSanityError(ValueError):
    """Raised when FLUXSynID sanity input is missing required data."""


@dataclass(frozen=True)
class FluxIdentity:
    identity: str
    doc_image: Path
    live_images: Tuple[Path, ...]

    @property
    def enrollment_images(self) -> Tuple[Path, ...]:
        return (self.doc_image, self.live_images[0], self.live_images[1])

    @property
    def query_image(self) -> Path:
        return self.live_images[2]


@dataclass(frozen=True)
class FluxSanityConfig:
    dataset_dir: Path
    output_dir: Path = Path("results") / "hybrid_face" / "flux_sanity"
    model_type: str = DEFAULT_MODEL_TYPE
    identities: Optional[int] = 20
    seed: int = 42
    threshold: float = DEFAULT_THRESHOLD
    impostors_per_identity: int = 5
    sweep_thresholds: Tuple[float, ...] = DEFAULT_SWEEP_THRESHOLDS


def discover_flux_identities(dataset_dir: Path) -> Dict[str, FluxIdentity]:
    """Return eligible FLUXSynID identities with one doc and at least three live images."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FluxSanityError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise FluxSanityError(f"Dataset path is not a directory: {dataset_dir}")

    identities: Dict[str, FluxIdentity] = {}
    for identity_dir in sorted(dataset_dir.iterdir(), key=lambda p: p.name.lower()):
        if not identity_dir.is_dir():
            continue
        images = [
            p for p in sorted(identity_dir.iterdir(), key=lambda p: p.name.lower())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        doc_images = [p for p in images if "_doc" in p.stem]
        live_images = tuple(p for p in images if "_live_" in p.stem)
        if doc_images and len(live_images) >= 3:
            identities[identity_dir.name] = FluxIdentity(
                identity=identity_dir.name,
                doc_image=doc_images[0],
                live_images=live_images[:3],
            )

    if not identities:
        raise FluxSanityError(f"No FLUXSynID identities found in {dataset_dir}.")
    return identities


def select_flux_identities(
    identities: Dict[str, FluxIdentity],
    *,
    count: Optional[int],
    seed: int,
) -> List[FluxIdentity]:
    if count is None:
        return [identities[key] for key in sorted(identities)]
    if count < 2:
        raise FluxSanityError("--identities must be at least 2.")
    if len(identities) < count:
        raise FluxSanityError(
            f"Requested {count} identities, but only {len(identities)} are eligible."
        )
    keys = sorted(identities)
    rng = random.Random(seed)
    rng.shuffle(keys)
    return [identities[key] for key in sorted(keys[:count])]


def run_flux_sanity(
    config: FluxSanityConfig,
    extract_embedding: Callable[[Path], np.ndarray],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Run FaceNet-style FLUXSynID sanity checks and write local artifacts."""
    discovered = discover_flux_identities(config.dataset_dir)
    selected = select_flux_identities(
        discovered,
        count=config.identities,
        seed=config.seed,
    )
    if config.impostors_per_identity < 1:
        raise FluxSanityError("--impostors-per-identity must be at least 1.")
    if config.impostors_per_identity >= len(selected):
        raise FluxSanityError("--impostors-per-identity must be lower than --identities.")

    prototypes, query_embeddings = _build_runtime_embeddings(
        selected,
        extract_embedding,
        progress_callback=progress_callback,
    )
    trials = _generate_flux_trials(
        selected,
        prototypes,
        query_embeddings,
        threshold=config.threshold,
        impostors_per_identity=config.impostors_per_identity,
        seed=config.seed,
    )
    metrics = summarize_flux_trials(config, selected, trials)
    metrics["threshold_sweep"] = threshold_sweep(trials, config.sweep_thresholds)
    artifacts = write_flux_sanity_artifacts(config.output_dir, metrics, trials)
    metrics["artifacts"] = artifacts
    return metrics


def _build_runtime_embeddings(
    identities: Sequence[FluxIdentity],
    extract_embedding: Callable[[Path], np.ndarray],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Dict[str, dict], Dict[str, np.ndarray]]:
    prototypes: Dict[str, dict] = {}
    query_embeddings: Dict[str, np.ndarray] = {}

    total = len(identities)
    for index, item in enumerate(identities, start=1):
        enrollment_embeddings = [
            normalize_embedding(extract_embedding(path))
            for path in item.enrollment_images
        ]
        prototype = normalize_embedding(np.mean(enrollment_embeddings, axis=0))
        prototypes[item.identity] = {
            "prototype": prototype,
            "enrollment_images": [str(path) for path in item.enrollment_images],
        }
        query_embeddings[item.identity] = normalize_embedding(extract_embedding(item.query_image))
        if progress_callback and (index % 100 == 0 or index == total):
            progress_callback(index, total)

    return prototypes, query_embeddings


def _generate_flux_trials(
    identities: Sequence[FluxIdentity],
    prototypes: Dict[str, dict],
    query_embeddings: Dict[str, np.ndarray],
    *,
    threshold: float,
    impostors_per_identity: int,
    seed: int,
) -> List[dict]:
    identity_ids = [item.identity for item in identities]
    by_id = {item.identity: item for item in identities}
    rng = random.Random(seed)
    trials: List[dict] = []

    for query_identity in sorted(identity_ids):
        item = by_id[query_identity]
        genuine_score = runtime_similarity_score(
            query_embeddings[query_identity],
            prototypes[query_identity]["prototype"],
        )
        trials.append(_trial_row(
            trial_type="genuine",
            enrolled_identity=query_identity,
            query_identity=query_identity,
            score=genuine_score,
            threshold=threshold,
            enrollment_images=prototypes[query_identity]["enrollment_images"],
            query_image=str(item.query_image),
        ))

        candidates = [identity for identity in sorted(identity_ids) if identity != query_identity]
        impostor_ids = sorted(rng.sample(candidates, impostors_per_identity))
        for enrolled_identity in impostor_ids:
            impostor_score = runtime_similarity_score(
                query_embeddings[query_identity],
                prototypes[enrolled_identity]["prototype"],
            )
            trials.append(_trial_row(
                trial_type="impostor",
                enrolled_identity=enrolled_identity,
                query_identity=query_identity,
                score=impostor_score,
                threshold=threshold,
                enrollment_images=prototypes[enrolled_identity]["enrollment_images"],
                query_image=str(item.query_image),
            ))

    return trials


def _trial_row(
    *,
    trial_type: str,
    enrolled_identity: str,
    query_identity: str,
    score: float,
    threshold: float,
    enrollment_images: Sequence[str],
    query_image: str,
) -> dict:
    return {
        "trial_type": trial_type,
        "enrolled_identity": enrolled_identity,
        "query_identity": query_identity,
        "score": round(float(score), 8),
        "threshold": round(float(threshold), 8),
        "passed": bool(score >= threshold),
        "enrollment_images": "|".join(enrollment_images),
        "query_image": query_image,
    }


def summarize_flux_trials(
    config: FluxSanityConfig,
    identities: Sequence[FluxIdentity],
    trials: Sequence[dict],
) -> dict:
    genuine = [float(t["score"]) for t in trials if t["trial_type"] == "genuine"]
    impostor = [float(t["score"]) for t in trials if t["trial_type"] == "impostor"]
    genuine_failures = [
        t for t in trials
        if t["trial_type"] == "genuine" and not t["passed"]
    ]
    false_accepts = [
        t for t in trials
        if t["trial_type"] == "impostor" and t["passed"]
    ]

    return {
        "model_type": config.model_type,
        "model_label": MODEL_LABELS.get(config.model_type, config.model_type),
        "dataset_path": str(config.dataset_dir),
        "threshold": round(float(config.threshold), 8),
        "seed": config.seed,
        "identities_tested": len(identities),
        "impostors_per_identity": config.impostors_per_identity,
        "genuine": {
            "trials": len(genuine),
            "passed": len(genuine) - len(genuine_failures),
            "failed": len(genuine_failures),
            "score_stats": _score_stats(genuine),
        },
        "impostor": {
            "trials": len(impostor),
            "false_accepts": len(false_accepts),
            "score_stats": _score_stats(impostor),
        },
        "worst_genuine": sorted(
            [t for t in trials if t["trial_type"] == "genuine"],
            key=lambda t: float(t["score"]),
        )[:5],
        "highest_impostors": sorted(
            [t for t in trials if t["trial_type"] == "impostor"],
            key=lambda t: float(t["score"]),
            reverse=True,
        )[:10],
        "failure_rows": sorted(
            genuine_failures + false_accepts,
            key=lambda t: (t["trial_type"], -float(t["score"])),
        ),
    }


def threshold_sweep(trials: Sequence[dict], thresholds: Sequence[float]) -> List[dict]:
    genuine = [float(t["score"]) for t in trials if t["trial_type"] == "genuine"]
    impostor = [float(t["score"]) for t in trials if t["trial_type"] == "impostor"]
    rows = []
    for threshold in thresholds:
        genuine_passed = sum(score >= threshold for score in genuine)
        impostor_false_accepts = sum(score >= threshold for score in impostor)
        genuine_failed = len(genuine) - genuine_passed
        rows.append({
            "threshold": round(float(threshold), 8),
            "genuine_passed": int(genuine_passed),
            "genuine_failed": int(genuine_failed),
            "impostor_false_accepts": int(impostor_false_accepts),
            "genuine_trials": int(len(genuine)),
            "impostor_trials": int(len(impostor)),
            "frr": round(float(genuine_failed / len(genuine)), 8) if genuine else None,
            "far": round(float(impostor_false_accepts / len(impostor)), 8) if impostor else None,
        })
    return rows


def _score_stats(scores: Sequence[float]) -> dict:
    if not scores:
        return {"min": None, "mean": None, "max": None}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), 8),
        "mean": round(float(np.mean(arr)), 8),
        "max": round(float(np.max(arr)), 8),
    }


def write_flux_sanity_artifacts(
    output_dir: Path,
    metrics: dict,
    trials: Sequence[dict],
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    failures_path = output_dir / "failures.csv"
    summary_path = output_dir / "summary.md"
    metrics_path = output_dir / "metrics.json"
    threshold_sweep_path = output_dir / "threshold_sweep.csv"

    _write_csv(results_path, trials)
    _write_csv(failures_path, metrics["failure_rows"])
    _write_threshold_sweep_csv(threshold_sweep_path, metrics["threshold_sweep"])
    artifacts = {
        "summary": str(summary_path),
        "metrics": str(metrics_path),
        "results": str(results_path),
        "failures": str(failures_path),
        "threshold_sweep": str(threshold_sweep_path),
    }
    metrics_path.write_text(
        json.dumps({**metrics, "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(render_summary_markdown(metrics), encoding="utf-8")

    return artifacts


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "trial_type",
        "enrolled_identity",
        "query_identity",
        "score",
        "threshold",
        "passed",
        "enrollment_images",
        "query_image",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_threshold_sweep_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "threshold",
        "genuine_passed",
        "genuine_failed",
        "impostor_false_accepts",
        "genuine_trials",
        "impostor_trials",
        "frr",
        "far",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_summary_markdown(metrics: dict) -> str:
    genuine = metrics["genuine"]
    impostor = metrics["impostor"]
    model_label = metrics.get("model_label") or metrics.get("model_type", "FaceNet-style")
    lines = [
        f"# {model_label} FLUXSynID Sanity Report",
        "",
        "This is a first-layer sanity check only. It tests a FaceNet-style "
        "FaceVerify model and does not modify production thresholds.",
        "",
        "## Summary",
        "",
        f"- Model: `{metrics.get('model_type', 'unknown')}`",
        f"- Dataset: `{metrics['dataset_path']}`",
        f"- Identities tested: `{metrics['identities_tested']}`",
        f"- Threshold: `{metrics['threshold']}`",
        f"- Genuine pass/fail: `{genuine['passed']}` / `{genuine['failed']}`",
        f"- Impostor false accepts: `{impostor['false_accepts']}` of `{impostor['trials']}`",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | Genuine Failed | Impostor False Accepts | FRR | FAR |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["threshold_sweep"]:
        lines.append(
            f"| {row['threshold']} | {row['genuine_failed']} | "
            f"{row['impostor_false_accepts']} | {row['frr']} | {row['far']} |"
        )
    lines.extend([
        "",
        "## Score Stats",
        "",
        f"- Genuine min/mean/max: `{_format_stats(genuine['score_stats'])}`",
        f"- Impostor min/mean/max: `{_format_stats(impostor['score_stats'])}`",
        "",
        "## Worst Genuine Scores",
        "",
    ])
    lines.extend(_trial_bullets(metrics["worst_genuine"]))
    lines.extend(["", "## Highest Impostor Scores", ""])
    lines.extend(_trial_bullets(metrics["highest_impostors"]))
    return "\n".join(lines) + "\n"


def _format_stats(stats: dict) -> str:
    return f"{stats['min']} / {stats['mean']} / {stats['max']}"


def _trial_bullets(trials: Sequence[dict]) -> List[str]:
    if not trials:
        return ["- None"]
    return [
        "- "
        f"{trial['trial_type']} enrolled=`{trial['enrolled_identity']}` "
        f"query=`{trial['query_identity']}` score=`{trial['score']}` "
        f"passed=`{trial['passed']}`"
        for trial in trials
    ]
