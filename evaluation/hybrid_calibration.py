"""
Report-only calibration workflow for the FaceVerify hybrid face model.

The functions here intentionally evaluate the same runtime score used by the
API: L2-normalized embeddings compared with cosine similarity mapped to [0, 1].
No production threshold file is modified by this module.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.metrics import compute_all_metrics, compute_far_frr


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".webp"}
DEFAULT_FAR_TARGETS = (0.10, 0.05, 0.01, 0.001)


class CalibrationError(ValueError):
    """Raised when a calibration dataset is missing required structure/data."""


@dataclass(frozen=True)
class CalibrationConfig:
    dataset_dir: Path
    output_dir: Path
    k_shot: int = 3
    seed: int = 42
    min_images: int = 4
    current_threshold: float = 0.3000000119
    calibration_fraction: float = 0.70
    bootstrap_iterations: int = 1000
    max_identities: Optional[int] = None
    max_impostors_per_query: Optional[int] = None


def discover_identity_images(dataset_dir: Path, min_images: int = 4) -> Dict[str, List[Path]]:
    """Return identity -> sorted image paths from a folder-per-identity dataset."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise CalibrationError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise CalibrationError(f"Dataset path is not a directory: {dataset_dir}")

    identities: Dict[str, List[Path]] = {}
    for child in sorted(dataset_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        images = [
            p
            for p in sorted(child.iterdir(), key=lambda p: p.name.lower())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(images) >= min_images:
            identities[child.name] = images

    if not identities:
        raise CalibrationError(
            f"No identities with at least {min_images} images were found in {dataset_dir}."
        )
    return identities


def split_identities(
    identities: Sequence[str],
    calibration_fraction: float = 0.70,
    seed: int = 42,
    max_identities: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Deterministically split identities into calibration and holdout sets."""
    ids = sorted(identities)
    rng = random.Random(seed)
    rng.shuffle(ids)

    if max_identities is not None:
        if max_identities < 2:
            raise CalibrationError("max_identities must be at least 2.")
        ids = ids[:max_identities]

    if len(ids) < 2:
        raise CalibrationError("At least two eligible identities are required.")

    split_at = int(math.floor(len(ids) * calibration_fraction))
    split_at = max(1, min(split_at, len(ids) - 1))
    return sorted(ids[:split_at]), sorted(ids[split_at:])


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a 1-D L2-normalized embedding."""
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise CalibrationError("Embedding norm is zero; cannot score calibration trial.")
    return arr / norm


def runtime_similarity_score(embedding: np.ndarray, prototype: np.ndarray) -> float:
    """Runtime score: cosine similarity mapped from [-1, 1] into [0, 1]."""
    emb = normalize_embedding(embedding)
    proto = normalize_embedding(prototype)
    cosine = float(np.dot(emb, proto))
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


def build_embeddings(
    identity_images: Dict[str, List[Path]],
    extract_embedding: Callable[[Path], np.ndarray],
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """Extract embeddings, returning usable samples and per-image failures."""
    embeddings: Dict[str, List[dict]] = {}
    failures: List[dict] = []

    for identity, paths in identity_images.items():
        samples = []
        for image_path in paths:
            try:
                embedding = normalize_embedding(extract_embedding(image_path))
            except Exception as exc:  # noqa: BLE001 - keep report-only run resilient.
                failures.append({
                    "identity": identity,
                    "path": str(image_path),
                    "error": str(exc),
                })
                continue
            samples.append({
                "identity": identity,
                "path": str(image_path),
                "embedding": embedding,
            })
        if samples:
            embeddings[identity] = samples

    return embeddings, failures


def generate_trials(
    embeddings_by_identity: Dict[str, List[dict]],
    identities: Sequence[str],
    *,
    split_name: str,
    k_shot: int = 3,
    seed: int = 42,
    max_impostors_per_query: Optional[int] = None,
) -> Tuple[List[dict], List[str]]:
    """Build genuine and impostor prototype-verification trials."""
    eligible = [
        identity
        for identity in sorted(identities)
        if len(embeddings_by_identity.get(identity, [])) >= k_shot + 1
    ]
    skipped = sorted(set(identities) - set(eligible))
    if len(eligible) < 2:
        return [], skipped

    prototypes: Dict[str, dict] = {}
    for identity in eligible:
        samples = embeddings_by_identity[identity]
        support = samples[:k_shot]
        prototype = normalize_embedding(np.mean([s["embedding"] for s in support], axis=0))
        prototypes[identity] = {
            "prototype": prototype,
            "support_paths": [s["path"] for s in support],
            "queries": samples[k_shot:],
        }

    rng = random.Random(seed)
    trials: List[dict] = []
    for query_identity in eligible:
        query_samples = prototypes[query_identity]["queries"]
        for query_sample in query_samples:
            query_embedding = query_sample["embedding"]

            genuine_score = runtime_similarity_score(
                query_embedding,
                prototypes[query_identity]["prototype"],
            )
            trials.append(_trial_row(
                split_name,
                "genuine",
                genuine_score,
                enrolled_identity=query_identity,
                query_identity=query_identity,
                query_path=query_sample["path"],
                support_paths=prototypes[query_identity]["support_paths"],
            ))

            impostor_identities = [i for i in eligible if i != query_identity]
            if max_impostors_per_query is not None:
                impostor_identities = rng.sample(
                    impostor_identities,
                    min(max_impostors_per_query, len(impostor_identities)),
                )
                impostor_identities.sort()

            for enrolled_identity in impostor_identities:
                impostor_score = runtime_similarity_score(
                    query_embedding,
                    prototypes[enrolled_identity]["prototype"],
                )
                trials.append(_trial_row(
                    split_name,
                    "impostor",
                    impostor_score,
                    enrolled_identity=enrolled_identity,
                    query_identity=query_identity,
                    query_path=query_sample["path"],
                    support_paths=prototypes[enrolled_identity]["support_paths"],
                ))

    return trials, skipped


def _trial_row(
    split_name: str,
    label: str,
    score: float,
    *,
    enrolled_identity: str,
    query_identity: str,
    query_path: str,
    support_paths: Sequence[str],
) -> dict:
    return {
        "split": split_name,
        "label": label,
        "score": round(float(score), 8),
        "enrolled_identity": enrolled_identity,
        "query_identity": query_identity,
        "query_path": query_path,
        "support_paths": "|".join(support_paths),
    }


def scores_from_trials(trials: Sequence[dict]) -> Tuple[List[float], List[float]]:
    genuine = [float(t["score"]) for t in trials if t["label"] == "genuine"]
    impostor = [float(t["score"]) for t in trials if t["label"] == "impostor"]
    return genuine, impostor


def threshold_for_target_far(impostor_scores: Sequence[float], target_far: float) -> dict:
    """Return a conservative empirical threshold for a target FAR."""
    scores = np.sort(np.asarray(impostor_scores, dtype=np.float64))[::-1]
    if scores.size == 0:
        raise CalibrationError("Cannot compute FAR threshold without impostor scores.")
    if not 0.0 < target_far < 1.0:
        raise CalibrationError("target_far must be between 0 and 1.")

    allowed_accepts = int(math.floor(target_far * scores.size))
    if allowed_accepts <= 0:
        threshold = min(1.0, float(np.nextafter(scores[0], 1.0)))
    elif allowed_accepts >= scores.size:
        threshold = 0.0
    else:
        threshold = float(np.nextafter(scores[allowed_accepts], 1.0))

    actual_far = float(np.mean(scores >= threshold))
    return {
        "target_far": target_far,
        "threshold": round(threshold, 8),
        "observed_far": round(actual_far, 8),
        "impostor_trials": int(scores.size),
        "estimable": bool(scores.size >= math.ceil(1.0 / target_far)),
    }


def threshold_metrics(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    threshold: float,
) -> dict:
    far, frr = compute_far_frr(genuine_scores, impostor_scores, threshold)
    genuine = np.asarray(genuine_scores)
    impostor = np.asarray(impostor_scores)
    accepted_genuine = int(np.sum(genuine >= threshold))
    accepted_impostor = int(np.sum(impostor >= threshold))
    total = len(genuine) + len(impostor)
    accuracy = (
        (accepted_genuine + int(np.sum(impostor < threshold))) / total
        if total else 0.0
    )
    return {
        "threshold": round(float(threshold), 8),
        "far": round(float(far), 8),
        "frr": round(float(frr), 8),
        "accuracy": round(float(accuracy), 8),
        "accepted_genuine": accepted_genuine,
        "accepted_impostor": accepted_impostor,
        "genuine_trials": int(len(genuine)),
        "impostor_trials": int(len(impostor)),
    }


def summarize_trials(
    trials: Sequence[dict],
    *,
    current_threshold: float,
    far_targets: Sequence[float] = DEFAULT_FAR_TARGETS,
) -> dict:
    genuine, impostor = scores_from_trials(trials)
    if len(genuine) < 2 or len(impostor) < 2:
        raise CalibrationError(
            f"Insufficient trials for metrics: genuine={len(genuine)}, "
            f"impostor={len(impostor)}."
        )

    metrics = compute_all_metrics(genuine, impostor)
    candidates = [threshold_for_target_far(impostor, target) for target in far_targets]

    return {
        "counts": {
            "identities": len({t["query_identity"] for t in trials}),
            "genuine_trials": len(genuine),
            "impostor_trials": len(impostor),
        },
        "score_stats": {
            "genuine_mean": round(float(np.mean(genuine)), 8),
            "genuine_min": round(float(np.min(genuine)), 8),
            "genuine_max": round(float(np.max(genuine)), 8),
            "impostor_mean": round(float(np.mean(impostor)), 8),
            "impostor_min": round(float(np.min(impostor)), 8),
            "impostor_max": round(float(np.max(impostor)), 8),
        },
        "eer": round(float(metrics["eer"]), 8),
        "eer_threshold": round(float(metrics["eer_threshold"]), 8),
        "auc": round(float(metrics["auc"]), 8),
        "d_prime": round(float(metrics["d_prime"]), 8),
        "current_threshold": threshold_metrics(genuine, impostor, current_threshold),
        "far_thresholds": candidates,
    }


def bootstrap_by_identity(
    trials: Sequence[dict],
    *,
    threshold: float,
    iterations: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap FAR/FRR/accuracy/EER confidence intervals by query identity."""
    if iterations <= 0:
        return {"iterations": 0}

    by_identity: Dict[str, List[dict]] = {}
    for trial in trials:
        by_identity.setdefault(trial["query_identity"], []).append(trial)
    identities = sorted(by_identity)
    if len(identities) < 2:
        return {"iterations": 0, "reason": "Need at least two identities."}

    rng = random.Random(seed)
    far_values = []
    frr_values = []
    accuracy_values = []
    eer_values = []
    eer_threshold_values = []

    for _ in range(iterations):
        sample_trials: List[dict] = []
        for identity in (rng.choice(identities) for _ in identities):
            sample_trials.extend(by_identity[identity])
        genuine, impostor = scores_from_trials(sample_trials)
        if len(genuine) < 2 or len(impostor) < 2:
            continue
        current = threshold_metrics(genuine, impostor, threshold)
        far_values.append(current["far"])
        frr_values.append(current["frr"])
        accuracy_values.append(current["accuracy"])
        try:
            metrics = compute_all_metrics(genuine, impostor)
            eer_values.append(float(metrics["eer"]))
            eer_threshold_values.append(float(metrics["eer_threshold"]))
        except Exception:
            continue

    return {
        "iterations": len(far_values),
        "threshold": round(float(threshold), 8),
        "far_ci95": _ci(far_values),
        "frr_ci95": _ci(frr_values),
        "accuracy_ci95": _ci(accuracy_values),
        "eer_ci95": _ci(eer_values),
        "eer_threshold_ci95": _ci(eer_threshold_values),
    }


def _ci(values: Sequence[float]) -> Optional[List[float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [round(float(lo), 8), round(float(hi), 8)]


def evaluate_thresholds_on_trials(
    trials: Sequence[dict],
    thresholds: Dict[str, float],
) -> dict:
    genuine, impostor = scores_from_trials(trials)
    return {
        name: threshold_metrics(genuine, impostor, threshold)
        for name, threshold in thresholds.items()
    }


def make_verdict(
    *,
    total_identities: int,
    calibration_summary: dict,
    holdout_summary: dict,
    holdout_thresholds: dict,
    current_threshold: float,
) -> dict:
    """Classify calibration quality using transparent, conservative rules."""
    holdout_current = holdout_thresholds["current_threshold"]
    holdout_eer = holdout_summary["eer"]
    holdout_auc = holdout_summary["auc"]
    calibration_auc = calibration_summary["auc"]
    current_far = holdout_current["far"]
    current_frr = holdout_current["frr"]
    calibrated_threshold = calibration_summary["eer_threshold"]
    threshold_delta = abs(float(calibrated_threshold) - float(current_threshold))

    reasons = []
    if total_identities < 20:
        reasons.append("Target dataset has fewer than 20 eligible identities.")
    if min(calibration_auc, holdout_auc) < 0.85 or holdout_eer > 0.20:
        level = "Red"
        reasons.append("Holdout separation is weak.")
    elif (
        total_identities >= 20
        and min(calibration_auc, holdout_auc) >= 0.95
        and holdout_eer <= 0.10
        and current_far <= 0.01
        and current_frr <= 0.20
        and threshold_delta <= 0.05
    ):
        level = "Green"
        reasons.append("Separation and current-threshold holdout behavior are stable.")
    else:
        level = "Yellow"
        reasons.append(
            "Model separates faces, but threshold support is not strong enough "
            "to call final."
        )

    supported = level == "Green"
    if not supported:
        reasons.append(
            f"Current threshold support check: FAR={current_far:.4f}, "
            f"FRR={current_frr:.4f}, EER threshold delta={threshold_delta:.4f}."
        )

    return {
        "level": level,
        "current_threshold_supported": supported,
        "reasons": reasons,
    }


def write_scores_csv(path: Path, trials: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "label",
        "score",
        "enrolled_identity",
        "query_identity",
        "query_path",
        "support_paths",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            writer.writerow({field: trial.get(field, "") for field in fields})


def write_metrics_json(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def write_plots(output_dir: Path, trials: Sequence[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import det_curve, roc_curve

    output_dir.mkdir(parents=True, exist_ok=True)
    genuine, impostor = scores_from_trials(trials)
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([np.asarray(genuine), np.asarray(impostor)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(genuine, bins=40, alpha=0.65, label="Genuine", color="#2f9e44", density=True)
    ax.hist(impostor, bins=40, alpha=0.65, label="Impostor", color="#c92a2a", density=True)
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Density")
    ax.set_title("Hybrid Face Score Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False acceptance rate")
    ax.set_ylabel("True acceptance rate")
    ax.set_title("Hybrid Face ROC Curve")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fpr_det, fnr_det, _ = det_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.maximum(fpr_det, 1e-6) * 100, np.maximum(fnr_det, 1e-6) * 100)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False acceptance rate (%)")
    ax.set_ylabel("False rejection rate (%)")
    ax.set_title("Hybrid Face DET Curve")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "det_curve.png", dpi=150)
    plt.close(fig)


def write_report(path: Path, metrics: dict) -> None:
    verdict = metrics["verdict"]
    cal = metrics["calibration"]
    hold = metrics["holdout"]
    holdout_thresholds = metrics["holdout_thresholds"]
    current = holdout_thresholds["current_threshold"]
    current_value = metrics["calibration"]["current_threshold"]["threshold"]

    safer = _safer_threshold_text(cal)
    lines = [
        "# Hybrid Face Calibration Report",
        "",
        f"**Verdict:** {verdict['level']}",
        "",
        "This report is evidence-only. It did not modify production thresholds or "
        "`inference/calibrated_thresholds.json`.",
        "",
        f"## Current Threshold {current_value:.3f}",
        "",
        f"- Supported by evidence: **{verdict['current_threshold_supported']}**",
        f"- Holdout FAR: `{current['far']}`",
        f"- Holdout FRR: `{current['frr']}`",
        f"- Holdout accuracy: `{current['accuracy']}`",
        "",
        "## Safer Threshold Range",
        "",
        safer,
        "",
        "## Calibration Split",
        "",
        f"- Identities: `{cal['counts']['identities']}`",
        f"- Genuine trials: `{cal['counts']['genuine_trials']}`",
        f"- Impostor trials: `{cal['counts']['impostor_trials']}`",
        f"- EER: `{cal['eer']}`",
        f"- EER threshold: `{cal['eer_threshold']}`",
        f"- AUC: `{cal['auc']}`",
        "",
        "## Holdout Split",
        "",
        f"- Identities: `{hold['counts']['identities']}`",
        f"- Genuine trials: `{hold['counts']['genuine_trials']}`",
        f"- Impostor trials: `{hold['counts']['impostor_trials']}`",
        f"- EER: `{hold['eer']}`",
        f"- EER threshold: `{hold['eer_threshold']}`",
        f"- AUC: `{hold['auc']}`",
        "",
        "## Confidence Intervals",
        "",
        f"- Calibration current-threshold FAR CI95: "
        f"`{metrics['bootstrap']['calibration_current'].get('far_ci95')}`",
        f"- Calibration current-threshold FRR CI95: "
        f"`{metrics['bootstrap']['calibration_current'].get('frr_ci95')}`",
        f"- Holdout current-threshold FAR CI95: "
        f"`{metrics['bootstrap']['holdout_current'].get('far_ci95')}`",
        f"- Holdout current-threshold FRR CI95: "
        f"`{metrics['bootstrap']['holdout_current'].get('frr_ci95')}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {reason}" for reason in verdict["reasons"])
    if metrics.get("image_failures"):
        lines.append(f"- Image extraction failures: `{len(metrics['image_failures'])}`")
    if metrics.get("skipped_identities"):
        lines.append(
            f"- Identities skipped after extraction: `{len(metrics['skipped_identities'])}`"
        )
    if metrics["dataset"]["eligible_identities"] < 20:
        lines.append("- More target data is needed before declaring a final threshold.")
    else:
        lines.append("- Dataset size meets the first-pass target-data requirement.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safer_threshold_text(calibration_summary: dict) -> str:
    candidates = calibration_summary["far_thresholds"]
    if not candidates:
        return "- No FAR-based threshold could be estimated."
    useful = [
        c for c in candidates
        if c["target_far"] in {0.05, 0.01} and c["estimable"]
    ]
    if not useful:
        useful = candidates[:2]
    values = ", ".join(
        f"{c['threshold']} for target FAR {c['target_far']:g}"
        for c in useful
    )
    return f"- Candidate conservative thresholds: {values}."


def run_calibration(
    config: CalibrationConfig,
    extract_embedding: Callable[[Path], np.ndarray],
) -> dict:
    """Run calibration and write report artifacts."""
    identity_images = discover_identity_images(config.dataset_dir, config.min_images)
    if config.max_identities is not None:
        ids = sorted(identity_images)
        rng = random.Random(config.seed)
        rng.shuffle(ids)
        selected = set(ids[:config.max_identities])
        identity_images = {
            identity: paths
            for identity, paths in identity_images.items()
            if identity in selected
        }

    embeddings, image_failures = build_embeddings(identity_images, extract_embedding)
    usable_identities = [
        identity for identity, samples in embeddings.items()
        if len(samples) >= config.k_shot + 1
    ]
    if len(usable_identities) < 2:
        raise CalibrationError(
            f"Need at least two identities with {config.k_shot + 1}+ usable images; "
            f"found {len(usable_identities)}."
        )

    calibration_ids, holdout_ids = split_identities(
        usable_identities,
        calibration_fraction=config.calibration_fraction,
        seed=config.seed,
    )

    calibration_trials, skipped_cal = generate_trials(
        embeddings,
        calibration_ids,
        split_name="calibration",
        k_shot=config.k_shot,
        seed=config.seed,
        max_impostors_per_query=config.max_impostors_per_query,
    )
    holdout_trials, skipped_hold = generate_trials(
        embeddings,
        holdout_ids,
        split_name="holdout",
        k_shot=config.k_shot,
        seed=config.seed + 1,
        max_impostors_per_query=config.max_impostors_per_query,
    )
    if not calibration_trials or not holdout_trials:
        raise CalibrationError("Calibration and holdout splits must both produce trials.")

    calibration_summary = summarize_trials(
        calibration_trials,
        current_threshold=config.current_threshold,
    )
    holdout_summary = summarize_trials(
        holdout_trials,
        current_threshold=config.current_threshold,
    )
    strict_far = next(
        c for c in calibration_summary["far_thresholds"] if c["target_far"] == 0.01
    )
    holdout_thresholds = evaluate_thresholds_on_trials(
        holdout_trials,
        {
            "current_threshold": config.current_threshold,
            "calibration_eer": calibration_summary["eer_threshold"],
            "calibration_far_0.01": strict_far["threshold"],
        },
    )
    verdict = make_verdict(
        total_identities=len(usable_identities),
        calibration_summary=calibration_summary,
        holdout_summary=holdout_summary,
        holdout_thresholds=holdout_thresholds,
        current_threshold=config.current_threshold,
    )

    metrics = {
        "dataset": {
            "path": str(config.dataset_dir),
            "eligible_identities": len(usable_identities),
            "calibration_identities": calibration_ids,
            "holdout_identities": holdout_ids,
            "k_shot": config.k_shot,
            "seed": config.seed,
        },
        "calibration": calibration_summary,
        "holdout": holdout_summary,
        "holdout_thresholds": holdout_thresholds,
        "bootstrap": {
            "calibration_current": bootstrap_by_identity(
                calibration_trials,
                threshold=config.current_threshold,
                iterations=config.bootstrap_iterations,
                seed=config.seed,
            ),
            "holdout_current": bootstrap_by_identity(
                holdout_trials,
                threshold=config.current_threshold,
                iterations=config.bootstrap_iterations,
                seed=config.seed + 1,
            ),
        },
        "verdict": verdict,
        "image_failures": image_failures,
        "skipped_identities": sorted(set(skipped_cal + skipped_hold)),
        "artifacts": {
            "report": str(config.output_dir / "report.md"),
            "metrics": str(config.output_dir / "metrics.json"),
            "scores": str(config.output_dir / "scores.csv"),
            "score_distribution": str(config.output_dir / "score_distribution.png"),
            "roc_curve": str(config.output_dir / "roc_curve.png"),
            "det_curve": str(config.output_dir / "det_curve.png"),
        },
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_trials = calibration_trials + holdout_trials
    write_scores_csv(config.output_dir / "scores.csv", all_trials)
    write_metrics_json(config.output_dir / "metrics.json", metrics)
    write_plots(config.output_dir, all_trials)
    write_report(config.output_dir / "report.md", metrics)
    return metrics
