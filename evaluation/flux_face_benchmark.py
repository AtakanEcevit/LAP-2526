"""
FLUXSynID benchmark workflow for registered FaceVerify face models.

The benchmark reuses the same identity split and trial structure across models:
three enrollment images build a prototype, and the remaining live image is used
as the query for one genuine and several impostor trials.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from evaluation.flux_hybrid_sanity import (
    DEFAULT_SWEEP_THRESHOLDS,
    FluxIdentity,
    FluxSanityConfig,
    FluxSanityError,
    discover_flux_identities,
    merge_thresholds,
    select_flux_identities,
    summarize_flux_trials,
    threshold_sweep,
    write_flux_sanity_artifacts,
)
from inference.config import MODEL_REGISTRY
from inference.engine import FACENET_STYLE_MODEL_TYPES, VerificationEngine
from inference.validation import validate_image


DEFAULT_ACCEPTABLE_FRR = 0.02
ALL_MODELS = "all"


@dataclass(frozen=True)
class FaceModelBenchmarkConfig:
    dataset_dir: Path
    output_dir: Path = Path("results") / "face_model_benchmark" / "flux_1000"
    model_types: Tuple[str, ...] = (ALL_MODELS,)
    identities: Optional[int] = 1000
    seed: int = 42
    threshold_overrides: Dict[str, float] = field(default_factory=dict)
    impostors_per_identity: int = 5
    sweep_thresholds: Tuple[float, ...] = DEFAULT_SWEEP_THRESHOLDS
    acceptable_frr: float = DEFAULT_ACCEPTABLE_FRR
    device: Optional[str] = None
    validate_images: bool = True
    strict: bool = False
    batch_size: int = 32
    parallel_models: int = 1


ModelRunner = Callable[
    [str, FluxSanityConfig, Optional[str], bool, int, Optional[Callable[[str, int, int], None]]],
    dict,
]


def discover_registered_face_models() -> Tuple[str, ...]:
    """Return registered face model types in registry order."""
    return tuple(
        model_type
        for (modality, model_type) in MODEL_REGISTRY
        if modality == "face"
    )


def expand_model_types(model_types: Sequence[str]) -> Tuple[str, ...]:
    """Expand 'all' to registered face models and validate explicit names."""
    requested = tuple(model_types or (ALL_MODELS,))
    registered = discover_registered_face_models()
    if ALL_MODELS in requested:
        if len(requested) > 1:
            raise FluxSanityError("--model-type all cannot be combined with explicit models.")
        return registered
    unknown = [model_type for model_type in requested if model_type not in registered]
    if unknown:
        raise FluxSanityError(
            f"Unknown face model type(s): {', '.join(unknown)}. "
            f"Known face models: {', '.join(registered)}."
        )
    return requested


def checkpoint_available(model_type: str) -> bool:
    return Path(MODEL_REGISTRY[("face", model_type)]["checkpoint"]).exists()


def registry_threshold(model_type: str) -> float:
    return float(MODEL_REGISTRY[("face", model_type)]["threshold"])


def model_label(model_type: str) -> str:
    labels = {
        "siamese": "Siamese",
        "prototypical": "Prototypical",
        "hybrid": "Hybrid FaceNet",
        "facenet_proto": "FaceNet Proto",
        "facenet_contrastive_proto": "FaceNet Contrastive Proto",
        "facenet_contrastive_proto_model5": "FaceNet Contrastive Proto Model 5",
        "facenet_arcface_triplet_model6": "FaceNet ArcFace Triplet Model 6",
    }
    return labels.get(model_type, model_type)


def run_face_model_benchmark(
    config: FaceModelBenchmarkConfig,
    model_runner: Optional[ModelRunner] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """Run one or more registered face models and write aggregate artifacts."""
    model_types = expand_model_types(config.model_types)
    if config.batch_size < 1:
        raise FluxSanityError("--batch-size must be at least 1.")
    if config.parallel_models < 1:
        raise FluxSanityError("--parallel-models must be at least 1.")
    runner = model_runner or run_flux_model_with_engine
    row_by_model: Dict[str, dict] = {}
    runnable: List[Tuple[str, dict, FluxSanityConfig]] = []

    for model_type in model_types:
        entry = MODEL_REGISTRY[("face", model_type)]
        model_output_dir = Path(config.output_dir) / model_type
        threshold = config.threshold_overrides.get(model_type, registry_threshold(model_type))
        base_row = {
            "model_type": model_type,
            "model_label": model_label(model_type),
            "checkpoint_path": entry["checkpoint"],
            "checkpoint_available": checkpoint_available(model_type),
            "threshold": round(float(threshold), 8),
            "output_dir": str(model_output_dir),
        }
        if not base_row["checkpoint_available"]:
            if config.strict:
                raise FluxSanityError(f"Missing checkpoint for face/{model_type}: {entry['checkpoint']}")
            row_by_model[model_type] = _missing_or_error_row(base_row, "missing_artifact")
            continue

        flux_config = FluxSanityConfig(
            dataset_dir=config.dataset_dir,
            output_dir=model_output_dir,
            model_type=model_type,
            identities=config.identities,
            seed=config.seed,
            threshold=threshold,
            impostors_per_identity=config.impostors_per_identity,
            sweep_thresholds=merge_thresholds(config.sweep_thresholds, (threshold,)),
        )
        runnable.append((model_type, base_row, flux_config))

    if config.parallel_models == 1 or len(runnable) <= 1:
        for model_type, base_row, flux_config in runnable:
            row_by_model[model_type] = _run_single_model_benchmark(
                model_type,
                base_row,
                flux_config,
                config,
                runner,
                progress_callback,
            )
    else:
        workers = min(config.parallel_models, len(runnable))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_by_model = {
                executor.submit(
                    _run_single_model_benchmark,
                    model_type,
                    base_row,
                    flux_config,
                    config,
                    runner,
                    progress_callback,
                ): model_type
                for model_type, base_row, flux_config in runnable
            }
            for future in as_completed(future_by_model):
                row_by_model[future_by_model[future]] = future.result()

    rows = [row_by_model[model_type] for model_type in model_types]

    artifacts = write_comparison_artifacts(config.output_dir, rows, config)
    return {
        "dataset_path": str(config.dataset_dir),
        "output_dir": str(config.output_dir),
        "batch_size": config.batch_size,
        "parallel_models": config.parallel_models,
        "models": rows,
        "artifacts": artifacts,
    }


def run_flux_model_with_engine(
    model_type: str,
    config: FluxSanityConfig,
    device: Optional[str],
    validate_images: bool,
    batch_size: int,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    engine = VerificationEngine()
    engine.load("face", model_type, device=device)
    scorer = EnginePrototypeScorer(engine, validate_images=validate_images)
    return run_flux_sanity_with_scorer(
        config,
        scorer.extract_embedding,
        scorer.score,
        extract_embeddings=scorer.extract_embeddings,
        batch_size=batch_size,
        progress_callback=(
            None if progress_callback is None
            else lambda done, total: progress_callback(model_type, done, total)
        ),
    )


class EnginePrototypeScorer:
    """Extract embeddings and score prototypes exactly like VerificationEngine."""

    def __init__(self, engine: VerificationEngine, validate_images: bool = True):
        self.engine = engine
        self.validate_images = validate_images

    def extract_embedding(self, image_path: Path) -> np.ndarray:
        return self.engine.extract_embedding(str(image_path), validate=self.validate_images)

    def extract_embeddings(self, image_paths: Sequence[Path], batch_size: int) -> List[np.ndarray]:
        if batch_size < 1:
            raise FluxSanityError("--batch-size must be at least 1.")
        embeddings: List[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            tensors = []
            for image_path in batch_paths:
                image_input = str(image_path)
                if self.validate_images:
                    validation = validate_image(image_input, self.engine.modality)
                    if not validation.passed:
                        raise ValueError(
                            f"Image validation failed for {image_path}: "
                            f"{'; '.join(validation.warnings)}"
                        )
                tensors.append(self.engine._preprocess_for_model(image_input))
            batch = torch.cat(tensors, dim=0).to(self.engine.device)
            with torch.no_grad():
                batch_embeddings = self.engine.model.get_embedding(batch)
            embeddings.extend(batch_embeddings.cpu().numpy())
        return embeddings

    def score(self, query_embedding: np.ndarray, prototype: np.ndarray) -> float:
        return score_embedding_against_prototype(self.engine, query_embedding, prototype)


def score_embedding_against_prototype(
    engine: VerificationEngine,
    query_embedding: np.ndarray,
    prototype: np.ndarray,
) -> float:
    """Score precomputed embeddings using VerificationEngine's prototype logic."""
    query_tensor = torch.from_numpy(np.asarray(query_embedding)).float().unsqueeze(0).to(engine.device)
    prototype_tensor = torch.from_numpy(np.asarray(prototype)).float().unsqueeze(0).to(engine.device)

    with torch.no_grad():
        if engine.model_type in FACENET_STYLE_MODEL_TYPES:
            sim = torch.mm(query_tensor, prototype_tensor.t()).squeeze().item()
            score = (sim + 1.0) / 2.0
        elif engine.model_type == "siamese":
            diff = torch.abs(query_tensor - prototype_tensor)
            score = torch.sigmoid(engine.model.classifier(diff)).squeeze().item()
        else:
            distance_type = getattr(engine.model, "distance_type", "euclidean")
            if distance_type == "cosine":
                sim = torch.mm(query_tensor, prototype_tensor.t()).squeeze().item()
                score = (sim + 1.0) / 2.0
            else:
                dist = torch.sqrt(((query_tensor - prototype_tensor) ** 2).sum(dim=1) + 1e-8)
                score = 1.0 / (1.0 + dist.item())
    return float(score)


def run_flux_sanity_with_scorer(
    config: FluxSanityConfig,
    extract_embedding: Callable[[Path], np.ndarray],
    score: Callable[[np.ndarray, np.ndarray], float],
    extract_embeddings: Optional[Callable[[Sequence[Path], int], Sequence[np.ndarray]]] = None,
    batch_size: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
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

    prototypes, query_embeddings = _build_embeddings(
        selected,
        extract_embedding,
        progress_callback,
        extract_embeddings=extract_embeddings,
        batch_size=batch_size,
    )
    trials = _generate_trials_with_scorer(
        selected,
        prototypes,
        query_embeddings,
        score,
        threshold=config.threshold,
        impostors_per_identity=config.impostors_per_identity,
        seed=config.seed,
    )
    validate_trial_pairing(trials, config.impostors_per_identity)
    metrics = summarize_flux_trials(config, selected, trials)
    metrics["threshold_sweep"] = threshold_sweep(
        trials,
        merge_thresholds(config.sweep_thresholds, (config.threshold,)),
    )
    artifacts = write_flux_sanity_artifacts(config.output_dir, metrics, trials)
    metrics["artifacts"] = artifacts
    return metrics


def _run_single_model_benchmark(
    model_type: str,
    base_row: dict,
    flux_config: FluxSanityConfig,
    config: FaceModelBenchmarkConfig,
    runner: ModelRunner,
    progress_callback: Optional[Callable[[str, int, int], None]],
) -> dict:
    try:
        metrics = runner(
            model_type,
            flux_config,
            config.device,
            config.validate_images,
            config.batch_size,
            progress_callback,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report per-model failures.
        if config.strict:
            raise
        return _missing_or_error_row(base_row, "error", str(exc))
    return _comparison_row_from_metrics(base_row, metrics, config.acceptable_frr)


def validate_trial_pairing(trials: Sequence[dict], impostors_per_identity: int) -> None:
    by_query: Dict[str, List[dict]] = {}
    for trial in trials:
        by_query.setdefault(trial["query_identity"], []).append(trial)
        enrollment_parents = {
            Path(path).parent.name
            for path in str(trial.get("enrollment_images", "")).split("|")
            if path
        }
        query_parent = Path(str(trial.get("query_image", ""))).parent.name
        if trial["trial_type"] == "genuine":
            if trial["enrolled_identity"] != trial["query_identity"]:
                raise FluxSanityError("Genuine trial identity mismatch.")
            if enrollment_parents != {trial["query_identity"]}:
                raise FluxSanityError("Genuine enrollment path does not match query identity.")
        elif trial["trial_type"] == "impostor":
            if trial["enrolled_identity"] == trial["query_identity"]:
                raise FluxSanityError("Impostor trial compared an identity to itself.")
            if enrollment_parents != {trial["enrolled_identity"]}:
                raise FluxSanityError("Impostor enrollment path does not match enrolled identity.")
        else:
            raise FluxSanityError(f"Unknown trial type: {trial['trial_type']}")
        if query_parent != trial["query_identity"]:
            raise FluxSanityError("Query image path does not match query identity.")

    for query_identity, rows in by_query.items():
        genuine = [row for row in rows if row["trial_type"] == "genuine"]
        impostor = [row for row in rows if row["trial_type"] == "impostor"]
        impostor_ids = [row["enrolled_identity"] for row in impostor]
        if len(genuine) != 1:
            raise FluxSanityError(f"{query_identity} has {len(genuine)} genuine trials.")
        if len(impostor) != impostors_per_identity:
            raise FluxSanityError(f"{query_identity} has {len(impostor)} impostor trials.")
        if len(impostor_ids) != len(set(impostor_ids)):
            raise FluxSanityError(f"{query_identity} has duplicate impostor identities.")


def select_best_threshold(
    sweep_rows: Sequence[dict],
    acceptable_frr: float = DEFAULT_ACCEPTABLE_FRR,
) -> dict:
    candidates = [
        row for row in sweep_rows
        if row.get("frr") is not None and float(row["frr"]) <= acceptable_frr
    ]
    note = "within_acceptable_frr"
    if not candidates:
        candidates = list(sweep_rows)
        note = "no_threshold_met_acceptable_frr"
    if not candidates:
        return {"threshold": None, "far": None, "frr": None, "note": "no_sweep_rows"}
    best = sorted(
        candidates,
        key=lambda row: (
            float("inf") if row.get("far") is None else float(row["far"]),
            float("inf") if row.get("frr") is None else float(row["frr"]),
            -float(row["threshold"]),
        ),
    )[0]
    return {
        "threshold": best["threshold"],
        "far": best["far"],
        "frr": best["frr"],
        "note": note,
    }


def write_comparison_artifacts(
    output_dir: Path,
    rows: Sequence[dict],
    config: FaceModelBenchmarkConfig,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    _write_comparison_csv(csv_path, rows)
    json_path.write_text(
        json.dumps({
            "dataset_path": str(config.dataset_dir),
            "identities": config.identities if config.identities is not None else "all",
            "seed": config.seed,
            "impostors_per_identity": config.impostors_per_identity,
            "acceptable_frr": config.acceptable_frr,
            "batch_size": config.batch_size,
            "parallel_models": config.parallel_models,
            "models": list(rows),
        }, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_comparison_markdown(rows, config), encoding="utf-8")
    return {
        "comparison": str(markdown_path),
        "comparison_csv": str(csv_path),
        "comparison_json": str(json_path),
    }


def render_comparison_markdown(
    rows: Sequence[dict],
    config: FaceModelBenchmarkConfig,
) -> str:
    lines = [
        "# FLUXSynID Face Model Benchmark",
        "",
        "Report-only comparison. Production thresholds were not changed.",
        "",
        "## Summary",
        "",
        f"- Dataset: `{config.dataset_dir}`",
        f"- Identities: `{config.identities if config.identities is not None else 'all'}`",
        f"- Seed: `{config.seed}`",
        f"- Impostors per identity: `{config.impostors_per_identity}`",
        f"- Acceptable FRR for candidate threshold: `{config.acceptable_frr}`",
        f"- Batch size: `{config.batch_size}`",
        f"- Parallel models: `{config.parallel_models}`",
        "",
        "| Model | Status | Threshold | Genuine Fail | False Accepts | FRR | FAR | Candidate Threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['model_type']} | {row['status']} | {_md_value(row.get('threshold'))} | "
            f"{_md_value(row.get('genuine_failed'))} | {_md_value(row.get('impostor_false_accepts'))} | "
            f"{_md_value(row.get('frr'))} | {_md_value(row.get('far'))} | "
            f"{_md_value(row.get('best_threshold'))} |"
        )
    return "\n".join(lines) + "\n"


def _build_embeddings(
    identities: Sequence[FluxIdentity],
    extract_embedding: Callable[[Path], np.ndarray],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    *,
    extract_embeddings: Optional[Callable[[Sequence[Path], int], Sequence[np.ndarray]]] = None,
    batch_size: int = 1,
) -> Tuple[Dict[str, dict], Dict[str, np.ndarray]]:
    prototypes: Dict[str, dict] = {}
    query_embeddings: Dict[str, np.ndarray] = {}
    total = len(identities)
    jobs: List[Tuple[str, str, Path]] = []
    for item in identities:
        for path in item.enrollment_images:
            jobs.append((item.identity, "enrollment", path))
        jobs.append((item.identity, "query", item.query_image))

    if extract_embeddings is None:
        extract_embeddings = lambda paths, _batch_size: [extract_embedding(path) for path in paths]

    raw: Dict[str, dict] = {
        item.identity: {
            "enrollment": [],
            "query": None,
            "enrollment_images": [str(path) for path in item.enrollment_images],
        }
        for item in identities
    }
    report_every_jobs = 100 * 4
    next_report = report_every_jobs
    completed_jobs = 0

    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        chunk_embeddings = extract_embeddings([path for _, _, path in chunk], batch_size)
        if len(chunk_embeddings) != len(chunk):
            raise FluxSanityError("Batch embedding extractor returned an unexpected number of embeddings.")
        for (identity, kind, _path), embedding in zip(chunk, chunk_embeddings):
            arr = np.asarray(embedding, dtype=np.float32)
            if kind == "enrollment":
                raw[identity]["enrollment"].append(arr)
            else:
                raw[identity]["query"] = arr
        completed_jobs += len(chunk)
        if progress_callback and (completed_jobs >= next_report or completed_jobs == len(jobs)):
            completed_identities = min(completed_jobs // 4, total)
            progress_callback(completed_identities, total)
            while next_report <= completed_jobs:
                next_report += report_every_jobs

    for item in identities:
        enrollment_embeddings = raw[item.identity]["enrollment"]
        if len(enrollment_embeddings) != len(item.enrollment_images):
            raise FluxSanityError(f"Missing enrollment embeddings for {item.identity}.")
        if raw[item.identity]["query"] is None:
            raise FluxSanityError(f"Missing query embedding for {item.identity}.")
        prototypes[item.identity] = {
            "prototype": np.mean(enrollment_embeddings, axis=0),
            "enrollment_images": raw[item.identity]["enrollment_images"],
        }
        query_embeddings[item.identity] = raw[item.identity]["query"]
    return prototypes, query_embeddings


def _generate_trials_with_scorer(
    identities: Sequence[FluxIdentity],
    prototypes: Dict[str, dict],
    query_embeddings: Dict[str, np.ndarray],
    score: Callable[[np.ndarray, np.ndarray], float],
    *,
    threshold: float,
    impostors_per_identity: int,
    seed: int,
) -> List[dict]:
    import random

    identity_ids = [item.identity for item in identities]
    by_id = {item.identity: item for item in identities}
    rng = random.Random(seed)
    trials: List[dict] = []

    for query_identity in sorted(identity_ids):
        item = by_id[query_identity]
        trials.append(_trial_row(
            trial_type="genuine",
            enrolled_identity=query_identity,
            query_identity=query_identity,
            score=score(query_embeddings[query_identity], prototypes[query_identity]["prototype"]),
            threshold=threshold,
            enrollment_images=prototypes[query_identity]["enrollment_images"],
            query_image=str(item.query_image),
        ))

        candidates = [identity for identity in sorted(identity_ids) if identity != query_identity]
        for enrolled_identity in sorted(rng.sample(candidates, impostors_per_identity)):
            trials.append(_trial_row(
                trial_type="impostor",
                enrolled_identity=enrolled_identity,
                query_identity=query_identity,
                score=score(query_embeddings[query_identity], prototypes[enrolled_identity]["prototype"]),
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


def _comparison_row_from_metrics(
    base_row: dict,
    metrics: dict,
    acceptable_frr: float,
) -> dict:
    genuine = metrics["genuine"]
    impostor = metrics["impostor"]
    genuine_trials = genuine["trials"]
    impostor_trials = impostor["trials"]
    best = select_best_threshold(metrics.get("threshold_sweep", []), acceptable_frr)
    return {
        **base_row,
        "status": "completed",
        "identities_tested": metrics["identities_tested"],
        "genuine_trials": genuine_trials,
        "genuine_passed": genuine["passed"],
        "genuine_failed": genuine["failed"],
        "impostor_trials": impostor_trials,
        "impostor_false_accepts": impostor["false_accepts"],
        "frr": round(genuine["failed"] / genuine_trials, 8) if genuine_trials else None,
        "far": round(impostor["false_accepts"] / impostor_trials, 8) if impostor_trials else None,
        "genuine_score_min": genuine["score_stats"]["min"],
        "genuine_score_mean": genuine["score_stats"]["mean"],
        "genuine_score_max": genuine["score_stats"]["max"],
        "impostor_score_min": impostor["score_stats"]["min"],
        "impostor_score_mean": impostor["score_stats"]["mean"],
        "impostor_score_max": impostor["score_stats"]["max"],
        "best_threshold": best["threshold"],
        "best_threshold_far": best["far"],
        "best_threshold_frr": best["frr"],
        "best_threshold_note": best["note"],
        "error": "",
    }


def _missing_or_error_row(base_row: dict, status: str, error: str = "") -> dict:
    return {
        **base_row,
        "status": status,
        "identities_tested": "",
        "genuine_trials": "",
        "genuine_passed": "",
        "genuine_failed": "",
        "impostor_trials": "",
        "impostor_false_accepts": "",
        "frr": "",
        "far": "",
        "genuine_score_min": "",
        "genuine_score_mean": "",
        "genuine_score_max": "",
        "impostor_score_min": "",
        "impostor_score_mean": "",
        "impostor_score_max": "",
        "best_threshold": "",
        "best_threshold_far": "",
        "best_threshold_frr": "",
        "best_threshold_note": "",
        "error": error,
    }


def _write_comparison_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "model_type",
        "model_label",
        "status",
        "checkpoint_available",
        "checkpoint_path",
        "threshold",
        "identities_tested",
        "genuine_trials",
        "genuine_passed",
        "genuine_failed",
        "impostor_trials",
        "impostor_false_accepts",
        "frr",
        "far",
        "genuine_score_min",
        "genuine_score_mean",
        "genuine_score_max",
        "impostor_score_min",
        "impostor_score_mean",
        "impostor_score_max",
        "best_threshold",
        "best_threshold_far",
        "best_threshold_frr",
        "best_threshold_note",
        "output_dir",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _md_value(value) -> str:
    return "" if value is None else str(value)
