import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation.flux_face_benchmark as benchmark  # noqa: E402
from evaluation.flux_face_benchmark import (  # noqa: E402
    ALL_MODELS,
    FaceModelBenchmarkConfig,
    expand_model_types,
    run_face_model_benchmark,
    run_flux_sanity_with_scorer,
    score_embedding_against_prototype,
    select_best_threshold,
    validate_trial_pairing,
)
from evaluation.flux_hybrid_sanity import (  # noqa: E402
    FluxSanityConfig,
    FluxSanityError,
    run_flux_sanity,
)


def test_discover_registered_face_models_returns_face_models_only():
    models = benchmark.discover_registered_face_models()

    assert models[:2] == ("siamese", "prototypical")
    assert "hybrid" in models
    assert "facenet_proto" in models
    assert "facenet_contrastive_proto" in models
    assert "siamese_signature" not in models


def test_expand_model_types_all_is_deterministic():
    assert expand_model_types((ALL_MODELS,)) == benchmark.discover_registered_face_models()
    assert expand_model_types(("facenet_proto",)) == ("facenet_proto",)

    with pytest.raises(FluxSanityError):
        expand_model_types((ALL_MODELS, "hybrid"))


def test_missing_checkpoint_is_reported_without_running_model(tmp_path, monkeypatch):
    monkeypatch.setitem(
        benchmark.MODEL_REGISTRY,
        ("face", "missing_flux_model"),
        {
            "config": str(tmp_path / "missing.yaml"),
            "checkpoint": str(tmp_path / "missing.pth"),
            "threshold": 0.5,
        },
    )

    result = run_face_model_benchmark(
        FaceModelBenchmarkConfig(
            dataset_dir=tmp_path,
            output_dir=tmp_path / "out",
            model_types=("missing_flux_model",),
        ),
        model_runner=_runner_should_not_run,
    )

    assert result["models"][0]["status"] == "missing_artifact"
    assert result["models"][0]["checkpoint_available"] is False
    assert (tmp_path / "out" / "comparison.csv").exists()


def test_benchmark_writes_per_model_and_aggregate_outputs(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "flux"
    for idx in range(6):
        _write_flux_identity(dataset_dir, f"person_{idx:03d}")
    _register_fake_model(monkeypatch, tmp_path, "fake_face_a", 0.75)
    _register_fake_model(monkeypatch, tmp_path, "fake_face_b", 0.85)

    result = run_face_model_benchmark(
        FaceModelBenchmarkConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "bench",
            model_types=("fake_face_a", "fake_face_b"),
            identities=4,
            seed=42,
            impostors_per_identity=2,
            batch_size=2,
            parallel_models=2,
        ),
        model_runner=_fake_model_runner,
    )

    assert result["batch_size"] == 2
    assert result["parallel_models"] == 2
    assert [row["model_type"] for row in result["models"]] == ["fake_face_a", "fake_face_b"]
    assert all(row["status"] == "completed" for row in result["models"])
    assert result["models"][0]["genuine_trials"] == 4
    assert result["models"][0]["impostor_trials"] == 8
    assert (tmp_path / "bench" / "comparison.md").exists()
    assert (tmp_path / "bench" / "comparison.csv").exists()
    assert (tmp_path / "bench" / "fake_face_a" / "results.csv").exists()
    assert (tmp_path / "bench" / "fake_face_b" / "metrics.json").exists()


def test_validate_trial_pairing_rejects_self_impostor(tmp_path):
    _write_flux_identity(tmp_path, "person_001")
    row = {
        "trial_type": "impostor",
        "enrolled_identity": "person_001",
        "query_identity": "person_001",
        "enrollment_images": str(tmp_path / "person_001" / "person_001_f_doc.jpg"),
        "query_image": str(tmp_path / "person_001" / "person_001_f_live_0_p_d1.jpg"),
    }

    with pytest.raises(FluxSanityError, match="itself"):
        validate_trial_pairing([row], impostors_per_identity=1)


def test_select_best_threshold_prefers_lowest_far_with_acceptable_frr():
    best = select_best_threshold(
        [
            {"threshold": 0.8, "far": 0.10, "frr": 0.0},
            {"threshold": 0.9, "far": 0.02, "frr": 0.015},
            {"threshold": 0.95, "far": 0.01, "frr": 0.08},
        ],
        acceptable_frr=0.02,
    )

    assert best["threshold"] == 0.9
    assert best["far"] == 0.02
    assert best["note"] == "within_acceptable_frr"


def test_score_embedding_against_prototype_matches_model_family_logic():
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    prototype = np.asarray([1.0, 0.0], dtype=np.float32)

    facenet_engine = _FakeEngine("hybrid")
    assert score_embedding_against_prototype(facenet_engine, query, prototype) == pytest.approx(1.0)

    proto_engine = _FakeEngine("prototypical", distance_type="euclidean")
    assert score_embedding_against_prototype(proto_engine, query, prototype) == pytest.approx(0.9999, abs=1e-3)

    siamese_engine = _FakeEngine("siamese", classifier=torch.nn.Linear(2, 1))
    torch.nn.init.zeros_(siamese_engine.model.classifier.weight)
    torch.nn.init.zeros_(siamese_engine.model.classifier.bias)
    assert score_embedding_against_prototype(siamese_engine, query, prototype) == pytest.approx(0.5)


def test_run_flux_sanity_sweep_includes_runtime_threshold(tmp_path):
    dataset_dir = tmp_path / "flux"
    for idx in range(5):
        _write_flux_identity(dataset_dir, f"person_{idx:03d}")

    metrics = run_flux_sanity(
        FluxSanityConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "out",
            identities=4,
            threshold=0.777,
            impostors_per_identity=2,
        ),
        _fake_embedding,
    )

    assert 0.777 in [row["threshold"] for row in metrics["threshold_sweep"]]


def test_run_flux_sanity_with_scorer_batches_embedding_extraction(tmp_path):
    dataset_dir = tmp_path / "flux"
    for idx in range(5):
        _write_flux_identity(dataset_dir, f"person_{idx:03d}")
    batch_sizes = []

    def extract_embeddings(paths, batch_size):
        batch_sizes.append(len(paths))
        return [_fake_embedding(path) for path in paths]

    metrics = run_flux_sanity_with_scorer(
        FluxSanityConfig(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "out",
            identities=4,
            threshold=0.75,
            impostors_per_identity=2,
        ),
        _fake_embedding,
        _fake_score,
        extract_embeddings=extract_embeddings,
        batch_size=5,
    )

    assert metrics["genuine"]["trials"] == 4
    assert batch_sizes == [5, 5, 5, 1]


class _FakeEngine:
    def __init__(self, model_type, distance_type=None, classifier=None):
        self.model_type = model_type
        self.device = "cpu"
        self.model = type("FakeModel", (), {})()
        if distance_type:
            self.model.distance_type = distance_type
        if classifier is not None:
            self.model.classifier = classifier


def _register_fake_model(monkeypatch, tmp_path, model_type, threshold):
    checkpoint = tmp_path / f"{model_type}.pth"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setitem(
        benchmark.MODEL_REGISTRY,
        ("face", model_type),
        {
            "config": str(tmp_path / f"{model_type}.yaml"),
            "checkpoint": str(checkpoint),
            "threshold": threshold,
        },
    )


def _fake_model_runner(model_type, config, device, validate_images, batch_size, progress_callback):
    return run_flux_sanity(config, _fake_embedding, _adapt_progress(model_type, progress_callback))


def _adapt_progress(model_type, progress_callback):
    if progress_callback is None:
        return None
    return lambda done, total: progress_callback(model_type, done, total)


def _runner_should_not_run(*args, **kwargs):
    raise AssertionError("runner should not be called for missing artifacts")


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


def _fake_score(query_embedding, prototype):
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    prototype = np.asarray(prototype, dtype=np.float32)
    return float(np.dot(query_embedding, prototype))
