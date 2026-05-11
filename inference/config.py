"""
Inference configuration — maps every trained model to its config and checkpoint.

Exposes both Siamese and Prototypical variants for each modality so the
caller can choose which architecture to use at runtime.
"""

import os

# Resolve project root (one level up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(rel_path: str) -> str:
    """Convert a project-relative path to absolute."""
    return os.path.join(_PROJECT_ROOT, rel_path)


# ── Model Registry ──────────────────────────────────────────────────────
# Each entry: config YAML path, checkpoint path, default decision threshold.
# Thresholds are initial defaults; they can be overridden or calibrated
# from the EER threshold stored in evaluation results.

MODEL_REGISTRY = {
    # ── Signature ────────────────────────────────────────────────────────
    ("signature", "siamese"): {
        "config": _abs("configs/siamese_signature.yaml"),
        "checkpoint": _abs("results/siamese_signature_cedar/checkpoints/best.pth"),
        "threshold": 0.65,
    },
    ("signature", "prototypical"): {
        "config": _abs("configs/proto_signature.yaml"),
        "checkpoint": _abs("results/proto_signature_cedar/checkpoints/best.pth"),
        "threshold": 0.65,
    },

    # ── Face ─────────────────────────────────────────────────────────────
    ("face", "siamese"): {
        "config": _abs("configs/siamese_face.yaml"),
        "checkpoint": _abs("results/siamese_face_att/checkpoints/best.pth"),
        "threshold": 0.65,
    },
    ("face", "prototypical"): {
        "config": _abs("configs/proto_face.yaml"),
        "checkpoint": _abs("results/proto_face_att/checkpoints/best.pth"),
        "threshold": 0.65,
    },
    ("face", "hybrid"): {
        "config": _abs("configs/hybrid_face.yaml"),
        "checkpoint": r"C:\Downloads\Model3 Model1_in_gelişmiş_hali\model3\best_face_model.pth",
        "threshold": 0.3000000119,
    },
    ("face", "facenet_proto"): {
        "config": _abs("configs/facenet_proto_face.yaml"),
        "checkpoint": _abs("results/facenet_proto_face/checkpoints/best.pth"),
        "threshold": 0.47,
    },
    ("face", "facenet_contrastive_proto"): {
        "config": _abs("configs/facenet_contrastive_proto_face.yaml"),
        "checkpoint": _abs(
            "results/facenet_contrastive_proto_face/checkpoints/best.pth"
        ),
        "threshold": 0.800884,
        "threshold_key": "far_threshold",
    },

    # ── Fingerprint ──────────────────────────────────────────────────────
    ("fingerprint", "siamese"): {
        "config": _abs("configs/siamese_fingerprint.yaml"),
        "checkpoint": _abs("results/siamese_fingerprint_socofing/checkpoints/best.pth"),
        "threshold": 0.65,
    },
    ("fingerprint", "prototypical"): {
        "config": _abs("configs/proto_fingerprint.yaml"),
        "checkpoint": _abs("results/proto_fingerprint_socofing/checkpoints/best.pth"),
        "threshold": 0.65,
    },
}


# ── Valid Values ─────────────────────────────────────────────────────────

VALID_MODALITIES = {"signature", "face", "fingerprint"}
VALID_MODEL_TYPES = {
    "siamese",
    "prototypical",
    "hybrid",
    "facenet_proto",
    "facenet_contrastive_proto",
}


# ── Image Sizes (H, W) per Modality ─────────────────────────────────────
# Canonical source is data.preprocessing; re-exported here for convenience.

from data.preprocessing import IMAGE_SIZES  # noqa: E402


# ── Enrollment Storage ───────────────────────────────────────────────────

DEFAULT_ENROLLMENT_PATH = _abs("data/enrollments.json")


# ── Auto-load Calibrated Thresholds ──────────────────────────────────────
# If calibrate_thresholds.py has been run, override defaults with EER-optimal
# values. Falls back silently to the hardcoded 0.5 if not calibrated yet.

def _load_calibrated_thresholds():
    """Load calibrated thresholds from JSON file if it exists."""
    import json
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibrated_thresholds.json")
    if os.path.exists(cal_path):
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Convert "modality,model_type" keys back to tuple keys
            return {
                tuple(k.split(",")): v
                for k, v in raw.items()
            }
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


# Apply calibrated thresholds at import time
_calibrated = _load_calibrated_thresholds()
for _key, _threshold in _calibrated.items():
    if _key in MODEL_REGISTRY:
        MODEL_REGISTRY[_key]["threshold"] = _threshold
