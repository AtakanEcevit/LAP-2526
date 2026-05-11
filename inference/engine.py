"""
Core verification engine.

Loads a trained checkpoint (Siamese, Prototypical, or FaceNet-style) and exposes
embedding extraction, pair comparison, and enrollment-based verification.
"""

import yaml
import torch
import numpy as np

from models.hybrid_face import HybridFaceModel, preprocess_hybrid_face
from models.siamese import SiameseNetwork
from inference.validation import validate_image
from models.prototypical import PrototypicalNetwork
from inference.config import MODEL_REGISTRY, VALID_MODALITIES, VALID_MODEL_TYPES
from inference.preprocessing import preprocess_image
from utils import get_device


FACENET_STYLE_MODEL_TYPES = {
    "hybrid",
    "facenet_proto",
    "facenet_contrastive_proto",
}


def _float_value(value) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _checkpoint_threshold(
    checkpoint: dict,
    explicit,
    default: float,
    threshold_key: str = None,
) -> float:
    if explicit is not None:
        return _float_value(explicit)
    for key in (threshold_key, "val_threshold", "threshold"):
        if not key:
            continue
        if key in checkpoint and checkpoint[key] is not None:
            return _float_value(checkpoint[key])
    return _float_value(default)


class VerificationEngine:
    """
    High-level interface for biometric verification inference.

    Usage:
        engine = VerificationEngine()
        engine.load("signature", "siamese")  # or "prototypical"

        # Extract an embedding
        emb = engine.extract_embedding("path/to/image.png")

        # Compare two images directly
        result = engine.compare("img1.png", "img2.png")
        # → {"match": True, "score": 0.93, "threshold": 0.5}

        # Verify against enrolled prototype
        result = engine.verify_against_prototype(query_tensor, prototype_vector)
    """

    def __init__(self):
        self.model = None
        self.model_type = None   # "siamese", "prototypical", or FaceNet-style
        self.modality = None     # "signature", "face", "fingerprint"
        self.threshold = 0.5
        self.device = None
        self._loaded = False

    def load(self, modality: str, model_type: str = "siamese",
             device=None, threshold: float = None):
        """
        Load a trained model for the given modality and architecture.

        Args:
            modality:   "signature", "face", or "fingerprint"
            model_type: one of the registered model types
            device:     torch device (auto-detected if None)
            threshold:  override the default decision threshold
        """
        if modality not in VALID_MODALITIES:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Must be one of: {sorted(VALID_MODALITIES)}"
            )
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Must be one of: {sorted(VALID_MODEL_TYPES)}"
            )

        key = (modality, model_type)
        if key not in MODEL_REGISTRY:
            raise ValueError(f"No model registered for {key}")

        entry = MODEL_REGISTRY[key]
        config_path = entry["config"]
        checkpoint_path = entry["checkpoint"]

        # Load YAML config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Determine device
        self.device = device or get_device()
        self.modality = modality
        self.model_type = model_type
        self.threshold = threshold if threshold is not None else entry["threshold"]

        # Build model architecture
        backbone = config["model"].get("backbone", "resnet")
        emb_dim = config["model"].get("embedding_dim", 128)
        in_channels = config["model"].get("in_channels", 1)

        if model_type in FACENET_STYLE_MODEL_TYPES:
            if modality != "face":
                raise ValueError(
                    f"{model_type} model is only supported for face modality."
                )
            self.model = None
        elif model_type == "siamese":
            self.model = SiameseNetwork(
                backbone=backbone,
                embedding_dim=emb_dim,
                pretrained=False,       # weights come from checkpoint
                in_channels=in_channels,
            )
        else:
            distance = config["model"].get("distance", "euclidean")
            self.model = PrototypicalNetwork(
                backbone=backbone,
                embedding_dim=emb_dim,
                pretrained=False,
                in_channels=in_channels,
                distance=distance,
            )

        # Load trained weights
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if model_type in FACENET_STYLE_MODEL_TYPES:
            self.model = HybridFaceModel.from_checkpoint(checkpoint)
            self.threshold = _checkpoint_threshold(
                checkpoint,
                threshold,
                entry["threshold"],
                entry.get("threshold_key"),
            )
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        msg = f"[Engine] Loaded {model_type}/{modality} from {checkpoint_path} -> {self.device}"
        print(msg.encode("ascii", errors="replace").decode("ascii"))

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "No model loaded. Call engine.load(modality, model_type) first."
            )

    # ── Embedding Extraction ─────────────────────────────────────────────

    def extract_embedding(self, image_input, validate: bool = True) -> np.ndarray:
        """
        Extract a model embedding from an image.

        Args:
            image_input: file path (str), raw bytes, PIL.Image, or numpy array.
            validate:    run input validation before processing.

        Returns:
            numpy array of shape (embedding_dim,), L2-normalized.

        Raises:
            ValueError: if validation hard-check fails.
        """
        self._ensure_loaded()

        if validate:
            val = validate_image(image_input, self.modality)
            if not val.passed:
                raise ValueError(
                    f"Image validation failed: {'; '.join(val.warnings)}"
                )

        tensor = self._preprocess_for_model(image_input).to(self.device)

        with torch.no_grad():
            embedding = self.model.get_embedding(tensor)

        return embedding.cpu().numpy().squeeze()

    # ── Pair Comparison ──────────────────────────────────────────────────

    def compare(self, image1_input, image2_input,
                validate: bool = True) -> dict:
        """
        Compare two images and return a match/no-match decision.

        Args:
            image1_input: first image (path, bytes, PIL, or ndarray)
            image2_input: second image (same formats)
            validate:     run input validation before processing.

        Returns:
            dict with keys:
                match (bool):      True if score > threshold
                score (float):     similarity score in [0, 1]
                threshold (float): decision threshold used
                validation (dict): per-image validation results
        """
        self._ensure_loaded()

        # Validate inputs
        val1 = validate_image(image1_input, self.modality) if validate else None
        val2 = validate_image(image2_input, self.modality) if validate else None

        tensor1 = self._preprocess_for_model(image1_input).to(self.device)
        tensor2 = self._preprocess_for_model(image2_input).to(self.device)

        with torch.no_grad():
            if self.model_type in {"siamese", *FACENET_STYLE_MODEL_TYPES}:
                # Use cosine similarity of L2-normalized embeddings directly.
                # This is loss-function agnostic (works with both BCE and
                # contrastive training) and correctly scores identical
                # images at 1.0.
                emb1 = self.model.get_embedding(tensor1)
                emb2 = self.model.get_embedding(tensor2)
                cosine_sim = torch.mm(emb1, emb2.t()).squeeze().item()
                score = (cosine_sim + 1.0) / 2.0  # map [-1,1] → [0,1]
            else:
                # Prototypical: respect model's distance type
                emb1 = self.model.get_embedding(tensor1)
                emb2 = self.model.get_embedding(tensor2)
                distance_type = getattr(self.model, 'distance_type', 'euclidean')
                if distance_type == 'cosine':
                    sim = torch.mm(emb1, emb2.t()).squeeze().item()
                    score = (sim + 1.0) / 2.0  # map [-1, 1] → [0, 1]
                else:
                    dist = torch.sqrt(
                        ((emb1 - emb2) ** 2).sum(dim=1) + 1e-8
                    )
                    score = 1.0 / (1.0 + dist.item())

        result = {
            "match": score > self.threshold,
            "score": round(score, 6),
            "threshold": self.threshold,
        }

        if validate:
            result["validation"] = {
                "image1": val1.to_dict(),
                "image2": val2.to_dict(),
            }

        return result

    # ── Prototype-Based Verification ─────────────────────────────────────

    def verify_against_prototype(
        self, query_input, enrolled_embeddings: np.ndarray,
        validate: bool = True,
    ) -> dict:
        """
        Verify a query image against the mean of enrolled embeddings.

        Args:
            query_input:         image (path, bytes, PIL, or ndarray)
            enrolled_embeddings: numpy array of shape (N, embedding_dim) —
                                 the enrolled reference samples.
            validate:            run input validation before processing.

        Returns:
            dict with match, score, threshold, and validation.
        """
        self._ensure_loaded()

        # Validate query input
        val = validate_image(query_input, self.modality) if validate else None

        # Compute prototype as mean of enrolled embeddings
        prototype = np.mean(enrolled_embeddings, axis=0)
        prototype_tensor = (
            torch.from_numpy(prototype).float().unsqueeze(0).to(self.device)
        )

        # Get query embedding
        query_tensor = self._preprocess_for_model(query_input).to(self.device)

        with torch.no_grad():
            query_emb = self.model.get_embedding(query_tensor)

            if self.model_type in FACENET_STYLE_MODEL_TYPES:
                sim = torch.mm(query_emb, prototype_tensor.t()).squeeze().item()
                score = (sim + 1.0) / 2.0
            elif self.model_type == "siamese":
                # Use the classifier head for Siamese
                diff = torch.abs(query_emb - prototype_tensor)
                similarity = torch.sigmoid(
                    self.model.classifier(diff)
                ).squeeze().item()
                score = similarity
            else:
                # Prototypical: respect model's distance type
                distance_type = getattr(self.model, 'distance_type', 'euclidean')
                if distance_type == 'cosine':
                    sim = torch.mm(query_emb, prototype_tensor.t()).squeeze().item()
                    score = (sim + 1.0) / 2.0
                else:
                    dist = torch.sqrt(
                        ((query_emb - prototype_tensor) ** 2).sum(dim=1) + 1e-8
                    )
                    score = 1.0 / (1.0 + dist.item())

        result = {
            "match": score > self.threshold,
            "score": round(score, 6),
            "threshold": self.threshold,
        }

        if validate:
            result["validation"] = val.to_dict()

        return result

    # ── Info ─────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return current engine state."""
        return {
            "loaded": self._loaded,
            "modality": self.modality,
            "model_type": self.model_type,
            "threshold": self.threshold,
            "device": str(self.device) if self.device else None,
        }

    def _preprocess_for_model(self, image_input) -> torch.Tensor:
        if self.model_type in FACENET_STYLE_MODEL_TYPES:
            return preprocess_hybrid_face(image_input)
        return preprocess_image(image_input, self.modality)
