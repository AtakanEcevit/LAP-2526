"""
InsightFace buffalo_l + Siamese projection head.

Architecture:
    Image ──► InsightFace SCRFD (detect + align, 112×112)
                      │ 512-d, L2-norm (ArcFace w600k_r50)
                      ▼
               SiameseHead (trainable)
                 512 → 256 → embedding_dim, L2-norm
              (shared weights, twin branches)

Training mode  : forward(feat1, feat2)  — inputs are pre-extracted 512-d features
Inference mode : extract_embedding(img) — full pipeline via InsightFace + head
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

_BUFFALO_ONNX = os.path.join(
    os.path.expanduser("~"), ".insightface", "models", "buffalo_l", "w600k_r50.onnx"
)
# Normalization constants for w600k_r50 (from insightface model metadata)
_MEAN = 127.5
_STD  = 127.5

# EER-calibrated cosine similarity threshold on AT&T validation set (val_ratio=0.2, seed=0).
# Genuine: mean=0.76 / Impostor: mean=0.058 — EER=12.3%, Acc=88.3% at this value.
CALIBRATED_THRESHOLD = 0.58


class NoFaceDetectedError(ValueError):
    """Raised when no face can be found in the image."""


# ── Pre-processing (training path) ───────────────────────────────────────────

def preprocess_for_backbone(image_input) -> np.ndarray:
    """
    Convert any supported image to a (3, 112, 112) float32 array in [-1, 1].

    Used during training where images are already face-cropped (AT&T, LFW, etc.).
    No face detection — the whole image is assumed to contain a face.

    Supports: str path, bytes, file-like, PIL Image, numpy array.
    Grayscale images are converted to 3-channel BGR.
    """
    try:
        from PIL import Image as _PIL
        if isinstance(image_input, _PIL.Image):
            image_input = np.array(image_input.convert("RGB"))
            image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
    except ImportError:
        pass

    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        if hasattr(image_input, "read"):
            image_input = image_input.read()
        buf = np.frombuffer(image_input, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")

    # Grayscale → BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32)
    arr = (arr - _MEAN) / _STD          # → [-1, 1]
    arr = arr.transpose(2, 0, 1)        # HWC → CHW
    return arr                           # (3, 112, 112)


# ── Siamese head ──────────────────────────────────────────────────────────────

class SiameseHead(nn.Module):
    """
    Trainable projection: 512-d backbone features → embedding_dim-d, L2-norm.

    Layout: Linear(512→256) → BN → ReLU → Dropout → Linear(256→dim) → BN → L2
    """

    def __init__(self, in_dim: int = 512, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


# ── Full model ────────────────────────────────────────────────────────────────

class InsightFaceSiamese(nn.Module):
    """
    Siamese network: buffalo_l backbone (frozen) + SiameseHead (trainable).

    Training:
        out = model(feat1, feat2)   # feat: (B, 512) pre-extracted tensors
        loss = contrastive(out['distance'], labels)

    Inference (real photos — uses InsightFace face detection):
        emb = model.extract_embedding("face.jpg")   # (embedding_dim,)
        res = model.compare("a.jpg", "b.jpg")       # dict

    Inference (pre-cropped face images — no detection, same as training path):
        emb = model.extract_embedding("face.jpg", detect=False)

    Loading from checkpoint:
        model = InsightFaceSiamese.from_checkpoint("path/to/best.pth")
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        onnx_path: str = None,
    ):
        super().__init__()
        self.head = SiameseHead(512, embedding_dim, dropout)
        self.embedding_dim = embedding_dim
        self._onnx_path = onnx_path or _BUFFALO_ONNX
        self._session = None        # ONNX session for direct backbone (no detection)
        self._input_name = None
        self._if_app = None         # InsightFace FaceAnalysis (with detector)

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, onnx_path: str = None) -> "InsightFaceSiamese":
        """
        Load a trained model from a checkpoint saved by train_att_siamese.py.

        Args:
            checkpoint_path: path to best.pth (must contain 'head_state_dict' and 'config')
            onnx_path:       override buffalo_l ONNX path (auto-detected if None)

        Returns:
            InsightFaceSiamese in eval mode with loaded weights.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        model = cls(
            embedding_dim=cfg.get("embedding_dim", 128),
            dropout=cfg.get("dropout", 0.3),
            onnx_path=onnx_path,
        )
        model.head.load_state_dict(ckpt["head_state_dict"])
        model.eval()
        print(
            f"[InsightFaceSiamese] Loaded from {checkpoint_path}  "
            f"(epoch={ckpt.get('epoch', '?')}, "
            f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f})"
        )
        return model

    # ── Lazy loaders ─────────────────────────────────────────────────────

    def _load_session(self):
        """Load the raw ONNX backbone session (used for direct/no-detection path)."""
        import onnxruntime as ort
        if not os.path.exists(self._onnx_path):
            import insightface
            app = insightface.app.FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0)
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self._onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def _ensure_session(self):
        if self._session is None:
            self._load_session()

    def _load_if_app(self):
        """Load InsightFace FaceAnalysis (SCRFD detector + ArcFace)."""
        try:
            import insightface
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
            self._if_app = insightface.app.FaceAnalysis(
                name="buffalo_l", providers=providers
            )
            self._if_app.prepare(ctx_id=0, det_size=(640, 640))
            print("[InsightFaceSiamese] buffalo_l FaceAnalysis loaded")
        except ImportError as e:
            raise ImportError(
                "insightface not installed. Run: pip install insightface onnxruntime"
            ) from e

    def _ensure_if_app(self):
        if self._if_app is None:
            self._load_if_app()

    # ── Image helpers ─────────────────────────────────────────────────────

    def _to_bgr(self, image_input) -> np.ndarray:
        """Convert any image input format to a BGR uint8 numpy array."""
        try:
            from PIL import Image as _PILImage
            if isinstance(image_input, _PILImage.Image):
                return cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        if isinstance(image_input, np.ndarray):
            img = image_input
            if img.ndim == 3 and img.shape[2] == 3:
                # Assume RGB if float or if caller passes RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image_input}")
            return img

        if hasattr(image_input, "read"):
            image_input = image_input.read()
        if not isinstance(image_input, (bytes, bytearray)):
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        buf = np.frombuffer(image_input, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return img

    # ── Backbone feature extraction ───────────────────────────────────────

    def _run_backbone_direct(self, image_input) -> np.ndarray:
        """
        Simple-resize path (no face detection). Returns (512,) features.
        Same preprocessing as training — suitable for pre-cropped face images.
        """
        self._ensure_session()
        arr = preprocess_for_backbone(image_input)      # (3, 112, 112)
        batch = arr[np.newaxis].astype(np.float32)
        out = self._session.run(None, {self._input_name: batch})[0]
        return out[0].astype(np.float32)                # (512,)

    def _run_backbone_with_detection(self, image_input) -> np.ndarray:
        """
        InsightFace detection + alignment path. Returns (512,) L2-normalised features.
        Detects the largest face, aligns it, then extracts ArcFace embedding.

        Raises:
            NoFaceDetectedError: if no face found.
        """
        self._ensure_if_app()
        bgr = self._to_bgr(image_input)
        faces = self._if_app.get(bgr)
        if not faces:
            raise NoFaceDetectedError("No face detected in image")
        # Pick face with the largest bounding box
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return face.normed_embedding.astype(np.float32)  # (512,) already L2-norm

    # ── Training forward (pre-extracted features) ─────────────────────────

    def forward_once(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, 512) backbone features → (B, embedding_dim) L2-norm embedding."""
        return self.head(feat)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> dict:
        """
        Training forward pass.

        Args:
            feat1, feat2: (B, 512) pre-extracted backbone features
        Returns:
            dict with 'emb1', 'emb2', 'distance' (Euclidean)
        """
        emb1 = self.forward_once(feat1)
        emb2 = self.forward_once(feat2)
        dist = torch.sqrt(((emb1 - emb2) ** 2).sum(dim=1) + 1e-8)
        return {"emb1": emb1, "emb2": emb2, "distance": dist}

    # ── Inference ─────────────────────────────────────────────────────────

    def extract_embedding(self, image_input, detect: bool = True) -> np.ndarray:
        """
        Extract a projected embedding from an image.

        Args:
            image_input: file path, bytes, PIL Image, or numpy array.
            detect:      If True (default), use InsightFace face detection +
                         alignment — correct for real/general photos.
                         If False, use simple resize (same as training path) —
                         use for pre-cropped face images (AT&T, LFW crops, etc.).

        Returns:
            np.ndarray of shape (embedding_dim,), L2-normalised, float32.

        Raises:
            NoFaceDetectedError: when detect=True and no face is found.
        """
        self.eval()
        feat = (
            self._run_backbone_with_detection(image_input)
            if detect
            else self._run_backbone_direct(image_input)
        )
        feat_t = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            emb = self.head(feat_t)
        return emb.squeeze(0).numpy().astype(np.float32)

    # ── Similarity ────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity. Safe for both normalised and raw vectors."""
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a / na, b / nb))

    def compare(
        self,
        img1,
        img2,
        threshold: float = CALIBRATED_THRESHOLD,
        detect: bool = True,
    ) -> dict:
        """
        Compare two face images and return a match decision.

        Args:
            img1, img2:  image inputs (path / bytes / PIL / ndarray)
            threshold:   cosine similarity cutoff.  Defaults to the EER-calibrated
                         value (0.58) derived from the AT&T validation set.
                         Increase for stricter (fewer false accepts).
            detect:      use InsightFace face detection (True = recommended for
                         real photos; False = pre-cropped images like AT&T).

        Returns:
            {"match": bool, "score": float, "threshold": float}
        """
        emb1 = self.extract_embedding(img1, detect=detect)
        emb2 = self.extract_embedding(img2, detect=detect)
        score = self._cosine(emb1, emb2)
        return {
            "match": score >= threshold,
            "score": round(score, 6),
            "threshold": threshold,
        }

    def verify(
        self,
        query,
        enrolled_embeddings: np.ndarray,
        threshold: float = CALIBRATED_THRESHOLD,
        detect: bool = True,
    ) -> dict:
        """
        Verify a query image against a set of enrolled embeddings.

        The decision is made against the L2-renormalised prototype
        (mean of enrolled embeddings).

        Args:
            query:               image (path / bytes / PIL / ndarray) OR
                                 pre-extracted (embedding_dim,) array.
            enrolled_embeddings: np.ndarray of shape (N, embedding_dim).
            threshold:           cosine similarity cutoff (default: EER-calibrated 0.58).
            detect:              use InsightFace detection for query image.

        Returns:
            {"match": bool, "score": float, "threshold": float}
        """
        if isinstance(query, np.ndarray) and query.ndim == 1:
            query_emb = query
        else:
            query_emb = self.extract_embedding(query, detect=detect)

        prototype = enrolled_embeddings.mean(axis=0)
        score = self._cosine(query_emb, prototype)
        return {
            "match": score >= threshold,
            "score": round(score, 6),
            "threshold": threshold,
        }

    def info(self) -> dict:
        return {
            "model": "insightface_siamese_att",
            "embedding_dim": self.embedding_dim,
            "threshold": CALIBRATED_THRESHOLD,
            "backbone": "buffalo_l / w600k_r50",
            "if_app_loaded": self._if_app is not None,
            "onnx_session_loaded": self._session is not None,
        }
