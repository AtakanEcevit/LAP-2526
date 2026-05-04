"""
InsightFace buffalo_l + Siamese projection head.

Architecture:
    Image ──► buffalo_l (frozen ONNX, 112×112)
                      │ 512-d, L2-norm
                      ▼
               SiameseHead (trainable)
                 512 → 256 → embedding_dim, L2-norm
              (shared weights, twin branches)

Training mode  : forward(feat1, feat2)  — inputs are pre-extracted 512-d features
Inference mode : extract_embedding(img) — full pipeline via ONNX + head
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


# ── Pre-processing ────────────────────────────────────────────────────────

def preprocess_for_backbone(image_input) -> np.ndarray:
    """
    Convert any supported image to a (3, 112, 112) float32 array in [-1, 1].

    Supports: str path, bytes, file-like, PIL Image, numpy array.
    Grayscale images are converted to 3-channel RGB.
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


# ── Siamese head ──────────────────────────────────────────────────────────

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


# ── Full model ────────────────────────────────────────────────────────────

class InsightFaceSiamese(nn.Module):
    """
    Siamese network: buffalo_l backbone (frozen) + SiameseHead (trainable).

    Training:
        out = model(feat1, feat2)   # feat: (B, 512) pre-extracted tensors
        loss = contrastive(out['distance'], labels)

    Inference:
        emb = model.extract_embedding("face.jpg")   # (embedding_dim,)
        res = model.compare("a.jpg", "b.jpg")       # dict
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
        self._session = None    # lazy-loaded

    # ── Lazy ONNX session ─────────────────────────────────────────────

    def _load_session(self):
        import onnxruntime as ort
        if not os.path.exists(self._onnx_path):
            # Trigger download
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

    # ── Training forward (pre-extracted features) ─────────────────────

    def forward_once(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, 512) backbone features → (B, embedding_dim) L2-norm embedding."""
        return self.head(feat)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> dict:
        """
        Training forward.

        Args:
            feat1, feat2: (B, 512) pre-extracted backbone features
        Returns:
            dict with 'emb1', 'emb2', 'distance'
        """
        emb1 = self.forward_once(feat1)
        emb2 = self.forward_once(feat2)
        dist = torch.sqrt(((emb1 - emb2) ** 2).sum(dim=1) + 1e-8)
        return {"emb1": emb1, "emb2": emb2, "distance": dist}

    # ── Inference (full image → embedding) ────────────────────────────

    def _run_backbone(self, image_input) -> np.ndarray:
        """image → (512,) backbone feature via ONNX."""
        self._ensure_session()
        arr = preprocess_for_backbone(image_input)          # (3,112,112)
        batch = arr[np.newaxis].astype(np.float32)          # (1,3,112,112)
        out = self._session.run(None, {self._input_name: batch})[0]
        return out[0]   # (512,)

    def extract_embedding(self, image_input) -> np.ndarray:
        """Full pipeline: image → backbone → head → (embedding_dim,) L2-norm."""
        self.eval()
        feat = self._run_backbone(image_input)
        feat_t = torch.from_numpy(feat).unsqueeze(0)        # (1, 512)
        with torch.no_grad():
            emb = self.head(feat_t)
        return emb.squeeze(0).numpy()                       # (embedding_dim,)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a / na, b / nb))

    def compare(self, img1, img2, threshold: float = 0.5) -> dict:
        """Compare two images. Returns match decision and cosine similarity."""
        emb1 = self.extract_embedding(img1)
        emb2 = self.extract_embedding(img2)
        score = self._cosine(emb1, emb2)
        return {"match": score >= threshold, "score": round(score, 6), "threshold": threshold}
