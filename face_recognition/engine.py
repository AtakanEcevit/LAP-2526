"""
InsightFace buffalo_l engine — face detection + ArcFace embedding extraction.

buffalo_l ships two ONNX models:
  - SCRFD-10G  : face detection
  - ArcFace w600k_r50 : 512-d L2-normalised embedding (cosine similarity)

The models are downloaded automatically to ~/.insightface/models/buffalo_l/
on the first call to _load().

Usage:
    engine = InsightFaceEngine()
    emb  = engine.extract_embedding("face.jpg")        # (512,) float32
    res  = engine.compare("a.jpg", "b.jpg")            # dict
    res  = engine.verify("query.jpg", enrolled_embs)   # dict
"""

import cv2
import numpy as np


# Cosine similarity threshold calibrated for buffalo_l on LFW:
#   sim > DEFAULT_THRESHOLD  → same person
DEFAULT_THRESHOLD = 0.35


class NoFaceDetectedError(ValueError):
    """Raised when no face is found in the image."""


class InsightFaceEngine:
    """
    Thin wrapper around insightface.app.FaceAnalysis (buffalo_l).

    Lazy-loads the model on first use so import is cheap.
    Thread-safe for read operations (the ONNX runtime itself is).
    """

    def __init__(
        self,
        det_size: tuple = (640, 640),
        providers: list = None,
    ):
        """
        Args:
            det_size:  Detection input resolution (H, W). Larger = better recall,
                       slower. (640, 640) is the standard for buffalo_l.
            providers: ONNX runtime providers list.
                       Defaults to CUDA if available, else CPU.
        """
        self.det_size = det_size
        self.providers = providers  # None → auto-select in _load()
        self._app = None

    # ── Internal ─────────────────────────────────────────────────────────

    def _load(self):
        try:
            import insightface
        except ImportError as e:
            raise ImportError(
                "insightface is not installed. Run:\n"
                "  pip install insightface onnxruntime\n"
                "(or onnxruntime-gpu for CUDA)"
            ) from e

        providers = self.providers
        if providers is None:
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in available
                    else ["CPUExecutionProvider"]
                )
            except ImportError:
                providers = ["CPUExecutionProvider"]

        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self._app.prepare(ctx_id=0, det_size=self.det_size)
        print(f"[InsightFace] buffalo_l loaded  providers={providers}")

    def _ensure_loaded(self):
        if self._app is None:
            self._load()

    def _to_bgr(self, image_input) -> np.ndarray:
        """
        Accept str path, bytes, file-like, PIL Image, or numpy array.
        Returns a BGR uint8 numpy array (OpenCV convention required by InsightFace).
        """
        # PIL Image — check before bytes/str since PIL objects aren't str/bytes
        try:
            from PIL import Image as _PILImage
            if isinstance(image_input, _PILImage.Image):
                return cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        if isinstance(image_input, np.ndarray):
            img = image_input
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image_input}")
            return img

        # bytes or file-like
        if hasattr(image_input, "read"):
            image_input = image_input.read()
        if not isinstance(image_input, (bytes, bytearray)):
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        buf = np.frombuffer(image_input, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return img

    def _get_best_face(self, bgr: np.ndarray):
        """Run detection and return the face with the largest bounding box."""
        faces = self._app.get(bgr)
        if not faces:
            raise NoFaceDetectedError("No face detected in image")
        # Pick the face with the largest bounding box area
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # ── Public API ───────────────────────────────────────────────────────

    def extract_embedding(self, image_input) -> np.ndarray:
        """
        Detect the largest face and return its ArcFace embedding.

        Returns:
            np.ndarray of shape (512,), float32, L2-normalised.

        Raises:
            NoFaceDetectedError: if no face is found.
        """
        self._ensure_loaded()
        bgr = self._to_bgr(image_input)
        face = self._get_best_face(bgr)
        return face.embedding.astype(np.float32)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity. Works for both already-normalised and raw vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a / na, b / nb))

    def compare(
        self,
        emb_or_img1,
        emb_or_img2,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict:
        """
        Compare two face images (or pre-extracted embeddings) and return a
        match decision based on cosine similarity.

        Args:
            emb_or_img1: image (path/bytes/PIL/ndarray) OR a 1-D embedding array
            emb_or_img2: same as above
            threshold:   cosine similarity cutoff (default 0.35)

        Returns:
            {"match": bool, "score": float, "threshold": float}
        """
        emb1 = (
            emb_or_img1
            if isinstance(emb_or_img1, np.ndarray) and emb_or_img1.ndim == 1
            else self.extract_embedding(emb_or_img1)
        )
        emb2 = (
            emb_or_img2
            if isinstance(emb_or_img2, np.ndarray) and emb_or_img2.ndim == 1
            else self.extract_embedding(emb_or_img2)
        )
        score = self._cosine_sim(emb1, emb2)
        return {
            "match": score >= threshold,
            "score": round(score, 6),
            "threshold": threshold,
        }

    def verify(
        self,
        query,
        enrolled_embeddings: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict:
        """
        Verify a query image (or pre-extracted embedding) against a set of
        enrolled embeddings. Decision is made against the L2-renormalised
        prototype (mean of enrolled embeddings).

        Args:
            query:               image (path/bytes/PIL/ndarray) OR 1-D embedding
            enrolled_embeddings: np.ndarray of shape (N, 512)
            threshold:           cosine similarity cutoff

        Returns:
            {"match": bool, "score": float, "threshold": float}
        """
        query_emb = (
            query
            if isinstance(query, np.ndarray) and query.ndim == 1
            else self.extract_embedding(query)
        )
        prototype = enrolled_embeddings.mean(axis=0)
        score = self._cosine_sim(query_emb, prototype)
        return {
            "match": score >= threshold,
            "score": round(score, 6),
            "threshold": threshold,
        }

    def info(self) -> dict:
        return {
            "model": "buffalo_l",
            "embedding_dim": 512,
            "det_size": self.det_size,
            "loaded": self._app is not None,
        }
