"""
Tests for data loading pipeline fixes (Scope 1).

Validates:
  - Bug 1: Augmentation output always has (C, H, W) shape
  - Bug 2: Train/val transform toggling (tested via trainer integration)
  - Bug 3: Signature preprocessing uses CLAHE (non-binary output)
  - Bug 4: Normalization is robust and deterministic
  - Bug 5: Image cache respects max_cache_size bound

Uses synthetic dummy images — no real dataset required.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import albumentations as A
from data.base_loader import BiometricDataset


# ── Minimal concrete dataset for testing ────────────────────────────────

class DummyDataset(BiometricDataset):
    """Concrete dataset with synthetic grayscale images for testing."""

    IMG_SIZE = (96, 96)

    def __init__(self, num_subjects=3, images_per_subject=5, **kwargs):
        self._num_subjects = num_subjects
        self._images_per_subject = images_per_subject
        self._tmpdir = tempfile.mkdtemp()
        super().__init__(self._tmpdir, **kwargs)

    def _load_data(self):
        """Create synthetic grayscale images on disk."""
        for subj in range(1, self._num_subjects + 1):
            subj_dir = os.path.join(self._tmpdir, f"subj_{subj}")
            os.makedirs(subj_dir, exist_ok=True)
            self.data[subj] = {'genuine': [], 'forgery': []}
            for i in range(self._images_per_subject):
                # Create a gradient image (not binary!) with varied values
                img = np.random.randint(0, 256, (self.IMG_SIZE[0], self.IMG_SIZE[1]),
                                        dtype=np.uint8)
                path = os.path.join(subj_dir, f"img_{i}.png")
                Image.fromarray(img, mode='L').save(path)
                self.data[subj]['genuine'].append(path)

    def _preprocess(self, image):
        """Simple resize, no binarization."""
        img = np.array(image, dtype=np.uint8)
        import cv2
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)


# ── Bug 1: Shape Tests ──────────────────────────────────────────────────

class TestAugmentationShape:
    """Verify load_image always returns (C, H, W) regardless of transform."""

    def test_no_transform_shape(self):
        """Without any transform, output should be (1, H, W)."""
        ds = DummyDataset(transform=None)
        path = ds.data[1]['genuine'][0]
        img = ds.load_image(path)
        assert img.ndim == 3, f"Expected 3D, got {img.ndim}D"
        assert img.shape[0] == 1, f"Expected channel=1, got shape {img.shape}"
        assert img.shape == (1, 96, 96)

    def test_empty_transform_shape(self):
        """Empty Compose (validation transform) must still return (1, H, W)."""
        ds = DummyDataset(transform=A.Compose([]))
        path = ds.data[1]['genuine'][0]
        img = ds.load_image(path)
        assert img.ndim == 3, f"Expected 3D, got {img.ndim}D with shape {img.shape}"
        assert img.shape[0] == 1, f"Expected channel=1, got shape {img.shape}"

    def test_training_transform_shape(self):
        """Training augmentation must return (1, H, W)."""
        transform = A.Compose([
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                     scale=(0.9, 1.1), rotate=(-5, 5),
                     border_mode=0, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ])
        ds = DummyDataset(transform=transform)
        path = ds.data[1]['genuine'][0]
        # Run multiple times (augmentation is stochastic)
        for _ in range(5):
            ds._image_cache.clear()  # Force re-augmentation
            img = ds.load_image(path)
            assert img.ndim == 3, f"Expected 3D, got {img.ndim}D"
            assert img.shape[0] == 1, f"Expected channel=1, got {img.shape}"

    def test_shape_consistent_across_loads(self):
        """Repeated loads of the same image must have consistent shape."""
        ds = DummyDataset(transform=A.Compose([
            A.RandomBrightnessContrast(p=1.0),
        ]))
        path = ds.data[1]['genuine'][0]
        shapes = set()
        for _ in range(10):
            img = ds.load_image(path)
            shapes.add(img.shape)
        assert len(shapes) == 1, f"Inconsistent shapes across loads: {shapes}"


# ── Bug 3: Signature Preprocessing ──────────────────────────────────────

class TestSignaturePreprocessing:
    """Verify CEDAR preprocessing uses CLAHE, not Otsu binarization."""

    def test_cedar_not_binary(self):
        """CEDAR preprocessed image should have >2 unique pixel values."""
        from data.signature_loader import CEDARDataset

        # Check that the preprocessing function doesn't binarize
        # We test the _preprocess method directly with a synthetic image
        ds_class = CEDARDataset.__new__(CEDARDataset)
        ds_class.IMG_SIZE = (155, 220)

        # Create a gradient image with many unique values
        gradient = np.linspace(50, 200, 155 * 220).reshape(155, 220).astype(np.uint8)
        pil_img = Image.fromarray(gradient, mode='L')

        result = ds_class._preprocess(pil_img)
        result_arr = np.array(result)
        unique_values = len(np.unique(result_arr))

        # CLAHE should preserve many unique values; Otsu would give <=2
        assert unique_values > 2, \
            f"Preprocessing appears to be binary ({unique_values} unique values). " \
            f"Expected CLAHE with many grayscale levels."

    def test_bhsig260_not_binary(self):
        """BHSig260 preprocessed image should have >2 unique pixel values."""
        from data.signature_loader import BHSig260Dataset

        ds_class = BHSig260Dataset.__new__(BHSig260Dataset)
        ds_class.IMG_SIZE = (155, 220)

        gradient = np.linspace(30, 230, 155 * 220).reshape(155, 220).astype(np.uint8)
        pil_img = Image.fromarray(gradient, mode='L')

        result = ds_class._preprocess(pil_img)
        result_arr = np.array(result)
        unique_values = len(np.unique(result_arr))

        assert unique_values > 2, \
            f"BHSig260 preprocessing appears binary ({unique_values} unique values)."


# ── Bug 4: Normalization ────────────────────────────────────────────────

class TestNormalization:
    """Verify normalization produces correct value ranges."""

    def test_values_in_0_1_range(self):
        """All loaded image values must be in [0.0, 1.0]."""
        ds = DummyDataset(transform=None)
        for subj in ds.data:
            for path in ds.data[subj]['genuine']:
                img = ds.load_image(path)
                assert img.min() >= 0.0, f"Negative pixel value: {img.min()}"
                assert img.max() <= 1.0, f"Pixel value > 1: {img.max()}"

    def test_dtype_is_float32(self):
        """Loaded images must be float32."""
        ds = DummyDataset(transform=None)
        path = ds.data[1]['genuine'][0]
        img = ds.load_image(path)
        assert img.dtype == np.float32, f"Expected float32, got {img.dtype}"

    def test_all_black_image(self):
        """An all-black image (zeros) should normalize correctly."""
        ds = DummyDataset(transform=None)
        # Replace one image with all-black
        path = ds.data[1]['genuine'][0]
        black = np.zeros((96, 96), dtype=np.uint8)
        Image.fromarray(black, mode='L').save(path)
        ds._image_cache.clear()

        img = ds.load_image(path)
        assert img.max() == 0.0, "All-black image should remain all zeros"
        assert img.shape == (1, 96, 96)

    def test_all_white_image(self):
        """An all-white image (255) should normalize to 1.0."""
        ds = DummyDataset(transform=None)
        path = ds.data[1]['genuine'][0]
        white = np.full((96, 96), 255, dtype=np.uint8)
        Image.fromarray(white, mode='L').save(path)
        ds._image_cache.clear()

        img = ds.load_image(path)
        assert abs(img.max() - 1.0) < 1e-6, f"All-white should normalize to 1.0, got {img.max()}"
        assert img.shape == (1, 96, 96)


# ── Bug 5: Cache Bounding ──────────────────────────────────────────────

class TestCacheBound:
    """Verify image cache respects max_cache_size."""

    def test_cache_does_not_exceed_limit(self):
        """Cache should evict oldest entries when full."""
        # Small limit for testing
        ds = DummyDataset(num_subjects=5, images_per_subject=10,
                          max_cache_size=10, transform=None)

        # Load all 50 images
        count = 0
        for subj in ds.data:
            for path in ds.data[subj]['genuine']:
                ds.load_image(path)
                count += 1

        assert count == 50, f"Expected 50 images, loaded {count}"
        assert len(ds._image_cache) <= 10, \
            f"Cache size {len(ds._image_cache)} exceeds limit 10"

    def test_default_cache_size(self):
        """Default max_cache_size should be 5000."""
        ds = DummyDataset()
        assert ds._max_cache_size == 5000
"""
Test the data loading pipeline for correctness after Scope 1 fixes.
"""
