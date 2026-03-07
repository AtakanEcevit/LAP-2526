"""
Tests for DataLoader-based training pipeline.

Validates:
  - SiamesePairDataset returns correct tensor shapes and dtypes
  - PrototypicalEpisodeDataset returns correct tensor shapes and dtypes
  - PairSampler.sample_epoch produces correct number of pairs
  - EpisodeSampler.sample_epoch produces correct number of episodes
  - DataLoader iteration works with num_workers=0
  - Trainer epoch methods work end-to-end with DataLoader
"""

import os
import tempfile
import numpy as np
import torch
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from data.base_loader import BiometricDataset
from data.pair_dataset import SiamesePairDataset
from data.episode_dataset import PrototypicalEpisodeDataset
from data.samplers import PairSampler, EpisodeSampler


# ── Minimal concrete dataset for testing ────────────────────────────────

class DummyDataset(BiometricDataset):
    """Concrete dataset with synthetic grayscale images for testing."""

    IMG_SIZE = (96, 96)

    def __init__(self, num_subjects=5, images_per_subject=10, **kwargs):
        self._num_subjects = num_subjects
        self._images_per_subject = images_per_subject
        self._tmpdir = tempfile.mkdtemp()
        super().__init__(self._tmpdir, **kwargs)

    def _load_data(self):
        """Create synthetic grayscale images on disk."""
        for subj_idx in range(self._num_subjects):
            subj_id = f"subject_{subj_idx:03d}"
            subj_dir = os.path.join(self.root_dir, subj_id)
            os.makedirs(subj_dir, exist_ok=True)

            paths = []
            for img_idx in range(self._images_per_subject):
                img_path = os.path.join(subj_dir, f"img_{img_idx:03d}.png")
                # Deterministic pixel values for reproducibility
                arr = np.random.RandomState(subj_idx * 100 + img_idx).randint(
                    0, 255, self.IMG_SIZE, dtype=np.uint8
                )
                Image.fromarray(arr, mode='L').save(img_path)
                paths.append(img_path)

            self.data[subj_id] = {
                'genuine': paths[:self._images_per_subject],
                'forgery': [],
            }

    def _preprocess(self, image):
        """Simple resize, no binarization."""
        return image.resize(self.IMG_SIZE, Image.Resampling.BILINEAR)


# ── SiamesePairDataset Tests ───────────────────────────────────────────

class TestSiamesePairDataset:
    """Verify SiamesePairDataset works with DataLoader."""

    def _make_dataset_and_pairs(self, num_pairs=16):
        ds = DummyDataset(num_subjects=3, images_per_subject=8)
        data = {s: ds.data[s] for s in ds.subjects}
        sampler = PairSampler(data, batch_size=num_pairs)
        pairs = sampler.sample_batch()
        return ds, pairs

    def test_getitem_shapes(self):
        """Each item should return (C,H,W), (C,H,W), scalar tensors."""
        ds, pairs = self._make_dataset_and_pairs(8)
        pair_ds = SiamesePairDataset(pairs, ds)

        img1, img2, label = pair_ds[0]
        assert img1.ndim == 3, f"Expected 3D tensor, got {img1.ndim}D"
        assert img1.shape[0] == 1, f"Expected 1 channel, got {img1.shape[0]}"
        assert img2.shape == img1.shape
        assert label.shape == (), f"Label should be scalar, got {label.shape}"

    def test_getitem_dtypes(self):
        """Images should be float32, labels should be float32."""
        ds, pairs = self._make_dataset_and_pairs(4)
        pair_ds = SiamesePairDataset(pairs, ds)

        img1, img2, label = pair_ds[0]
        assert img1.dtype == torch.float32
        assert img2.dtype == torch.float32
        assert label.dtype == torch.float32

    def test_len(self):
        """Dataset length should match number of pairs."""
        ds, pairs = self._make_dataset_and_pairs(16)
        pair_ds = SiamesePairDataset(pairs, ds)
        assert len(pair_ds) == 16

    def test_dataloader_iteration(self):
        """DataLoader should produce batched tensors with correct shapes."""
        ds, pairs = self._make_dataset_and_pairs(16)
        pair_ds = SiamesePairDataset(pairs, ds)
        loader = DataLoader(pair_ds, batch_size=8, shuffle=False, num_workers=0)

        batch = next(iter(loader))
        images1, images2, labels = batch
        assert images1.shape[0] == 8
        assert images1.ndim == 4  # (B, C, H, W)
        assert images2.shape == images1.shape
        assert labels.shape == (8,)

    def test_values_in_valid_range(self):
        """All pixel values should be in [0, 1]."""
        ds, pairs = self._make_dataset_and_pairs(4)
        pair_ds = SiamesePairDataset(pairs, ds)
        img1, img2, _ = pair_ds[0]
        assert img1.min() >= 0.0 and img1.max() <= 1.0
        assert img2.min() >= 0.0 and img2.max() <= 1.0


# ── PrototypicalEpisodeDataset Tests ───────────────────────────────────

class TestPrototypicalEpisodeDataset:
    """Verify PrototypicalEpisodeDataset works with DataLoader."""

    def _make_episode(self):
        ds = DummyDataset(num_subjects=5, images_per_subject=10)
        data = {s: ds.data[s] for s in ds.subjects}
        sampler = EpisodeSampler(data, n_way=3, k_shot=2, q_query=2)
        support_paths, query_paths = sampler.sample_episode()
        return ds, support_paths, query_paths

    def test_getitem_shapes(self):
        """Each item should return (C,H,W) tensor and long scalar."""
        ds, support_paths, _ = self._make_episode()
        episode_ds = PrototypicalEpisodeDataset(support_paths, ds)

        img, class_idx = episode_ds[0]
        assert img.ndim == 3
        assert img.shape[0] == 1
        assert class_idx.dtype == torch.long

    def test_dataloader_iteration(self):
        """DataLoader loads full episode in one batch."""
        ds, support_paths, _ = self._make_episode()
        episode_ds = PrototypicalEpisodeDataset(support_paths, ds)
        loader = DataLoader(
            episode_ds, batch_size=len(support_paths),
            shuffle=False, num_workers=0,
        )

        images, labels = next(iter(loader))
        # 3-way × 2-shot = 6 support images
        assert images.shape[0] == 6
        assert images.ndim == 4
        assert labels.shape[0] == 6


# ── Sampler Epoch Pre-Sampling Tests ───────────────────────────────────

class TestSamplerEpoch:
    """Verify sample_epoch produces correct quantities."""

    def test_pair_sampler_epoch_count(self):
        """sample_epoch should return num_iterations * batch_size pairs."""
        ds = DummyDataset(num_subjects=3, images_per_subject=8)
        data = {s: ds.data[s] for s in ds.subjects}
        batch_size = 8
        num_iterations = 5
        sampler = PairSampler(data, batch_size=batch_size)
        pairs = sampler.sample_epoch(num_iterations)
        assert len(pairs) == batch_size * num_iterations

    def test_episode_sampler_epoch_count(self):
        """sample_epoch should return exactly num_episodes episodes."""
        ds = DummyDataset(num_subjects=5, images_per_subject=10)
        data = {s: ds.data[s] for s in ds.subjects}
        sampler = EpisodeSampler(data, n_way=3, k_shot=2, q_query=2)
        episodes = sampler.sample_epoch(10)
        assert len(episodes) == 10
        # Each episode is (support_paths, query_paths) tuple
        for support, query in episodes:
            assert len(support) == 3 * 2  # n_way * k_shot
            assert len(query) == 3 * 2    # n_way * q_query

    def test_pair_epoch_all_valid_labels(self):
        """All pairs should have labels in {0, 1}."""
        ds = DummyDataset(num_subjects=3, images_per_subject=8)
        data = {s: ds.data[s] for s in ds.subjects}
        sampler = PairSampler(data, batch_size=16)
        pairs = sampler.sample_epoch(3)
        for _, _, label in pairs:
            assert label in (0, 1)
