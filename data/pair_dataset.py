"""
PyTorch Dataset wrapper for pre-sampled Siamese pairs.

Enables DataLoader-based parallel image loading instead of
sequential per-pair loading in the training loop.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class SiamesePairDataset(Dataset):
    """
    Wraps pre-sampled pairs for DataLoader parallelism.

    Each item returns two preprocessed image tensors and a label.
    Image loading/preprocessing/augmentation happens in DataLoader
    worker processes rather than the main training thread.
    """

    def __init__(self, pairs, dataset):
        """
        Args:
            pairs: list of (path1, path2, label) from PairSampler.sample_epoch()
            dataset: BiometricDataset instance (provides load_image)
        """
        self.pairs = pairs
        self.dataset = dataset

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = self.dataset.load_image(path1)
        img2 = self.dataset.load_image(path2)
        return (
            torch.from_numpy(np.ascontiguousarray(img1)),
            torch.from_numpy(np.ascontiguousarray(img2)),
            torch.tensor(label, dtype=torch.float32),
        )
