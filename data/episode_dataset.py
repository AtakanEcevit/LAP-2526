"""
PyTorch Dataset wrapper for pre-sampled Prototypical episodes.

Enables DataLoader-based parallel image loading instead of
sequential per-image loading in the training loop.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class PrototypicalEpisodeDataset(Dataset):
    """
    Wraps pre-sampled episode paths for DataLoader parallelism.

    Each item returns a preprocessed image tensor and its class index.
    Used for both support and query sets within a single episode.
    """

    def __init__(self, paths_with_labels, dataset):
        """
        Args:
            paths_with_labels: list of (path, class_idx) tuples
            dataset: BiometricDataset instance (provides load_image)
        """
        self.items = paths_with_labels
        self.dataset = dataset

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, class_idx = self.items[idx]
        img = self.dataset.load_image(path)
        return (
            torch.from_numpy(np.ascontiguousarray(img)),
            torch.tensor(class_idx, dtype=torch.long),
        )
