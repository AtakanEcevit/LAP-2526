"""
Base dataset class for biometric verification.
All modality-specific loaders inherit from this.
"""

import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BiometricDataset(Dataset, ABC):
    """
    Abstract base class for biometric datasets.
    
    Each dataset organizes data by subjects (identities). For each subject,
    there are genuine samples and (optionally) forgery samples.
    
    Subclasses must implement:
        - _load_data(): populate self.data dict
        - _preprocess(image): apply modality-specific preprocessing
    """

    def __init__(self, root_dir, split="train", transform=None, k_shot=5,
                 max_cache_size=5000):
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train', 'val', or 'test'
            transform: Optional Albumentations transform pipeline
            k_shot: Number of support samples per class (for few-shot)
            max_cache_size: Maximum number of preprocessed images to cache in RAM
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.k_shot = k_shot

        # data[subject_id] = {
        #     'genuine': [path1, path2, ...],
        #     'forgery': [path1, path2, ...]  (optional)
        # }
        self.data = {}
        self.subjects = []
        self.all_samples = []  # List of (path, subject_id, is_genuine)
        self._image_cache = {}  # Cache preprocessed images in RAM
        self._max_cache_size = max_cache_size

        self._load_data()
        self._build_index()

    @abstractmethod
    def _load_data(self):
        """Load dataset file paths into self.data dict."""
        pass

    @abstractmethod
    def _preprocess(self, image):
        """Apply modality-specific preprocessing to a PIL Image."""
        pass

    def _build_index(self):
        """Build a flat index of all samples for iteration."""
        self.subjects = sorted(self.data.keys())
        self.all_samples = []
        for subj in self.subjects:
            for path in self.data[subj].get('genuine', []):
                self.all_samples.append((path, subj, True))
            for path in self.data[subj].get('forgery', []):
                self.all_samples.append((path, subj, False))

    def load_image(self, path):
        """Load and preprocess a single image, using memory cache if available."""
        if path in self._image_cache:
            img = self._image_cache[path]
        else:
            img = Image.open(path).convert('L')  # Grayscale by default
            img = self._preprocess(img)
            
            # Convert to numpy if still PIL
            if isinstance(img, Image.Image):
                img = np.array(img, dtype=np.float32)

            # Always cast to float32 and normalize to [0, 1]
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0

            # Add channel dimension if needed: (H, W) -> (1, H, W)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)

            # Evict oldest entry if cache is full
            if len(self._image_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._image_cache))
                del self._image_cache[oldest_key]
            self._image_cache[path] = img

        # Apply transforms on the fly (not cached!)
        if self.transform:
            transformed = self.transform(image=img.transpose(1, 2, 0) if img.ndim == 3 else img)
            img = transformed['image']
            # Guarantee (C, H, W) output regardless of albumentations behavior
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)   # (H,W) → (1,H,W)
            elif img.ndim == 3 and img.shape[-1] in (1, 3):
                img = img.transpose(2, 0, 1)         # (H,W,C) → (C,H,W)

        return img

    def get_subject_samples(self, subject_id, genuine_only=True, max_samples=None):
        """Get sample paths for a specific subject."""
        samples = list(self.data[subject_id].get('genuine', []))
        if not genuine_only:
            samples.extend(self.data[subject_id].get('forgery', []))
        if max_samples:
            samples = samples[:max_samples]
        return samples

    def get_num_subjects(self):
        """Return total number of subjects/identities."""
        return len(self.subjects)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        path, subject_id, is_genuine = self.all_samples[idx]
        image = self.load_image(path)
        return image, subject_id, is_genuine

    def split_subjects(self, train_ratio=0.6, val_ratio=0.2):
        """
        Split subjects into train/val/test sets.
        Returns dicts with the same structure as self.data.
        """
        n = len(self.subjects)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        rng = np.random.RandomState(42)
        shuffled = rng.permutation(self.subjects).tolist()

        train_subj = shuffled[:n_train]
        val_subj = shuffled[n_train:n_train + n_val]
        test_subj = shuffled[n_train + n_val:]

        return (
            {s: self.data[s] for s in train_subj},
            {s: self.data[s] for s in val_subj},
            {s: self.data[s] for s in test_subj},
        )
