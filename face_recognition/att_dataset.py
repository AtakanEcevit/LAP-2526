"""
AT&T (ORL) face dataset for InsightFace Siamese training.

Pipeline:
  1. Scan AT&T directory for all s*/  subject folders
  2. Pre-extract 512-d backbone features for all 400 images (once, ~3s on CPU)
  3. Optionally cache to <att_root>/.feature_cache/buffalo_l.npz
  4. Build (feat1, feat2, label) pair dataset:
       label=1  → same identity   (genuine)
       label=0  → different       (impostor)
"""

import os
import glob
from itertools import combinations

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from face_recognition.insightface_siamese import preprocess_for_backbone


# ── Feature extraction ────────────────────────────────────────────────────

def load_att_subjects(att_root: str) -> dict:
    """
    Scan AT&T folder structure.

    Returns:
        {subject_id (int): [abs_path, ...], ...}
    """
    subjects = {}
    for subj_dir in sorted(os.listdir(att_root)):
        if not subj_dir.startswith("s"):
            continue
        subj_path = os.path.join(att_root, subj_dir)
        if not os.path.isdir(subj_path):
            continue
        try:
            subj_id = int(subj_dir[1:])
        except ValueError:
            continue
        imgs = sorted(
            p for ext in ("*.pgm", "*.png", "*.jpg", "*.jpeg", "*.bmp")
            for p in glob.glob(os.path.join(subj_path, ext))
        )
        if imgs:
            subjects[subj_id] = imgs
    return subjects


def extract_features(
    subjects: dict,
    onnx_path: str,
    batch_size: int = 32,
    cache_file: str = None,
) -> dict:
    """
    Pre-extract buffalo_l features for all images.

    Args:
        subjects:   {subject_id: [path, ...]}
        onnx_path:  path to w600k_r50.onnx
        batch_size: ONNX batch size
        cache_file: optional .npz path; loaded if it exists, saved otherwise

    Returns:
        {abs_path: (512,) np.float32}
    """
    if cache_file and os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        cache = {str(k): v for k, v in data.items()}
        print(f"[ATT] Loaded {len(cache)} cached features from {cache_file}")
        return cache

    import onnxruntime as ort
    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    all_paths = [p for paths in subjects.values() for p in paths]
    all_paths.sort()

    print(f"[ATT] Extracting features for {len(all_paths)} images...")
    features = {}
    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i : i + batch_size]
        batch = np.stack([preprocess_for_backbone(p) for p in batch_paths]).astype(np.float32)
        out = session.run(None, {input_name: batch})[0]     # (B, 512)
        for path, feat in zip(batch_paths, out):
            features[path] = feat.astype(np.float32)
        print(f"  {min(i + batch_size, len(all_paths))}/{len(all_paths)}", end="\r")
    print()

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(cache_file, **features)
        print(f"[ATT] Feature cache saved to {cache_file}")

    return features


# ── Pair dataset ──────────────────────────────────────────────────────────

class ATTSiamesePairDataset(Dataset):
    """
    Dataset of (feat1, feat2, label) pairs for contrastive training.

    Positives: all C(k,2) within-subject combinations
    Negatives: random cross-subject pairs, matched to positive count
    """

    def __init__(
        self,
        subjects: dict,
        feature_cache: dict,
        neg_ratio: float = 1.0,
        seed: int = 42,
    ):
        """
        Args:
            subjects:      {subject_id: [path, ...]} — subset for this split
            feature_cache: {path: (512,) array}      — features for ALL images
            neg_ratio:     negatives per positive (1.0 = balanced)
            seed:          RNG seed for reproducible negative sampling
        """
        self.pairs = []
        self._build(subjects, feature_cache, neg_ratio, seed)

    def _build(self, subjects, cache, neg_ratio, seed):
        # Positives
        pos_pairs = []
        subject_of = {}
        all_cached = []
        for subj, paths in subjects.items():
            valid = [p for p in paths if p in cache]
            for p in valid:
                subject_of[p] = subj
                all_cached.append(p)
            for p1, p2 in combinations(valid, 2):
                pos_pairs.append((cache[p1], cache[p2], 1))

        # Negatives
        rng = np.random.default_rng(seed)
        n_neg = int(len(pos_pairs) * neg_ratio)
        neg_pairs = []
        attempts = 0
        max_attempts = n_neg * 20
        while len(neg_pairs) < n_neg and attempts < max_attempts:
            i, j = rng.integers(0, len(all_cached), size=2)
            if i != j:
                p1, p2 = all_cached[i], all_cached[j]
                if subject_of[p1] != subject_of[p2]:
                    neg_pairs.append((cache[p1], cache[p2], 0))
            attempts += 1

        self.pairs = pos_pairs + neg_pairs
        rng.shuffle(self.pairs)

        n_pos = sum(1 for *_, l in self.pairs if l == 1)
        print(
            f"[Dataset] {len(self.pairs)} pairs  "
            f"(pos={n_pos}, neg={len(self.pairs)-n_pos})"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1, f2, label = self.pairs[idx]
        return (
            torch.from_numpy(f1.copy()),
            torch.from_numpy(f2.copy()),
            torch.tensor(label, dtype=torch.float32),
        )


# ── Split helpers ─────────────────────────────────────────────────────────

def split_subjects(subjects: dict, val_ratio: float = 0.2, seed: int = 0):
    """
    Split subject dict into train/val by subject identity (no data leakage).

    Returns:
        (train_subjects, val_subjects)  — both are dicts of same format
    """
    rng = np.random.default_rng(seed)
    ids = sorted(subjects.keys())
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train = {k: v for k, v in subjects.items() if k not in val_ids}
    val   = {k: v for k, v in subjects.items() if k in val_ids}
    return train, val
