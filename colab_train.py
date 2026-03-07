#!/usr/bin/env python3
"""
==========================================================================
  Biometric Few-Shot Verification — Google Colab Training Script
  Trains Siamese & Prototypical Networks on Signatures, Faces, Fingerprints
  
  INSTRUCTIONS:
    1. Upload data_raw.zip to your Google Drive root
    2. Open this script in Google Colab (or paste cells into a notebook)
    3. Set Runtime → Change runtime type → GPU (T4)
    4. Run all cells
    5. Download results.zip when done
==========================================================================
"""

# ── Cell 1: Setup & Dependencies ─────────────────────────────────────────
import subprocess, sys

def install_deps():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'torch', 'torchvision', 'opencv-python-headless', 'albumentations',
        'scikit-learn', 'matplotlib', 'seaborn', 'pyyaml', 'tqdm'])

install_deps()

# ── Cell 2: Mount Google Drive & Extract Data ─────────────────────────────
import os, zipfile, shutil

def setup_data():
    """Extract data_raw.zip and auto-discover dataset paths."""
    zip_path = 'data_raw.zip'
    
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found in current directory!")
        print("Make sure you copied it: !cp /content/drive/MyDrive/data_raw.zip .")
        return {}
    
    os.makedirs('data/raw', exist_ok=True)
    print("Extracting datasets...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall('data/raw')
    
    # Print directory tree (depth <= 4)
    print("\nExtracted directory structure:")
    for root, dirs, files in os.walk('data/raw'):
        depth = root.replace('data/raw', '').count(os.sep)
        if depth < 4:
            indent = ' ' * 2 * depth
            print(f'{indent}{os.path.basename(root)}/ ({len(files)} files, {len(dirs)} dirs)')
    
    # Auto-discover dataset paths by searching for marker folders/files
    discovered = {}
    
    for root, dirs, files in os.walk('data/raw'):
        basename = os.path.basename(root)
        # CEDAR: look for full_org folder
        if basename == 'full_org':
            cedar_root = os.path.dirname(root)
            discovered['cedar'] = cedar_root
            print(f"\n[AUTO-DETECT] CEDAR signatures at: {cedar_root}")
        # ATT: look for folder named 's1' inside a directory with many s* folders
        if basename == 's1' and os.path.isdir(root):
            att_root = os.path.dirname(root)
            # Verify it has multiple s* directories
            s_dirs = [d for d in os.listdir(att_root) if d.startswith('s') and os.path.isdir(os.path.join(att_root, d))]
            if len(s_dirs) >= 10:
                discovered['att'] = att_root
                print(f"[AUTO-DETECT] ATT Faces at: {att_root}")
        # SOCOFing: look for Real folder
        if basename == 'Real' and os.path.isdir(root):
            parent = os.path.dirname(root)
            # Verify it also has Altered
            if os.path.isdir(os.path.join(parent, 'Altered')):
                discovered['socofing'] = parent
                print(f"[AUTO-DETECT] SOCOFing at: {parent}")
        # LFW: look for lfw-funneled or lfw as a directory with person-name subdirs
        if basename in ('lfw-funneled', 'lfw', 'lfw_funneled'):
            if os.path.isdir(root) and len(dirs) > 100:
                discovered['lfw'] = root
                print(f"[AUTO-DETECT] LFW at: {root}")
    
    if not discovered:
        print("\n[WARNING] Could not auto-detect any datasets! Check the zip structure.")
    
    return discovered

discovered_paths = setup_data()

# ── Cell 3: Check GPU ─────────────────────────────────────────────────────
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] No GPU detected — training will be slow on CPU!")
print(f"[Device] {device}")

# ── Cell 4: All Project Modules (inline) ──────────────────────────────────
import time, random, yaml, warnings
import numpy as np
import cv2
from PIL import Image
from abc import ABC, abstractmethod

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import albumentations as A

warnings.filterwarnings('ignore', category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
# DATA: Base Loader
# ═══════════════════════════════════════════════════════════════════════════

class BiometricDataset(ABC):
    def __init__(self, root_dir, split="train", transform=None, k_shot=5):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.k_shot = k_shot
        self.data = {}
        self.subjects = []
        self.all_samples = []
        self._image_cache = {}
        self._load_data()
        self._build_index()

    @abstractmethod
    def _load_data(self): pass

    @abstractmethod
    def _preprocess(self, image): pass

    def _build_index(self):
        self.subjects = sorted(self.data.keys())
        self.all_samples = []
        for subj in self.subjects:
            for path in self.data[subj].get('genuine', []):
                self.all_samples.append((path, subj, True))
            for path in self.data[subj].get('forgery', []):
                self.all_samples.append((path, subj, False))

    def load_image(self, path):
        if path in self._image_cache:
            img = self._image_cache[path]
        else:
            img = Image.open(path).convert('L')
            img = self._preprocess(img)
            if isinstance(img, Image.Image):
                img = np.array(img, dtype=np.float32)
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            self._image_cache[path] = img

        if self.transform:
            transformed = self.transform(image=img.transpose(1, 2, 0) if img.ndim == 3 else img)
            img = transformed['image']
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.transpose(2, 0, 1)
        return img

    def get_subject_samples(self, subject_id, genuine_only=True, max_samples=None):
        samples = list(self.data[subject_id].get('genuine', []))
        if not genuine_only:
            samples.extend(self.data[subject_id].get('forgery', []))
        if max_samples:
            samples = samples[:max_samples]
        return samples

    def get_num_subjects(self):
        return len(self.subjects)

    def __len__(self):
        return len(self.all_samples)

    def split_subjects(self, train_ratio=0.6, val_ratio=0.2):
        n = len(self.subjects)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        np.random.seed(42)
        shuffled = np.random.permutation(self.subjects).tolist()
        train_subj = shuffled[:n_train]
        val_subj = shuffled[n_train:n_train + n_val]
        test_subj = shuffled[n_train + n_val:]
        return (
            {s: self.data[s] for s in train_subj},
            {s: self.data[s] for s in val_subj},
            {s: self.data[s] for s in test_subj},
        )

# ═══════════════════════════════════════════════════════════════════════════
# DATA: Signature Loader (CEDAR)
# ═══════════════════════════════════════════════════════════════════════════

class CEDARDataset(BiometricDataset):
    IMG_SIZE = (155, 220)

    def _load_data(self):
        genuine_dir = os.path.join(self.root_dir, 'full_org')
        forgery_dir = os.path.join(self.root_dir, 'full_forg')
        if os.path.exists(genuine_dir):
            for fname in sorted(os.listdir(genuine_dir)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')): continue
                parts = fname.replace('.', '_').split('_')
                try: writer_id = int(parts[1])
                except (IndexError, ValueError): continue
                if writer_id not in self.data:
                    self.data[writer_id] = {'genuine': [], 'forgery': []}
                self.data[writer_id]['genuine'].append(os.path.join(genuine_dir, fname))
        if os.path.exists(forgery_dir):
            for fname in sorted(os.listdir(forgery_dir)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')): continue
                parts = fname.replace('.', '_').split('_')
                try: writer_id = int(parts[1])
                except (IndexError, ValueError): continue
                if writer_id not in self.data:
                    self.data[writer_id] = {'genuine': [], 'forgery': []}
                self.data[writer_id]['forgery'].append(os.path.join(forgery_dir, fname))
        print(f"[CEDAR] Loaded {len(self.data)} writers, "
              f"{sum(len(v['genuine']) for v in self.data.values())} genuine, "
              f"{sum(len(v['forgery']) for v in self.data.values())} forgery")

    def _preprocess(self, image):
        img = np.array(image, dtype=np.uint8)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(img) < 127: img = 255 - img
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)

# ═══════════════════════════════════════════════════════════════════════════
# DATA: Face Loaders (ATT, LFW)
# ═══════════════════════════════════════════════════════════════════════════

class ATTFaceDataset(BiometricDataset):
    IMG_SIZE = (105, 105)

    def _load_data(self):
        for subj_dir in sorted(os.listdir(self.root_dir)):
            subj_path = os.path.join(self.root_dir, subj_dir)
            if not os.path.isdir(subj_path) or not subj_dir.startswith('s'): continue
            try: subj_id = int(subj_dir[1:])
            except ValueError: subj_id = subj_dir
            self.data[subj_id] = {'genuine': [], 'forgery': []}
            for fname in sorted(os.listdir(subj_path)):
                if fname.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp')):
                    self.data[subj_id]['genuine'].append(os.path.join(subj_path, fname))
        print(f"[ATT/ORL] Loaded {len(self.data)} subjects, "
              f"{sum(len(v['genuine']) for v in self.data.values())} images")

    def _preprocess(self, image):
        img = np.array(image, dtype=np.uint8)
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)


class LFWDataset(BiometricDataset):
    IMG_SIZE = (105, 105)

    def __init__(self, root_dir, min_images=5, **kwargs):
        self.min_images = min_images
        super().__init__(root_dir, **kwargs)

    def _load_data(self):
        for subj_dir in sorted(os.listdir(self.root_dir)):
            subj_path = os.path.join(self.root_dir, subj_dir)
            if not os.path.isdir(subj_path): continue
            images = []
            for fname in sorted(os.listdir(subj_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(subj_path, fname))
            if len(images) >= self.min_images:
                self.data[subj_dir] = {'genuine': images, 'forgery': []}
        print(f"[LFW] Loaded {len(self.data)} identities "
              f"(filtered >= {self.min_images} images), "
              f"{sum(len(v['genuine']) for v in self.data.values())} total images")

    def _preprocess(self, image):
        img = np.array(image, dtype=np.uint8)
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)

# ═══════════════════════════════════════════════════════════════════════════
# DATA: Fingerprint Loader (SOCOFing)
# ═══════════════════════════════════════════════════════════════════════════

class SOCOFingDataset(BiometricDataset):
    IMG_SIZE = (96, 96)

    def _load_data(self):
        real_dir = os.path.join(self.root_dir, 'Real')
        altered_dirs = [
            os.path.join(self.root_dir, 'Altered', 'Altered-Easy'),
            os.path.join(self.root_dir, 'Altered', 'Altered-Medium'),
            os.path.join(self.root_dir, 'Altered', 'Altered-Hard'),
        ]
        if os.path.exists(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if not fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')): continue
                try: subj_id = int(fname.split('__')[0])
                except (ValueError, IndexError): continue
                if subj_id not in self.data:
                    self.data[subj_id] = {'genuine': [], 'forgery': []}
                self.data[subj_id]['genuine'].append(os.path.join(real_dir, fname))
        for alt_dir in altered_dirs:
            if not os.path.exists(alt_dir): continue
            for fname in sorted(os.listdir(alt_dir)):
                if not fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')): continue
                try: subj_id = int(fname.split('__')[0])
                except (ValueError, IndexError): continue
                if subj_id not in self.data:
                    self.data[subj_id] = {'genuine': [], 'forgery': []}
                self.data[subj_id]['forgery'].append(os.path.join(alt_dir, fname))
        print(f"[SOCOFing] Loaded {len(self.data)} subjects, "
              f"{sum(len(v['genuine']) for v in self.data.values())} genuine, "
              f"{sum(len(v['forgery']) for v in self.data.values())} forgery/altered")

    def _preprocess(self, image):
        img = np.array(image, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)

# ═══════════════════════════════════════════════════════════════════════════
# DATA: Augmentations
# ═══════════════════════════════════════════════════════════════════════════

def get_augmentation(modality, training=True):
    if not training:
        return A.Compose([])
    if modality == 'signature':
        return A.Compose([
            A.Affine(shift_limit=0.05, scale=(0.9, 1.1), rotate=(-5, 5), border_mode=0, p=0.5),
            A.ElasticTransform(alpha=20, sigma=5, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ])
    elif modality == 'face':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(shift_limit=0.05, scale=(0.95, 1.05), rotate=(-10, 10), border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])
    elif modality == 'fingerprint':
        return A.Compose([
            A.Affine(shift_limit=0.03, scale=(0.95, 1.05), rotate=(-8, 8), border_mode=0, p=0.4),
            A.ElasticTransform(alpha=15, sigma=4, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ])
    return A.Compose([])

# ═══════════════════════════════════════════════════════════════════════════
# DATA: Samplers
# ═══════════════════════════════════════════════════════════════════════════

class PairSampler:
    def __init__(self, dataset_data, batch_size=32, neg_ratio=0.5):
        self.data = dataset_data
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.subjects = list(dataset_data.keys())

    def sample_batch(self):
        pairs = []
        n_neg = int(self.batch_size * self.neg_ratio)
        n_pos = self.batch_size - n_neg
        for _ in range(n_pos):
            subj = random.choice(self.subjects)
            genuine = self.data[subj]['genuine']
            if len(genuine) >= 2:
                a, b = random.sample(genuine, 2)
                pairs.append((a, b, 1))
            elif len(genuine) == 1:
                pairs.append((genuine[0], genuine[0], 1))
        for _ in range(n_neg):
            strategy = random.random()
            if strategy < 0.5:
                subj = random.choice(self.subjects)
                genuine = self.data[subj]['genuine']
                forgery = self.data[subj].get('forgery', [])
                if genuine and forgery:
                    pairs.append((random.choice(genuine), random.choice(forgery), 0))
                    continue
            s1, s2 = random.sample(self.subjects, 2)
            g1, g2 = self.data[s1]['genuine'], self.data[s2]['genuine']
            if g1 and g2:
                pairs.append((random.choice(g1), random.choice(g2), 0))
        random.shuffle(pairs)
        return pairs


class EpisodeSampler:
    def __init__(self, dataset_data, n_way=5, k_shot=5, q_query=5):
        self.data = dataset_data
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        min_samples = k_shot + q_query
        self.valid_subjects = [
            s for s in dataset_data.keys()
            if len(dataset_data[s]['genuine']) >= min_samples
        ]
        if len(self.valid_subjects) < n_way:
            raise ValueError(f"Not enough valid subjects ({len(self.valid_subjects)}) "
                             f"for {n_way}-way episodes.")
        print(f"[EpisodeSampler] {len(self.valid_subjects)} valid subjects "
              f"for {n_way}-way {k_shot}-shot episodes")

    def sample_episode(self):
        episode_classes = random.sample(self.valid_subjects, self.n_way)
        support_paths, query_paths = [], []
        for class_idx, subj in enumerate(episode_classes):
            genuine = list(self.data[subj]['genuine'])
            selected = random.sample(genuine, self.k_shot + self.q_query)
            support = selected[:self.k_shot]
            query = selected[self.k_shot:]
            for path in support: support_paths.append((path, class_idx))
            for path in query: query_paths.append((path, class_idx))
        return support_paths, query_paths

# ═══════════════════════════════════════════════════════════════════════════
# MODELS: Backbone
# ═══════════════════════════════════════════════════════════════════════════

class ResNetEncoder(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, in_channels=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    resnet.conv1.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class LightCNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(nn.Linear(256, embedding_dim))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# ═══════════════════════════════════════════════════════════════════════════
# MODELS: Siamese Network
# ═══════════════════════════════════════════════════════════════════════════

class SiameseNetwork(nn.Module):
    def __init__(self, backbone='resnet', embedding_dim=128, pretrained=True, in_channels=1):
        super().__init__()
        if backbone == 'resnet':
            self.encoder = ResNetEncoder(embedding_dim, pretrained, in_channels)
        else:
            self.encoder = LightCNNEncoder(embedding_dim, in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, img1, img2):
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        distance = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1) + 1e-8)
        diff = torch.abs(emb1 - emb2)
        similarity = torch.sigmoid(self.classifier(diff)).squeeze(1)
        return {'emb1': emb1, 'emb2': emb2, 'distance': distance, 'similarity': similarity}

    def get_embedding(self, x):
        return self.encoder(x)

# ═══════════════════════════════════════════════════════════════════════════
# MODELS: Prototypical Network
# ═══════════════════════════════════════════════════════════════════════════

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone='resnet', embedding_dim=128, pretrained=True,
                 in_channels=1, distance='euclidean'):
        super().__init__()
        self.distance_type = distance
        if backbone == 'resnet':
            self.encoder = ResNetEncoder(embedding_dim, pretrained, in_channels)
        else:
            self.encoder = LightCNNEncoder(embedding_dim, in_channels)

    def compute_prototypes(self, support_embeddings, support_labels):
        classes = torch.unique(support_labels)
        prototypes = torch.zeros(len(classes), support_embeddings.size(1),
                                 device=support_embeddings.device)
        for i, c in enumerate(classes):
            mask = support_labels == c
            prototypes[i] = support_embeddings[mask].mean(dim=0)
        return prototypes, classes

    def compute_distances(self, query_embeddings, prototypes):
        if self.distance_type == 'euclidean':
            n_q, n_p = query_embeddings.size(0), prototypes.size(0)
            distances = (
                query_embeddings.unsqueeze(1).expand(n_q, n_p, -1) -
                prototypes.unsqueeze(0).expand(n_q, n_p, -1)
            ).pow(2).sum(dim=2)
            return -distances
        else:
            return torch.mm(query_embeddings, prototypes.t())

    def forward(self, support_images, support_labels, query_images):
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)
        prototypes, classes = self.compute_prototypes(support_embeddings, support_labels)
        logits = self.compute_distances(query_embeddings, prototypes)
        return {'logits': logits, 'prototypes': prototypes,
                'query_embeddings': query_embeddings, 'support_embeddings': support_embeddings,
                'classes': classes}

    def get_embedding(self, x):
        return self.encoder(x)

# ═══════════════════════════════════════════════════════════════════════════
# LOSSES
# ═══════════════════════════════════════════════════════════════════════════

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        label = label.float()
        pos_loss = label * distance.pow(2)
        neg_loss = (1 - label) * F.relu(self.margin - distance).pow(2)
        return 0.5 * (pos_loss + neg_loss).mean()


class PrototypicalLoss(nn.Module):
    def forward(self, logits, query_labels):
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, query_labels)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == query_labels).float().mean().item()
        return loss, accuracy


class BinaryCrossEntropyLoss(nn.Module):
    def forward(self, similarity, label):
        label = label.float()
        return F.binary_cross_entropy(similarity.squeeze(), label)

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config['training'].get('lr', 1e-4),
                                    weight_decay=config['training'].get('weight_decay', 1e-5))
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training'].get('lr_step', 20),
            gamma=config['training'].get('lr_gamma', 0.5))
        self.criterion = self._build_criterion()
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.results_dir = config.get('results_dir', 'results')
        os.makedirs(os.path.join(self.results_dir, 'checkpoints'), exist_ok=True)

    def _build_model(self):
        cfg = self.config['model']
        model_type = cfg['type']
        backbone = cfg.get('backbone', 'resnet')
        emb_dim = cfg.get('embedding_dim', 128)
        pretrained = cfg.get('pretrained', True)
        in_channels = cfg.get('in_channels', 1)
        if model_type == 'siamese':
            return SiameseNetwork(backbone, emb_dim, pretrained, in_channels)
        else:
            return PrototypicalNetwork(backbone, emb_dim, pretrained, in_channels,
                                       cfg.get('distance', 'euclidean'))

    def _build_criterion(self):
        loss_type = self.config['training'].get('loss', 'bce')
        if self.config['model']['type'] == 'siamese':
            if loss_type == 'contrastive':
                return ContrastiveLoss(margin=self.config['training'].get('margin', 2.0))
            else:
                return BinaryCrossEntropyLoss()
        return PrototypicalLoss()

    def _run_siamese_batch(self, batch, dataset, training=True):
        images1, images2, labels = [], [], []
        for path1, path2, label in batch:
            images1.append(dataset.load_image(path1))
            images2.append(dataset.load_image(path2))
            labels.append(label)
        images1 = torch.FloatTensor(np.stack(images1)).to(self.device)
        images2 = torch.FloatTensor(np.stack(images2)).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        if training:
            self.optimizer.zero_grad()
        output = self.model(images1, images2)
        if isinstance(self.criterion, ContrastiveLoss):
            loss = self.criterion(output['distance'], labels)
        else:
            loss = self.criterion(output['similarity'], labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        with torch.no_grad():
            if isinstance(self.criterion, ContrastiveLoss):
                preds = (output['distance'] < self.criterion.margin / 2).float()
            else:
                preds = (output['similarity'] > 0.5).float()
            correct = (preds == labels).sum().item()
        return loss.item(), correct, len(labels)

    def train_siamese_epoch(self, sampler, dataset, num_iterations=100):
        self.model.train()
        total_loss, total_correct, total_pairs = 0, 0, 0
        for i in range(num_iterations):
            batch = sampler.sample_batch()
            loss, correct, count = self._run_siamese_batch(batch, dataset, training=True)
            total_loss += loss
            total_correct += correct
            total_pairs += count
        return total_loss / num_iterations, total_correct / total_pairs if total_pairs > 0 else 0

    def validate_siamese_epoch(self, sampler, dataset, num_iterations=20):
        self.model.eval()
        total_loss, total_correct, total_pairs = 0, 0, 0
        with torch.no_grad():
            for i in range(num_iterations):
                batch = sampler.sample_batch()
                loss, correct, count = self._run_siamese_batch(batch, dataset, training=False)
                total_loss += loss
                total_correct += correct
                total_pairs += count
        return total_loss / num_iterations, total_correct / total_pairs if total_pairs > 0 else 0

    def _run_prototypical_batch(self, episode, dataset, training=True):
        support_paths, query_paths = episode
        support_images, support_labels = [], []
        for path, ci in support_paths:
            support_images.append(dataset.load_image(path))
            support_labels.append(ci)
        query_images, query_labels = [], []
        for path, ci in query_paths:
            query_images.append(dataset.load_image(path))
            query_labels.append(ci)
        support_images = torch.FloatTensor(np.stack(support_images)).to(self.device)
        support_labels = torch.LongTensor(support_labels).to(self.device)
        query_images = torch.FloatTensor(np.stack(query_images)).to(self.device)
        query_labels = torch.LongTensor(query_labels).to(self.device)
        if training:
            self.optimizer.zero_grad()
        output = self.model(support_images, support_labels, query_images)
        loss, acc = self.criterion(output['logits'], query_labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return loss.item(), acc

    def train_prototypical_epoch(self, sampler, dataset, num_episodes=100):
        self.model.train()
        total_loss, total_accuracy = 0, 0
        for i in range(num_episodes):
            episode = sampler.sample_episode()
            loss, acc = self._run_prototypical_batch(episode, dataset, training=True)
            total_loss += loss
            total_accuracy += acc
        return total_loss / num_episodes, total_accuracy / num_episodes

    def validate_prototypical_epoch(self, sampler, dataset, num_episodes=20):
        self.model.eval()
        total_loss, total_accuracy = 0, 0
        with torch.no_grad():
            for i in range(num_episodes):
                episode = sampler.sample_episode()
                loss, acc = self._run_prototypical_batch(episode, dataset, training=False)
                total_loss += loss
                total_accuracy += acc
        return total_loss / num_episodes, total_accuracy / num_episodes

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }
        path = os.path.join(self.results_dir, 'checkpoints', f'checkpoint_epoch_{self.epoch}.pth')
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.results_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)

    def train(self, dataset):
        model_type = self.config['model']['type']
        epochs = self.config['training'].get('epochs', 100)
        patience = self.config['training'].get('patience', 15)
        iterations = self.config['training'].get('iterations_per_epoch', 100)
        val_iterations = max(iterations // 5, 10)

        train_data, val_data, _ = dataset.split_subjects()

        if model_type == 'siamese':
            batch_size = self.config['training'].get('batch_size', 32)
            train_sampler = PairSampler(train_data, batch_size=batch_size)
            val_sampler = PairSampler(val_data, batch_size=batch_size)
        else:
            n_way = self.config['training'].get('n_way', 5)
            k_shot = self.config['training'].get('k_shot', 5)
            q_query = self.config['training'].get('q_query', 5)
            train_sampler = EpisodeSampler(train_data, n_way=n_way, k_shot=k_shot, q_query=q_query)
            val_sampler = EpisodeSampler(val_data, n_way=n_way, k_shot=k_shot, q_query=q_query)

        self.best_val_loss = float('inf')

        print(f"\n{'='*60}")
        print(f"  Training {model_type.upper()} Network")
        print(f"  Epochs: {epochs} | Patience: {patience} | Device: {self.device}")
        print(f"  Train subjects: {len(train_data)} | Val subjects: {len(val_data)}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            start_time = time.time()

            # Training pass
            if model_type == 'siamese':
                train_loss, train_acc = self.train_siamese_epoch(
                    train_sampler, dataset, num_iterations=iterations)
            else:
                train_loss, train_acc = self.train_prototypical_epoch(
                    train_sampler, dataset, num_episodes=iterations)

            # Validation pass
            if model_type == 'siamese':
                val_loss, val_acc = self.validate_siamese_epoch(
                    val_sampler, dataset, num_iterations=val_iterations)
            else:
                val_loss, val_acc = self.validate_prototypical_epoch(
                    val_sampler, dataset, num_episodes=val_iterations)

            elapsed = time.time() - start_time
            self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if is_best:
                self.save_checkpoint(is_best=True)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                  f"LR: {lr:.6f} | {'[BEST]' if is_best else ''} | {elapsed:.1f}s")

            if self.patience_counter >= patience:
                print(f"\n[Early Stopping] No val improvement for {patience} epochs.")
                break

        print(f"\n{'='*60}")
        print(f"  Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGS
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = [
    {
        'name': 'siamese_signature_cedar',
        'model': {'type': 'siamese', 'backbone': 'resnet', 'embedding_dim': 128,
                  'pretrained': True, 'in_channels': 1},
        'dataset': {'modality': 'signature', 'name': 'cedar',
                    'root_dir': 'data/raw/signatures/CEDAR'},
        'training': {'epochs': 10, 'batch_size': 32, 'lr': 0.0001, 'weight_decay': 1e-5,
                     'loss': 'bce', 'margin': 2.0, 'scheduler': 'step',
                     'lr_step': 20, 'lr_gamma': 0.5, 'patience': 15,
                     'iterations_per_epoch': 100},
        'results_dir': 'results/siamese_signature_cedar',
    },
    {
        'name': 'proto_signature_cedar',
        'model': {'type': 'prototypical', 'backbone': 'resnet', 'embedding_dim': 128,
                  'pretrained': True, 'in_channels': 1, 'distance': 'cosine'},
        'dataset': {'modality': 'signature', 'name': 'cedar',
                    'root_dir': 'data/raw/signatures/CEDAR'},
        'training': {'epochs': 10, 'n_way': 5, 'k_shot': 5, 'q_query': 5,
                     'lr': 0.0001, 'weight_decay': 1e-5, 'scheduler': 'step',
                     'lr_step': 20, 'lr_gamma': 0.5, 'patience': 15,
                     'iterations_per_epoch': 100},
        'results_dir': 'results/proto_signature_cedar',
    },
    {
        'name': 'siamese_face_att',
        'model': {'type': 'siamese', 'backbone': 'resnet', 'embedding_dim': 128,
                  'pretrained': True, 'in_channels': 1},
        'dataset': {'modality': 'face', 'name': 'att',
                    'root_dir': 'data/raw/faces/att_faces'},
        'training': {'epochs': 10, 'batch_size': 32, 'lr': 0.0001, 'weight_decay': 1e-5,
                     'loss': 'bce', 'margin': 2.0, 'scheduler': 'step',
                     'lr_step': 20, 'lr_gamma': 0.5, 'patience': 15,
                     'iterations_per_epoch': 100},
        'results_dir': 'results/siamese_face_att',
    },
    {
        'name': 'proto_face_att',
        'model': {'type': 'prototypical', 'backbone': 'resnet', 'embedding_dim': 128,
                  'pretrained': True, 'in_channels': 1, 'distance': 'cosine'},
        'dataset': {'modality': 'face', 'name': 'att',
                    'root_dir': 'data/raw/faces/att_faces'},
        'training': {'epochs': 10, 'n_way': 5, 'k_shot': 5, 'q_query': 5,
                     'lr': 0.0001, 'weight_decay': 1e-5, 'scheduler': 'step',
                     'lr_step': 20, 'lr_gamma': 0.5, 'patience': 15,
                     'iterations_per_epoch': 100},
        'results_dir': 'results/proto_face_att',
    },
    {
        'name': 'siamese_fingerprint_socofing',
        'model': {'type': 'siamese', 'backbone': 'resnet', 'embedding_dim': 128,
                  'pretrained': True, 'in_channels': 1},
        'dataset': {'modality': 'fingerprint', 'name': 'socofing',
                    'root_dir': 'data/raw/fingerprints/SOCOFing'},
        'training': {'epochs': 10, 'batch_size': 32, 'lr': 0.0001, 'weight_decay': 1e-5,
                     'loss': 'bce', 'margin': 2.0, 'scheduler': 'step',
                     'lr_step': 20, 'lr_gamma': 0.5, 'patience': 15,
                     'iterations_per_epoch': 100},
        'results_dir': 'results/siamese_fingerprint_socofing',
    },
    {
        'name': 'proto_fingerprint_socofing',
        'model': {'type': 'prototypical', 'backbone': 'resnet', 'embedding_dim': 128,
                  'pretrained': True, 'in_channels': 1, 'distance': 'cosine'},
        'dataset': {'modality': 'fingerprint', 'name': 'socofing',
                    'root_dir': 'data/raw/fingerprints/SOCOFing'},
        'training': {'epochs': 10, 'n_way': 5, 'k_shot': 5, 'q_query': 5,
                     'lr': 0.0001, 'weight_decay': 1e-5, 'scheduler': 'step',
                     'lr_step': 20, 'lr_gamma': 0.5, 'patience': 15,
                     'iterations_per_epoch': 100},
        'results_dir': 'results/proto_fingerprint_socofing',
    },
]


def get_dataset_for_config(config):
    modality = config['dataset']['modality']
    name = config['dataset']['name']
    root_dir = config['dataset']['root_dir']
    transform = get_augmentation(modality, training=True)
    
    # Use auto-discovered paths if available
    if name == 'cedar' and 'cedar' in discovered_paths:
        root_dir = discovered_paths['cedar']
    elif name == 'att' and 'att' in discovered_paths:
        root_dir = discovered_paths['att']
    elif name == 'socofing' and 'socofing' in discovered_paths:
        root_dir = discovered_paths['socofing']
    elif name == 'lfw' and 'lfw' in discovered_paths:
        root_dir = discovered_paths['lfw']
    
    print(f"  Using root_dir: {root_dir}")
    
    if modality == 'signature' and name == 'cedar':
        return CEDARDataset(root_dir, transform=transform)
    elif modality == 'face' and name == 'att':
        return ATTFaceDataset(root_dir, transform=transform)
    elif modality == 'face' and name == 'lfw':
        return LFWDataset(root_dir, min_images=5, transform=transform)
    elif modality == 'fingerprint' and name == 'socofing':
        return SOCOFingDataset(root_dir, transform=transform)
    raise ValueError(f"Unknown dataset: {modality}/{name}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Run All Training
# ═══════════════════════════════════════════════════════════════════════════

def run_all_training():
    print("=" * 70)
    print("  BIOMETRIC FEW-SHOT VERIFICATION — FULL TRAINING SUITE")
    print(f"  Device: {device}")
    print(f"  Experiments: {len(CONFIGS)}")
    print("=" * 70)

    results_summary = []

    for i, config in enumerate(CONFIGS):
        print(f"\n\n{'#' * 70}")
        print(f"  EXPERIMENT {i+1}/{len(CONFIGS)}: {config['name']}")
        print(f"{'#' * 70}\n")

        try:
            dataset = get_dataset_for_config(config)
            trainer = Trainer(config, device)
            trainer.train(dataset)
            results_summary.append({
                'name': config['name'],
                'status': 'SUCCESS',
                'best_loss': trainer.best_loss,
                'epochs': trainer.epoch,
            })
        except Exception as e:
            print(f"ERROR in {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'name': config['name'],
                'status': f'FAILED: {e}',
            })

    # Print Summary
    print(f"\n\n{'=' * 70}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 70}")
    for r in results_summary:
        status = r['status']
        name = r['name']
        if status == 'SUCCESS':
            print(f"  [OK] {name:40s} | best_loss={r['best_loss']:.4f} | epochs={r['epochs']}")
        else:
            print(f"  [FAIL] {name:40s} | {status}")

    # Zip only best checkpoints (skip per-epoch checkpoints to avoid multi-GB zip)
    print("\nZipping best checkpoints for download...")
    import zipfile
    with zipfile.ZipFile('results_trained.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for config in CONFIGS:
            results_dir = config.get('results_dir', 'results')
            best_path = os.path.join(results_dir, 'checkpoints', 'best.pth')
            if os.path.exists(best_path):
                zf.write(best_path)
    print("Results saved to results_trained.zip")
    print("Download it from the Colab file browser (left sidebar) or run:")
    print("  from google.colab import files; files.download('results_trained.zip')")


if __name__ == '__main__':
    run_all_training()
