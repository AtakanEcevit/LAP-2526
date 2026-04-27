"""
Siamese Network for Face Verification — v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import FaceResNet50, FaceEfficientNet, LightCNNEncoder, build_backbone


class SiameseNetwork(nn.Module):
    """
    Siamese Network with shared-weight twin branches.

    Architecture:
        Image1 → Encoder → Embedding1 (512-dim, L2-norm) ─┐
                                                            ├── Cosine Sim → Score
        Image2 → Encoder → Embedding2 (512-dim, L2-norm) ─┘

    Improvements over v1:
        1. ResNet-50 / EfficientNet backbone (vs ResNet-18)
        2. 512-dim embeddings (vs 128)
        3. GeM Pooling (vs Average Pooling)
        4. Cosine similarity + classifier head (vs L2 diff only)
    """

    def __init__(self, backbone='resnet50', embedding_dim=512,
                 pretrained=True, in_channels=1):
        """
        Args:
            backbone: 'resnet50', 'efficientnet', or 'light'
            embedding_dim: 512 recommended for faces
            pretrained: Use ImageNet pretrained weights
            in_channels: 1 for grayscale (mevcut dataset uyumlu)
        """
        super().__init__()

        if backbone == 'resnet50':
            self.encoder = FaceResNet50(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
            )
        elif backbone in ('efficientnet', 'efficientnet_b3'):
            self.encoder = FaceEfficientNet(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
            )
        elif backbone == 'light':
            self.encoder = LightCNNEncoder(
                embedding_dim=min(embedding_dim, 256),
            )
        else:
            # build_backbone ile config'den yükle
            self.encoder = build_backbone({'backbone': backbone, 'embedding_dim': embedding_dim})

        # Classifier head: diff + product → similarity score
        head_dim = self.encoder.embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(head_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward_once(self, x):
        """Tek bir görüntüyü encoder'dan geçir."""
        return self.encoder(x)

    def forward(self, img1, img2):
        """
        Args:
            img1: (batch, C, H, W)
            img2: (batch, C, H, W)

        Returns:
            dict:
                'emb1': embedding (batch, embedding_dim)
                'emb2': embedding (batch, embedding_dim)
                'distance': L2 distance (batch,)
                'cosine_sim': cosine similarity [-1, 1] (batch,)
                'similarity': sigmoid score [0, 1] (batch,)
        """
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)

        # L2 distance
        distance = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1) + 1e-8)

        # Cosine similarity (L2-normalize edilmiş embedding için ana metrik)
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)

        # Classifier: diff + product
        diff = torch.abs(emb1 - emb2)
        prod = emb1 * emb2
        combined = torch.cat([diff, prod], dim=1)
        similarity = torch.sigmoid(self.classifier(combined)).squeeze(1)

        return {
            'emb1': emb1,
            'emb2': emb2,
            'distance': distance,
            'cosine_sim': cosine_sim,
            'similarity': similarity,
        }

    def get_embedding(self, x):
        """Tek görüntü veya batch için embedding döner (inference)."""
        return self.forward_once(x)

    def get_similarity(self, img1, img2):
        """Sadece cosine similarity döner."""
        with torch.no_grad():
            emb1 = self.forward_once(img1)
            emb2 = self.forward_once(img2)
            return F.cosine_similarity(emb1, emb2, dim=1)
