"""
Improved Siamese Network for Face Verification — v2
Changes:
  - Supports resnet50 and efficientnet backbones
  - Stronger classifier head (cosine similarity instead of L2 diff)
  - Returns cosine similarity directly (more stable than L2 for 512-dim)
  - get_embedding() supports batch inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import ResNetEncoder, EfficientNetEncoder, LightCNNEncoder


class SiameseNetwork(nn.Module):
    """
    Improved Siamese Network with shared-weight twin branches.

    Architecture:
        Image1 → Encoder → Embedding1 (512-dim, L2-norm) ─┐
                                                            ├── Cosine Sim → Score
        Image2 → Encoder → Embedding2 (512-dim, L2-norm) ─┘

    Improvements over v1:
        1. ResNet-50 / EfficientNet backbone (vs ResNet-18)
        2. 512-dim embeddings (vs 128)
        3. RGB input (vs grayscale)
        4. Cosine similarity head (vs L2 diff)
        5. Stronger classifier head with residual connection
    """

    def __init__(self, backbone='resnet50', embedding_dim=512,
                 pretrained=True, in_channels=3):
        """
        Args:
            backbone: 'resnet50', 'efficientnet', or 'light'
            embedding_dim: 512 recommended for faces
            pretrained: Use ImageNet pretrained weights
            in_channels: 3 for RGB (recommended)
        """
        super().__init__()

        if backbone == 'resnet50':
            self.encoder = ResNetEncoder(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                in_channels=in_channels
            )
        elif backbone == 'efficientnet':
            self.encoder = EfficientNetEncoder(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                in_channels=in_channels
            )
        elif backbone in ('resnet', 'light'):
            self.encoder = LightCNNEncoder(
                embedding_dim=embedding_dim,
                in_channels=in_channels
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose: resnet50, efficientnet, light")

        # Improved classifier head
        # Takes |emb1 - emb2| and emb1 * emb2 (element-wise product) as features
        # This captures both magnitude and directional differences
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward_once(self, x):
        """Pass one image through the encoder."""
        return self.encoder(x)

    def forward(self, img1, img2):
        """
        Forward pass for a pair of images.

        Args:
            img1: (batch, C, H, W)
            img2: (batch, C, H, W)

        Returns:
            dict:
                'emb1': embedding of img1 (batch, embedding_dim)
                'emb2': embedding of img2 (batch, embedding_dim)
                'distance': L2 distance (batch,)
                'cosine_sim': cosine similarity [-1, 1] (batch,)
                'similarity': sigmoid score [0, 1] (batch,)
        """
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)

        # L2 distance (kept for backward compatibility)
        distance = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1) + 1e-8)

        # Cosine similarity (primary metric for normalized embeddings)
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)

        # Classifier on combined features (diff + product)
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
        """Get embedding for a single image or batch (inference)."""
        return self.forward_once(x)

    def get_similarity(self, img1, img2):
        """Convenience method — returns cosine similarity only."""
        with torch.no_grad():
            emb1 = self.forward_once(img1)
            emb2 = self.forward_once(img2)
            return F.cosine_similarity(emb1, emb2, dim=1)
