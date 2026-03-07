"""
Siamese Network for biometric verification.
Takes two images and outputs a similarity score.
"""

import torch
import torch.nn as nn
from models.backbone import ResNetEncoder, LightCNNEncoder


class SiameseNetwork(nn.Module):
    """
    Siamese Network with shared-weight twin branches.
    
    Architecture:
        Image1 → Encoder → Embedding1 ─┐
                                        ├── Distance → Similarity Score
        Image2 → Encoder → Embedding2 ─┘
    
    The same encoder processes both images (weight sharing).
    Distance between embeddings determines if same person or not.
    """

    def __init__(self, backbone='resnet', embedding_dim=128,
                 pretrained=True, in_channels=1):
        """
        Args:
            backbone: 'resnet' or 'light' — encoder architecture choice
            embedding_dim: Size of embedding vectors
            pretrained: Use pretrained weights for ResNet
            in_channels: Input image channels (1=grayscale)
        """
        super().__init__()

        if backbone == 'resnet':
            self.encoder = ResNetEncoder(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                in_channels=in_channels
            )
        elif backbone == 'light':
            self.encoder = LightCNNEncoder(
                embedding_dim=embedding_dim,
                in_channels=in_channels
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Similarity head: takes concatenated/difference features → score
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward_once(self, x):
        """Pass one image through the encoder."""
        return self.encoder(x)

    def forward(self, img1, img2):
        """
        Forward pass for a pair of images.
        
        Args:
            img1: Tensor of shape (batch, C, H, W) — first image
            img2: Tensor of shape (batch, C, H, W) — second image
            
        Returns:
            dict with:
                'emb1': embedding of img1 (batch, embedding_dim)
                'emb2': embedding of img2 (batch, embedding_dim)
                'distance': L2 distance between embeddings (batch,)
                'similarity': sigmoid similarity score (batch,)
        """
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)

        # L2 distance
        distance = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1) + 1e-8)

        # Similarity via classifier on absolute difference
        diff = torch.abs(emb1 - emb2)
        similarity = torch.sigmoid(self.classifier(diff)).squeeze(1)

        return {
            'emb1': emb1,
            'emb2': emb2,
            'distance': distance,
            'similarity': similarity,
        }

    def get_embedding(self, x):
        """Get embedding for a single image (for evaluation/inference)."""
        return self.forward_once(x)
