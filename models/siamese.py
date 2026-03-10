"""
Improved Siamese Network for biometric verification.

Optimized for signature / fingerprint / face verification tasks.

Key Improvements:
- L2 embedding normalization
- Cosine similarity metric
- Projection head for better embedding learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ResNetEncoder, LightCNNEncoder


class SiameseNetwork(nn.Module):
    """
    Improved Siamese Network with metric-learning friendly embeddings.

    Architecture:

        Image1 → Encoder → Projection → Normalize → Embedding1
                                                       │
                                                       │ cosine similarity
                                                       │
        Image2 → Encoder → Projection → Normalize → Embedding2

    """

    def __init__(
        self,
        backbone="resnet",
        embedding_dim=128,
        pretrained=True,
        in_channels=1,
    ):
        super().__init__()

        # -------------------------
        # Encoder backbone
        # -------------------------

        if backbone == "resnet":
            self.encoder = ResNetEncoder(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                in_channels=in_channels,
            )

        elif backbone == "light":
            self.encoder = LightCNNEncoder(
                embedding_dim=embedding_dim,
                in_channels=in_channels,
            )

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # -------------------------
        # Projection Head
        # improves embedding space
        # -------------------------

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Optional similarity classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    # -------------------------------------------------
    # Single image forward
    # -------------------------------------------------

    def forward_once(self, x):
        """
        Extract normalized embedding from a single image.
        """

        emb = self.encoder(x)

        emb = self.projection(emb)

        # L2 normalization (very important)
        emb = F.normalize(emb, p=2, dim=1)

        return emb

    # -------------------------------------------------
    # Pair forward
    # -------------------------------------------------

    def forward(self, img1, img2):
        """
        Forward pass for image pair.

        Returns:
            emb1
            emb2
            cosine similarity
            euclidean distance
        """

        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)

        # cosine similarity
        cosine_sim = F.cosine_similarity(emb1, emb2)

        # euclidean distance
        distance = torch.norm(emb1 - emb2, dim=1)

        # classifier similarity (optional)
        diff = torch.abs(emb1 - emb2)
        similarity = torch.sigmoid(self.classifier(diff)).squeeze(1)

        return {
            "emb1": emb1,
            "emb2": emb2,
            "cosine_similarity": cosine_sim,
            "distance": distance,
            "similarity": similarity,
        }

    # -------------------------------------------------
    # Inference embedding
    # -------------------------------------------------

    def get_embedding(self, x):
        """
        Extract embedding for a single image.
        Used for evaluation and inference.
        """

        return self.forward_once(x)
