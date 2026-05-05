"""
Prototypical Network for few-shot biometric verification.
"""

import torch
import torch.nn as nn
from models.backbone import ResNetEncoder, LightCNNEncoder


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot classification.
    
    Key idea:
        1. Compute a "prototype" for each class as the mean of its
           support set embeddings
        2. Classify query samples by finding nearest prototype
    
    For verification:
        - Compute prototype from K genuine samples (support set)
        - Compare query sample's distance to genuine prototype
        - If distance < threshold → genuine, else → forgery
    """

    def __init__(self, backbone='resnet', embedding_dim=512,
                 pretrained=True, in_channels=1, distance='euclidean',
                 architecture='resnet18', use_attention=True):
        """
        Args:
            backbone: 'resnet' or 'light'
            embedding_dim: Size of embedding vectors
            pretrained: Use pretrained weights
            in_channels: Input channels (1=grayscale)
            distance: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.distance_type = distance

        if backbone == 'resnet':
            self.encoder = ResNetEncoder(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                in_channels=in_channels,
                architecture=architecture,
                use_attention=use_attention,
            )
        elif backbone == 'light':
            self.encoder = LightCNNEncoder(
                embedding_dim=embedding_dim,
                in_channels=in_channels
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def compute_prototypes(self, support_embeddings, support_labels):
        """
        Compute class prototypes as the mean of support embeddings per class.
        
        Args:
            support_embeddings: (n_support, embedding_dim)
            support_labels: (n_support,) — integer class labels
            
        Returns:
            prototypes: (n_classes, embedding_dim)
            classes: sorted list of unique class labels
        """
        classes = torch.unique(support_labels)
        prototypes = torch.zeros(
            len(classes),
            support_embeddings.size(1),
            device=support_embeddings.device
        )

        for i, c in enumerate(classes):
            mask = support_labels == c
            prototypes[i] = support_embeddings[mask].mean(dim=0)

        return prototypes, classes

    def compute_distances(self, query_embeddings, prototypes):
        """
        Compute distances between query embeddings and prototypes.
        
        Args:
            query_embeddings: (n_query, embedding_dim)
            prototypes: (n_classes, embedding_dim)
            
        Returns:
            distances: (n_query, n_classes) — negative distances (for softmax)
        """
        if self.distance_type == 'euclidean':
            # Efficient pairwise Euclidean distance
            # ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q·p
            n_q = query_embeddings.size(0)
            n_p = prototypes.size(0)

            distances = (
                query_embeddings.unsqueeze(1).expand(n_q, n_p, -1) -
                prototypes.unsqueeze(0).expand(n_q, n_p, -1)
            ).pow(2).sum(dim=2)

            return -distances  # Negative because we want "closer = higher score"

        elif self.distance_type == 'cosine':
            # Cosine similarity (already L2-normalized from backbone)
            return torch.mm(query_embeddings, prototypes.t())

        else:
            raise ValueError(f"Unknown distance: {self.distance_type}")

    def forward(self, support_images, support_labels, query_images):
        """
        Full forward pass for an episode.
        
        Args:
            support_images: (n_support, C, H, W)
            support_labels: (n_support,)
            query_images: (n_query, C, H, W)
            
        Returns:
            dict with:
                'logits': (n_query, n_classes) — log-probabilities
                'prototypes': (n_classes, embedding_dim)
                'query_embeddings': (n_query, embedding_dim)
                'support_embeddings': (n_support, embedding_dim)
        """
        # Encode support and query
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        # Compute prototypes
        prototypes, classes = self.compute_prototypes(
            support_embeddings, support_labels
        )

        # Compute distances (negative, so higher = closer)
        logits = self.compute_distances(query_embeddings, prototypes)

        return {
            'logits': logits,
            'prototypes': prototypes,
            'query_embeddings': query_embeddings,
            'support_embeddings': support_embeddings,
            'classes': classes,
        }

    def get_embedding(self, x):
        """Get embedding for a single image."""
        return self.encoder(x)

    def verify(self, support_images, query_image):
        """
        Verification mode: compare a single query against K support samples.
        
        Args:
            support_images: (K, C, H, W) — genuine reference samples
            query_image: (1, C, H, W) — image to verify
            
        Returns:
            distance: scalar distance to the genuine prototype
            is_genuine: boolean based on learned threshold
        """
        with torch.no_grad():
            support_emb = self.encoder(support_images)
            query_emb = self.encoder(query_image)

            # Prototype = mean of support embeddings
            prototype = support_emb.mean(dim=0, keepdim=True)

            # Distance to prototype
            if self.distance_type == 'euclidean':
                distance = torch.sqrt(
                    ((query_emb - prototype) ** 2).sum(dim=1) + 1e-8
                )
            else:
                distance = 1 - torch.mm(query_emb, prototype.t()).squeeze()

            return distance.item()
