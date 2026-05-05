"""
Prototypical Networks for Few-Shot Face Verification
Author: LAP Project
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import FaceResNet50, FaceEfficientNet, LightCNNEncoder, build_backbone


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot face verification.
    
    Architecture:
        Support Set (3-5 images) → Encoder → Embeddings → Mean → Prototype
        Query Image → Encoder → Embedding → Distance to Prototype → Confidence
    
    Advantages:
        - Learns from 3-5 images only (few-shot learning)
        - Computationally efficient (simple mean pooling)
        - Generalizes well with limited data
        - No fine-tuning required after enrollment
    
    Paper: Prototypical Networks for Few-shot Learning (Snell et al., 2017)
    """

    def __init__(self, backbone='resnet50', embedding_dim=512, 
                 pretrained=True, in_channels=3):
        """
        Args:
            backbone: 'resnet50', 'efficientnet', or 'light'
            embedding_dim: Output embedding dimension (512 recommended)
            pretrained: Use ImageNet pretrained weights
            in_channels: 3 (RGB) or 1 (grayscale)
        """
        super().__init__()

        # Encoder selection
        if backbone == 'resnet50':
            self.encoder = FaceResNet50(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                in_channels=in_channels,
            )
        elif backbone in ('efficientnet', 'efficientnet_b3'):
            self.encoder = FaceEfficientNet(
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                in_channels=in_channels,
            )
        elif backbone == 'light':
            self.encoder = LightCNNEncoder(
                embedding_dim=min(embedding_dim, 256),
                in_channels=in_channels,
            )
        else:
            self.encoder = build_backbone({
                'backbone': backbone,
                'embedding_dim': embedding_dim,
                'in_channels': in_channels,
            })

        self.embedding_dim = self.encoder.embedding_dim

    def encode(self, images):
        """
        Encode images to embeddings.
        
        Args:
            images: (B, C, H, W) tensor
        
        Returns:
            embeddings: (B, embedding_dim) L2-normalized
        """
        return self.encoder(images)

    def create_prototype(self, support_images):
        """
        Create prototype from support images (enrollment).
        
        Args:
            support_images: (N, C, H, W) tensor, N=3-5 enrollment images
        
        Returns:
            prototype: (embedding_dim,) tensor, mean of support embeddings
        """
        support_embs = self.encode(support_images)  # (N, embedding_dim)
        prototype = support_embs.mean(dim=0)  # (embedding_dim,)
        return prototype

    def compute_distance(self, query_emb, prototype):
        """
        Compute distance between query embedding and prototype.
        
        Args:
            query_emb: (embedding_dim,) or (B, embedding_dim)
            prototype: (embedding_dim,)
        
        Returns:
            distance: Euclidean distance
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)  # (1, embedding_dim)
        
        # Euclidean distance: ||query - prototype||_2
        distance = torch.norm(query_emb - prototype, p=2, dim=1)  # (B,)
        return distance

    def compute_confidence(self, distance):
        """
        Convert distance to confidence score [0, 1].
        
        Args:
            distance: Euclidean distance
        
        Returns:
            confidence: [0, 1] similarity score
        """
        # Normalize distance to [0, 1] range
        # distance ≈ 0 → confidence ≈ 1 (very similar)
        # distance ≈ 2 → confidence ≈ 0 (very different)
        confidence = 1.0 - (distance / 2.0)
        confidence = torch.clamp(confidence, 0.0, 1.0)
        return confidence

    def verify(self, query_image, prototype):
        """
        Verify query image against prototype.
        
        Args:
            query_image: (1, C, H, W) or (C, H, W) tensor
            prototype: (embedding_dim,) tensor
        
        Returns:
            dict:
                'embedding': query embedding
                'distance': distance to prototype
                'confidence': confidence score [0, 1]
                'verdict': 'MATCH' or 'NO MATCH' (threshold=0.6)
        """
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)  # (1, C, H, W)

        with torch.no_grad():
            query_emb = self.encode(query_image)  # (1, embedding_dim)
            query_emb = query_emb.squeeze(0)  # (embedding_dim,)
            
            distance = self.compute_distance(query_emb, prototype)  # (1,)
            confidence = self.compute_confidence(distance)  # (1,)
            
            # Decision threshold
            threshold = 0.6
            verdict = 'MATCH' if confidence.item() > threshold else 'NO MATCH'

        return {
            'embedding': query_emb.detach().cpu(),
            'distance': distance.item(),
            'confidence': confidence.item(),
            'verdict': verdict,
        }

    def batch_verify(self, query_images, prototype):
        """
        Verify multiple query images at once.
        
        Args:
            query_images: (B, C, H, W) tensor
            prototype: (embedding_dim,) tensor
        
        Returns:
            dict with batch results
        """
        with torch.no_grad():
            query_embs = self.encode(query_images)  # (B, embedding_dim)
            distances = self.compute_distance(query_embs, prototype)  # (B,)
            confidences = self.compute_confidence(distances)  # (B,)

        return {
            'embeddings': query_embs.detach().cpu(),
            'distances': distances.detach().cpu(),
            'confidences': confidences.detach().cpu(),
            'verdicts': ['MATCH' if conf > 0.6 else 'NO MATCH' 
                        for conf in confidences.cpu().numpy()],
        }

    def forward(self, query_images, support_images=None):
        """
        Forward pass for training (optional).
        
        Args:
            query_images: (B, C, H, W)
            support_images: (B, N, C, H, W) optional, N=support set size
        
        Returns:
            embeddings or (query_embs, prototypes)
        """
        query_embs = self.encode(query_images)  # (B, embedding_dim)
        
        if support_images is not None:
            B, N, C, H, W = support_images.shape
            support_images = support_images.view(B * N, C, H, W)
            support_embs = self.encode(support_images)  # (B*N, embedding_dim)
            support_embs = support_embs.view(B, N, self.embedding_dim)
            prototypes = support_embs.mean(dim=1)  # (B, embedding_dim)
            return query_embs, prototypes
        
        return query_embs


# Alias for compatibility
ProtoNet = PrototypicalNetwork
