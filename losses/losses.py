"""
Advanced Loss Functions for Face Verification
- Contrastive Loss (for Siamese)
- Triplet Loss (for metric learning)
- Prototypical Loss (for few-shot)
- Combined Loss (fusion)

Author: LAP Project
Version: 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    
    Formula:
        L = (1-Y) * 0.5 * D² + Y * 0.5 * max(margin - D, 0)²
    
    Where:
        Y = 0 (same person) → minimize distance
        Y = 1 (different person) → maximize distance beyond margin
        D = Euclidean distance between embeddings
        margin = 1.0 (typical)
    
    Reference: Hadsell et al., 2006
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        """
        Args:
            emb1: (B, embedding_dim) first embedding
            emb2: (B, embedding_dim) second embedding
            labels: (B,) binary labels, 0=same, 1=different
        
        Returns:
            loss: scalar loss value
        """
        # Euclidean distance
        distance = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1) + 1e-8)
        
        # Contrastive loss formula
        same_person = (labels == 0).float()
        diff_person = (labels == 1).float()
        
        loss_same = same_person * 0.5 * (distance ** 2)
        loss_diff = diff_person * 0.5 * torch.clamp(self.margin - distance, min=0) ** 2
        
        return (loss_same + loss_diff).mean()


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    Formula:
        L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    
    Where:
        d(.) = Euclidean distance
        margin = 0.5 (typical for normalized embeddings)
    
    Reference: Schroff et al., 2015 (FaceNet)
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (B, embedding_dim)
            positive: (B, embedding_dim) same person as anchor
            negative: (B, embedding_dim) different person
        
        Returns:
            loss: scalar
        """
        # Distances
        pos_distance = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1) + 1e-8)
        neg_distance = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1) + 1e-8)
        
        # Triplet loss
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0)
        
        return loss.mean()


class PrototypicalLoss(nn.Module):
    """
    Prototypical Loss for few-shot learning.
    
    Formula:
        L = -log(exp(-d(query, correct_prototype)) / Σ exp(-d(query, all_prototypes)))
    
    Which is equivalent to cross-entropy loss over distances.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_embs, support_embs, labels):
        """
        Args:
            query_embs: (B, embedding_dim) query embeddings
            support_embs: (B, N, embedding_dim) support set embeddings per query
            labels: (B, N) binary labels, 1=same, 0=different
        
        Returns:
            loss: scalar
        """
        B, N, D = support_embs.shape
        
        # Compute prototypes (mean of same-person supports)
        prototypes = []
        for b in range(B):
            same_mask = labels[b] == 1
            if same_mask.sum() > 0:
                proto = support_embs[b, same_mask].mean(dim=0)
            else:
                proto = support_embs[b, 0]  # fallback
            prototypes.append(proto)
        
        prototypes = torch.stack(prototypes)  # (B, D)
        
        # Distance to prototype
        distances = torch.sqrt(torch.sum((query_embs - prototypes) ** 2, dim=1) + 1e-8)
        
        # Cross-entropy style loss: penalize large distances
        loss = distances.mean()
        
        return loss


class CosineContrastiveLoss(nn.Module):
    """
    Contrastive Loss using Cosine Similarity (better for L2-normalized embeddings).
    
    For normalized embeddings, cosine similarity is equivalent to dot product.
    """

    def __init__(self, margin=0.3, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale  # temperature scaling

    def forward(self, emb1, emb2, labels):
        """
        Args:
            emb1: (B, embedding_dim) L2-normalized
            emb2: (B, embedding_dim) L2-normalized
            labels: (B,) binary, 0=same, 1=different
        
        Returns:
            loss: scalar
        """
        # Cosine similarity (already normalized)
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)  # (B,)
        
        # Scale for numerical stability
        logits = cosine_sim * self.scale  # (B,)
        
        # Same person: maximize similarity (target=1)
        # Different person: minimize similarity (target=0)
        targets = (labels == 0).float()
        
        # Cross-entropy on cosine similarity
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss - State-of-the-art angular margin loss for face recognition.
    
    Formula:
        L = -log(exp(s*cos(θ + m)) / (exp(s*cos(θ + m)) + Σ exp(s*cos(θ_j))))
    
    Where:
        s = scale (64.0 typical)
        m = angular margin (0.5 ≈ 28.6°)
        θ = angle between embedding and class center
    
    Reference: Deng et al., 2019 (ArcFace: Additive Angular Margin Loss)
    """

    def __init__(self, num_classes, embedding_dim=512, scale=64.0, margin=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        
        # Weight matrix (class centers)
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        """
        Args:
            emb: (B, embedding_dim) L2-normalized embeddings
            labels: (B,) class indices
        
        Returns:
            loss: scalar
        """
        # Normalize weight (class centers)
        W = F.normalize(self.weight, p=2, dim=1)  # (num_classes, embedding_dim)
        
        # Cosine similarity: emb @ W.T
        cos_theta = F.linear(emb, W)  # (B, num_classes)
        
        # Clamp to avoid numerical issues with acos
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # Compute angles
        theta = torch.acos(cos_theta)  # (B, num_classes)
        
        # Add margin to correct class
        theta_m = theta.clone()
        batch_range = torch.arange(emb.size(0), device=emb.device)
        theta_m[batch_range, labels] = theta[batch_range, labels] + self.margin
        
        # Compute logits: s * cos(θ)
        logits = self.scale * torch.cos(theta_m)  # (B, num_classes)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class CombinedFaceLoss(nn.Module):
    """
    Combined loss = α * ContrastiveLoss + β * PrototypicalLoss
    
    Best of both worlds:
    - Contrastive: direct pairwise learning
    - Prototypical: few-shot capability
    """

    def __init__(self, alpha=0.5, beta=0.5, contrastive_margin=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        self.proto_loss = PrototypicalLoss()

    def forward(self, emb1, emb2, labels, support_embs=None, support_labels=None):
        """
        Args:
            emb1, emb2: embeddings for contrastive loss
            labels: binary labels for contrastive
            support_embs: optional, for prototypical loss
            support_labels: optional, for prototypical loss
        
        Returns:
            combined_loss: weighted combination
        """
        # Contrastive component
        c_loss = self.contrastive_loss(emb1, emb2, labels)
        
        # Prototypical component (if provided)
        if support_embs is not None and support_labels is not None:
            p_loss = self.proto_loss(emb1, support_embs, support_labels)
            combined = self.alpha * c_loss + self.beta * p_loss
        else:
            combined = c_loss
        
        return combined


# Utility function to get loss by name
def get_loss(loss_name='contrastive', **kwargs):
    """
    Factory function to get loss by name.
    
    Args:
        loss_name: 'contrastive', 'triplet', 'proto', 'cosine_contrastive', 'arcface', 'combined'
        **kwargs: additional arguments for the loss
    
    Returns:
        loss_fn: instantiated loss module
    """
    losses = {
        'contrastive': ContrastiveLoss,
        'triplet': TripletLoss,
        'proto': PrototypicalLoss,
        'prototypical': PrototypicalLoss,
        'cosine_contrastive': CosineContrastiveLoss,
        'arcface': ArcFaceLoss,
        'combined': CombinedFaceLoss,
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name](**kwargs)
