"""
Loss functions — upgraded for face verification.
NEW: ArcFace Loss (state-of-the-art for face recognition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────
# NEW: ArcFace Loss (best for face recognition)
# ──────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    
    State-of-the-art loss for face recognition (2019, InsightFace).
    Adds an angular margin 'm' to the target class angle, forcing
    embeddings to be more discriminative on the hypersphere.
    
    L = -log( exp(s·cos(θ_y + m)) / (exp(s·cos(θ_y + m)) + Σ exp(s·cos(θ_j))) )
    
    Args:
        embedding_dim: Size of input embeddings (512 recommended)
        num_classes: Number of identities in training set
        s: Feature scale (default 64.0)
        m: Angular margin in radians (default 0.5 ≈ 28.6°)
    """

    def __init__(self, embedding_dim=512, num_classes=100, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: L2-normalized embeddings (batch, embedding_dim)
            labels: class indices (batch,)
        Returns:
            Scalar loss
        """
        # Normalize weights
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))

        # cos(θ + m) = cos θ · cos m - sin θ · sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability: if cos(θ) < threshold, use fallback
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return F.cross_entropy(output, labels.long())


# ──────────────────────────────────────────────
# Existing losses (kept + minor improvements)
# ──────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    L = y·d² + (1-y)·max(0, margin-d)²
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        label = label.float()
        pos_loss = label * distance.pow(2)
        neg_loss = (1 - label) * F.relu(self.margin - distance).pow(2)
        return (0.5 * (pos_loss + neg_loss)).mean()


class CosineContrastiveLoss(nn.Module):
    """Cosine-based contrastive loss for L2-normalized embeddings."""

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, emb1, emb2, label):
        targets = label.float() * 2 - 1  # {0,1} → {-1,+1}
        return self.loss_fn(emb1, emb2, targets)


class TripletLoss(nn.Module):
    """
    Triplet Loss.
    L = max(0, d(a,p) - d(a,n) + margin)
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = (anchor - positive).pow(2).sum(dim=1)
        d_neg = (anchor - negative).pow(2).sum(dim=1)
        return F.relu(d_pos - d_neg + self.margin).mean()


class PrototypicalLoss(nn.Module):
    """Prototypical Network Loss."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, query_labels):
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, query_labels)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == query_labels).float().mean().item()
        return loss, accuracy


class BinaryCrossEntropyLoss(nn.Module):
    """BCE loss for Siamese similarity scores."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, similarity, label):
        return self.bce(similarity, label.float())


# ──────────────────────────────────────────────
# NEW: Combined Loss (ArcFace + Contrastive)
# ──────────────────────────────────────────────

class CombinedFaceLoss(nn.Module):
    """
    Combined ArcFace + Cosine Contrastive Loss.
    
    ArcFace handles identity classification (discriminative embeddings).
    Cosine Contrastive handles pair-level similarity (verification).
    
    L = α·ArcFace + β·CosineContrastive
    
    Best of both worlds for face verification tasks.
    """

    def __init__(self, embedding_dim=512, num_classes=100,
                 alpha=0.7, beta=0.3):
        super().__init__()
        self.arcface = ArcFaceLoss(embedding_dim, num_classes)
        self.cosine = CosineContrastiveLoss(margin=0.5)
        self.alpha = alpha
        self.beta = beta

    def forward(self, emb1, emb2, label, identity_labels=None):
        """
        Args:
            emb1, emb2: embeddings (batch, dim)
            label: pair label 1=same, 0=different
            identity_labels: class indices for ArcFace (optional)
        """
        cosine_loss = self.cosine(emb1, emb2, label)

        if identity_labels is not None:
            # Combine both embeddings for ArcFace
            all_emb = torch.cat([emb1, emb2], dim=0)
            all_labels = torch.cat([identity_labels, identity_labels], dim=0)
            arc_loss = self.arcface(all_emb, all_labels)
            return self.alpha * arc_loss + self.beta * cosine_loss
        
        return cosine_loss
