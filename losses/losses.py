"""
Loss functions for Siamese and Prototypical training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    
    L = y * d^2 + (1 - y) * max(0, margin - d)^2
    
    Where:
        y = 1 for genuine pairs (same person) → pull together
        y = 0 for impostor pairs (different person) → push apart
        d = Euclidean distance between embeddings
        margin = minimum distance for negative pairs
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        """
        Args:
            distance: Euclidean distance between pair embeddings (batch,)
            label: 1 = same person, 0 = different (batch,)
        Returns:
            Scalar loss value
        """
        label = label.float()
        pos_loss = label * distance.pow(2)
        neg_loss = (1 - label) * F.relu(self.margin - distance).pow(2)
        loss = 0.5 * (pos_loss + neg_loss)
        return loss.mean()


class CosineContrastiveLoss(nn.Module):
    """
    Cosine-based contrastive loss for L2-normalized embeddings.

    Directly optimizes cosine similarity — the same metric used at inference.
    Wraps torch.nn.CosineEmbeddingLoss internally.

    Args:
        margin: minimum cosine dissimilarity for negative pairs (default 0.5).
                Negative pairs are pushed to have cosine_sim < margin.
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, emb1, emb2, label):
        """
        Args:
            emb1: (batch, emb_dim) — first embedding (L2-normalized)
            emb2: (batch, emb_dim) — second embedding (L2-normalized)
            label: 1 = same person, 0 = different (batch,)
        Returns:
            Scalar loss value
        """
        # CosineEmbeddingLoss expects +1 = similar, -1 = dissimilar
        targets = label.float() * 2 - 1  # map {0,1} → {-1,+1}
        return self.loss_fn(emb1, emb2, targets)


class TripletLoss(nn.Module):
    """
    Triplet Loss: alternative to ContrastiveLoss.
    
    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: embeddings (batch, emb_dim)
        """
        d_pos = (anchor - positive).pow(2).sum(dim=1)
        d_neg = (anchor - negative).pow(2).sum(dim=1)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


class PrototypicalLoss(nn.Module):
    """
    Prototypical Network Loss.
    
    Negative log-probability of the correct class under the softmax
    over distances to all prototypes.
    
    L = -log( exp(-d(q, p_y)) / sum_k exp(-d(q, p_k)) )
    
    Where p_y is the correct class prototype.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits, query_labels):
        """
        Args:
            logits: (n_query, n_classes) — negative distances to prototypes
            query_labels: (n_query,) — ground truth class indices (0..N-1)
        Returns:
            loss: scalar
            accuracy: classification accuracy on this episode
        """
        log_probs = F.log_softmax(logits / self.temperature, dim=1)
        loss = F.nll_loss(log_probs, query_labels)

        # Compute episode accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == query_labels).float().mean().item()

        return loss, accuracy


class BinaryCrossEntropyLoss(nn.Module):
    """
    BCE loss for Siamese similarity scores.
    Alternative to ContrastiveLoss when using the classifier head.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, similarity, label):
        """
        Args:
            similarity: predicted similarity score [0, 1] (batch,)
            label: 1 = same, 0 = different (batch,)
        """
        return self.bce(similarity, label.float())


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for face verification.
    Adds angular margin to cosine similarity for better class separation.
    """
    def __init__(self, embedding_dim=512, num_classes=1000, margin=0.5, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(torch.pi - margin))
        self.mm = torch.sin(torch.tensor(torch.pi - margin)) * margin

    def forward(self, embeddings, labels):
        cosine = nn.functional.linear(
            nn.functional.normalize(embeddings),
            nn.functional.normalize(self.weight)
        )
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return nn.functional.cross_entropy(output, labels.long())
