"""
Tests for model correctness — validates gradient flow, distance metrics,
and the fix alignment between loss functions and accuracy measurement.

These tests run WITHOUT trained checkpoints (only random weights).
"""

import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese import SiameseNetwork
from models.prototypical import PrototypicalNetwork
from losses.losses import ContrastiveLoss, BinaryCrossEntropyLoss, PrototypicalLoss


# ── Fix 1: BCE loss trains classifier head ──────────────────────────────

class TestBCETrainsClassifier:
    """Verify that BCE loss propagates gradients through the classifier head."""

    def test_bce_updates_classifier_weights(self):
        model = SiameseNetwork(
            backbone='light', embedding_dim=32, pretrained=False, in_channels=1
        )
        criterion = BinaryCrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Save initial classifier weights
        initial_weights = model.classifier[0].weight.clone().detach()

        # One training step
        img1 = torch.randn(4, 1, 96, 96)
        img2 = torch.randn(4, 1, 96, 96)
        labels = torch.FloatTensor([1, 0, 1, 0])

        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output['similarity'], labels)
        loss.backward()
        optimizer.step()

        # Classifier weights MUST have changed
        updated_weights = model.classifier[0].weight.detach()
        assert not torch.equal(initial_weights, updated_weights), \
            "BCE loss did not update classifier weights!"

    def test_contrastive_does_not_train_classifier(self):
        """Contrastive loss only backprops through encoder, not classifier."""
        model = SiameseNetwork(
            backbone='light', embedding_dim=32, pretrained=False, in_channels=1
        )
        criterion = ContrastiveLoss(margin=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_weights = model.classifier[0].weight.clone().detach()

        img1 = torch.randn(4, 1, 96, 96)
        img2 = torch.randn(4, 1, 96, 96)
        labels = torch.FloatTensor([1, 0, 1, 0])

        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output['distance'], labels)
        loss.backward()
        optimizer.step()

        updated_weights = model.classifier[0].weight.detach()
        assert torch.equal(initial_weights, updated_weights), \
            "Contrastive loss should NOT update classifier weights!"


# ── Fix 1: Accuracy aligned with loss signal ────────────────────────────

class TestAccuracyAlignment:
    """Verify accuracy measurement uses the same signal as the loss."""

    def test_bce_accuracy_uses_similarity(self):
        model = SiameseNetwork(
            backbone='light', embedding_dim=32, pretrained=False, in_channels=1
        )
        img1 = torch.randn(4, 1, 96, 96)
        img2 = torch.randn(4, 1, 96, 96)

        with torch.no_grad():
            output = model(img1, img2)
            # Similarity-based preds
            preds = (output['similarity'] > 0.5).float()
            assert preds.shape == (4, 1) or preds.shape == (4,)

    def test_contrastive_accuracy_uses_distance(self):
        model = SiameseNetwork(
            backbone='light', embedding_dim=32, pretrained=False, in_channels=1
        )
        criterion = ContrastiveLoss(margin=2.0)
        img1 = torch.randn(4, 1, 96, 96)
        img2 = torch.randn(4, 1, 96, 96)

        with torch.no_grad():
            output = model(img1, img2)
            # Distance-based preds: below half-margin → same
            preds = (output['distance'] < criterion.margin / 2).float()
            assert preds.shape == (4,)


# ── Fix 2: Cosine distance with L2-normalized embeddings ────────────────

class TestCosineDistance:
    """Validate cosine distance works correctly with unit vectors."""

    def test_self_similarity_is_one(self):
        model = PrototypicalNetwork(
            backbone='light', embedding_dim=32,
            pretrained=False, in_channels=1, distance='cosine'
        )
        model.eval()
        img = torch.randn(1, 1, 96, 96)

        with torch.no_grad():
            emb = model.encoder(img)
            # Self-similarity via dot product of unit vectors = 1.0
            sim = torch.mm(emb, emb.t()).item()
        assert abs(sim - 1.0) < 0.01, f"Self-similarity should be ~1.0, got {sim}"

    def test_embedding_is_normalized(self):
        model = PrototypicalNetwork(
            backbone='light', embedding_dim=32,
            pretrained=False, in_channels=1, distance='cosine'
        )
        model.eval()
        img = torch.randn(1, 1, 96, 96)

        with torch.no_grad():
            emb = model.encoder(img).numpy()
            norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.05, f"Embedding L2 norm should be ~1.0, got {norm}"


# ── Fix 7: Distance type consistency ────────────────────────────────────

class TestDistanceTypeConsistency:
    """Verify prototypical model exposes distance_type attribute."""

    def test_cosine_distance_type_set(self):
        model = PrototypicalNetwork(
            backbone='light', embedding_dim=32,
            pretrained=False, in_channels=1, distance='cosine'
        )
        assert model.distance_type == 'cosine'

    def test_euclidean_distance_type_set(self):
        model = PrototypicalNetwork(
            backbone='light', embedding_dim=32,
            pretrained=False, in_channels=1, distance='euclidean'
        )
        assert model.distance_type == 'euclidean'


# ── Siamese output shape validation ─────────────────────────────────────

class TestSiameseOutput:
    """Validate the SiameseNetwork forward pass returns expected keys."""

    def test_output_keys(self):
        model = SiameseNetwork(
            backbone='light', embedding_dim=32, pretrained=False, in_channels=1
        )
        img1 = torch.randn(2, 1, 96, 96)
        img2 = torch.randn(2, 1, 96, 96)

        with torch.no_grad():
            output = model(img1, img2)

        assert 'similarity' in output
        assert 'distance' in output
        assert 'embedding1' in output
        assert 'embedding2' in output

    def test_similarity_range(self):
        model = SiameseNetwork(
            backbone='light', embedding_dim=32, pretrained=False, in_channels=1
        )
        img1 = torch.randn(4, 1, 96, 96)
        img2 = torch.randn(4, 1, 96, 96)

        with torch.no_grad():
            output = model(img1, img2)
            sim = output['similarity']
        # Sigmoid output should be in [0, 1]
        assert (sim >= 0).all() and (sim <= 1).all(), \
            f"Similarity scores out of [0,1] range: {sim}"
