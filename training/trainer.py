"""
Unified training loop for both Siamese and Prototypical networks.
Driven by YAML configs.
"""

import os
import sys
import time
import yaml
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.siamese import SiameseNetwork
from models.prototypical import PrototypicalNetwork
from losses.losses import ContrastiveLoss, PrototypicalLoss, BinaryCrossEntropyLoss
from data.samplers import PairSampler, EpisodeSampler
from data.pair_dataset import SiamesePairDataset
from data.episode_dataset import PrototypicalEpisodeDataset
from data.augmentations import get_augmentation
from utils import get_device


class _nullcontext:
    """Minimal no-op context manager (like contextlib.nullcontext)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class Trainer:
    """
    Unified trainer for Siamese and Prototypical networks.
    
    Handles:
        - Model creation from config
        - Data loading and sampling
        - Training loop with logging
        - Checkpoint saving/loading
        - Early stopping
    """

    def __init__(self, config_path):
        """
        Args:
            config_path: Path to YAML config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = get_device()
        self.model = self._build_model()
        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = self._build_criterion()

        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Results directory
        self.results_dir = self.config.get('results_dir', 'results')
        os.makedirs(os.path.join(self.results_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'logs'), exist_ok=True)

    def _build_model(self):
        """Create model based on config."""
        model_type = self.config['model']['type']
        backbone = self.config['model'].get('backbone', 'resnet')
        emb_dim = self.config['model'].get('embedding_dim', 128)
        pretrained = self.config['model'].get('pretrained', True)
        in_channels = self.config['model'].get('in_channels', 1)

        if model_type == 'siamese':
            return SiameseNetwork(
                backbone=backbone,
                embedding_dim=emb_dim,
                pretrained=pretrained,
                in_channels=in_channels,
            )
        elif model_type == 'prototypical':
            distance = self.config['model'].get('distance', 'euclidean')
            return PrototypicalNetwork(
                backbone=backbone,
                embedding_dim=emb_dim,
                pretrained=pretrained,
                in_channels=in_channels,
                distance=distance,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _build_optimizer(self):
        """Create optimizer from config."""
        lr = self.config['training'].get('lr', 1e-4)
        weight_decay = self.config['training'].get('weight_decay', 1e-5)
        return optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _build_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config['training'].get('scheduler', 'step')
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training'].get('lr_step', 20),
                gamma=self.config['training'].get('lr_gamma', 0.5),
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training'].get('epochs', 100),
            )
        return None

    def _build_criterion(self):
        """Create loss function from config."""
        model_type = self.config['model']['type']
        if model_type == 'siamese':
            loss_type = self.config['training'].get('loss', 'contrastive')
            if loss_type == 'contrastive':
                margin = self.config['training'].get('margin', 1.0)
                return ContrastiveLoss(margin=margin)
            elif loss_type == 'bce':
                return BinaryCrossEntropyLoss()
        elif model_type == 'prototypical':
            return PrototypicalLoss()
        raise ValueError(f"Cannot build criterion for config")

    def _get_dataloader_kwargs(self):
        """Build DataLoader keyword arguments from config.

        Returns dict with num_workers, prefetch_factor, and pin_memory.
        Defaults to num_workers=2 for parallel I/O on Linux/Colab.
        Windows auto-defaults to 0 unless explicitly overridden.
        """
        training_cfg = self.config.get('training', {})
        num_workers = training_cfg.get('num_workers', 2)

        # Windows uses 'spawn' which is slow — default to 0 unless explicit
        if sys.platform == 'win32' and 'num_workers' not in training_cfg:
            num_workers = 0

        kwargs = {
            'num_workers': num_workers,
            'pin_memory': training_cfg.get('pin_memory', True) and self.device.type == 'cuda',
        }

        if num_workers > 0:
            kwargs['prefetch_factor'] = training_cfg.get('prefetch_factor', 2)

        return kwargs

    # ── Siamese ──────────────────────────────────────────────────────────

    def _run_siamese_batch(self, images1, images2, labels, training=True):
        """Run a single Siamese forward/backward pass on pre-loaded tensors.

        Args:
            images1: (B, C, H, W) float tensor, already on device
            images2: (B, C, H, W) float tensor, already on device
            labels:  (B,) float tensor, already on device
            training: if True, run backward + optimizer step
        """
        if training:
            self.optimizer.zero_grad()

        output = self.model(images1, images2)

        if isinstance(self.criterion, ContrastiveLoss):
            loss = self.criterion(output['distance'], labels)
        else:
            loss = self.criterion(output['similarity'], labels)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        # Accuracy: use the SAME signal as the loss
        with torch.no_grad():
            if isinstance(self.criterion, ContrastiveLoss):
                preds = (output['distance'] < self.criterion.margin / 2).float()
            else:
                preds = (output['similarity'] > 0.5).float()
            correct = (preds == labels).sum().item()

        return loss.item(), correct, len(labels)

    def _run_siamese_epoch(self, sampler, dataset, num_iterations, training):
        """Shared Siamese epoch logic for train and validate.

        Pre-samples all pairs, wraps in SiamesePairDataset, and iterates
        via DataLoader for parallel I/O.
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        all_pairs = sampler.sample_epoch(num_iterations)
        pair_ds = SiamesePairDataset(all_pairs, dataset)
        loader = DataLoader(
            pair_ds,
            batch_size=sampler.batch_size,
            shuffle=False,  # already shuffled by sampler
            drop_last=False,
            **self._get_dataloader_kwargs(),
        )

        total_loss = 0
        total_correct = 0
        total_pairs = 0
        num_batches = 0

        ctx = torch.no_grad() if not training else _nullcontext()
        with ctx:
            for images1, images2, labels in loader:
                images1 = images1.to(self.device)
                images2 = images2.to(self.device)
                labels = labels.to(self.device)
                loss, correct, count = self._run_siamese_batch(
                    images1, images2, labels, training=training
                )
                total_loss += loss
                total_correct += correct
                total_pairs += count
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = total_correct / total_pairs if total_pairs > 0 else 0
        return avg_loss, accuracy

    def train_siamese_epoch(self, sampler, dataset, num_iterations=100):
        """Train one epoch for Siamese network."""
        return self._run_siamese_epoch(sampler, dataset, num_iterations, training=True)

    def validate_siamese_epoch(self, sampler, dataset, num_iterations=20):
        """Validate one epoch for Siamese network (no gradient updates)."""
        return self._run_siamese_epoch(sampler, dataset, num_iterations, training=False)

    # ── Prototypical ──────────────────────────────────────────────────────

    def _run_prototypical_batch(self, support_images, support_labels,
                                query_images, query_labels, training=True):
        """Run a single Prototypical episode on pre-loaded tensors.

        Args:
            support_images: (N_support, C, H, W) on device
            support_labels: (N_support,) LongTensor on device
            query_images:   (N_query, C, H, W) on device
            query_labels:   (N_query,) LongTensor on device
            training:       if True, run backward + optimizer step
        """
        if training:
            self.optimizer.zero_grad()

        output = self.model(support_images, support_labels, query_images)
        loss, acc = self.criterion(output['logits'], query_labels)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss.item(), acc

    def _run_prototypical_epoch(self, sampler, dataset, num_episodes, training):
        """Shared Prototypical epoch logic for train and validate.

        Flattens all episodes into a single DataLoader for efficient
        parallel loading, then reconstructs per-episode support/query
        boundaries from the flat output.
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        all_episodes = sampler.sample_epoch(num_episodes)

        # Flatten all episodes into one list for a single DataLoader pass.
        # Track sizes so we can reconstruct episode boundaries afterward.
        all_items = []           # flat list of (path, class_idx)
        episode_support_sizes = []
        episode_query_sizes = []
        for support_paths, query_paths in all_episodes:
            episode_support_sizes.append(len(support_paths))
            episode_query_sizes.append(len(query_paths))
            all_items.extend(support_paths)
            all_items.extend(query_paths)

        # Load all images in one DataLoader pass
        flat_ds = PrototypicalEpisodeDataset(all_items, dataset)
        loader = DataLoader(
            flat_ds,
            batch_size=len(all_items),  # load everything in one batch
            shuffle=False,
            **self._get_dataloader_kwargs(),
        )
        all_images, all_labels = next(iter(loader))
        all_images = all_images.to(self.device)
        all_labels = all_labels.to(self.device)

        # Reconstruct per-episode support/query sets and run model
        total_loss = 0
        total_accuracy = 0
        offset = 0

        ctx = torch.no_grad() if not training else _nullcontext()
        with ctx:
            for ep_idx in range(num_episodes):
                s_size = episode_support_sizes[ep_idx]
                q_size = episode_query_sizes[ep_idx]

                support_images = all_images[offset:offset + s_size]
                support_labels = all_labels[offset:offset + s_size]
                offset += s_size

                query_images = all_images[offset:offset + q_size]
                query_labels = all_labels[offset:offset + q_size]
                offset += q_size

                loss, acc = self._run_prototypical_batch(
                    support_images, support_labels,
                    query_images, query_labels,
                    training=training,
                )
                total_loss += loss
                total_accuracy += acc

        avg_loss = total_loss / num_episodes if num_episodes > 0 else 0
        avg_accuracy = total_accuracy / num_episodes if num_episodes > 0 else 0
        return avg_loss, avg_accuracy

    def train_prototypical_epoch(self, sampler, dataset, num_episodes=100):
        """Train one epoch for Prototypical network."""
        return self._run_prototypical_epoch(sampler, dataset, num_episodes, training=True)

    def validate_prototypical_epoch(self, sampler, dataset, num_episodes=20):
        """Validate one epoch for Prototypical network (no gradient updates)."""
        return self._run_prototypical_epoch(sampler, dataset, num_episodes, training=False)

    def save_checkpoint(self, filename=None, is_best=False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pth"

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        path = os.path.join(self.results_dir, 'checkpoints', filename)
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.results_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss',
                                              checkpoint.get('best_loss', float('inf')))
        self.model.to(self.device)
        print(f"[Trainer] Loaded checkpoint from epoch {self.epoch}")

    def train(self, dataset):
        """
        Full training loop with validation-based best-model selection.

        Args:
            dataset: BiometricDataset for training (split internally via split_subjects)
        """
        model_type = self.config['model']['type']
        epochs = self.config['training'].get('epochs', 100)
        patience = self.config['training'].get('patience', 15)
        iterations = self.config['training'].get('iterations_per_epoch', 100)
        val_iterations = max(iterations // 5, 10)  # 20% of training iters

        # Prepare transforms for toggling between train and val
        modality = self.config['dataset']['modality']
        train_transform = get_augmentation(modality, training=True)
        val_transform = get_augmentation(modality, training=False)

        # Ensure augmentation is set for training
        if dataset.transform is None:
            dataset.transform = train_transform
            print(f"[Trainer] Auto-applied {modality} training augmentation")

        # Create train and validation samplers
        train_data, val_data, _ = dataset.split_subjects()

        if model_type == 'siamese':
            batch_size = self.config['training'].get('batch_size', 32)
            train_sampler = PairSampler(train_data, batch_size=batch_size)
            val_sampler = PairSampler(val_data, batch_size=batch_size)
        else:
            n_way = self.config['training'].get('n_way', 5)
            k_shot = self.config['training'].get('k_shot', 5)
            q_query = self.config['training'].get('q_query', 5)
            train_sampler = EpisodeSampler(
                train_data, n_way=n_way, k_shot=k_shot, q_query=q_query
            )
            val_sampler = EpisodeSampler(
                val_data, n_way=n_way, k_shot=k_shot, q_query=q_query
            )



        print(f"\n{'='*60}")
        print(f"  Training {model_type.upper()} Network")
        print(f"  Epochs: {epochs} | Patience: {patience}")
        print(f"  Device: {self.device}")
        print(f"  Train subjects: {len(train_data)} | Val subjects: {len(val_data)}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            start_time = time.time()

            # ── Training pass (with augmentation) ──
            dataset.transform = train_transform
            if model_type == 'siamese':
                train_loss, train_acc = self.train_siamese_epoch(
                    train_sampler, dataset, num_iterations=iterations
                )
            else:
                train_loss, train_acc = self.train_prototypical_epoch(
                    train_sampler, dataset, num_episodes=iterations
                )

            # ── Validation pass (no augmentation) ──
            dataset.transform = val_transform
            if model_type == 'siamese':
                val_loss, val_acc = self.validate_siamese_epoch(
                    val_sampler, dataset, num_iterations=val_iterations
                )
            else:
                val_loss, val_acc = self.validate_prototypical_epoch(
                    val_sampler, dataset, num_episodes=val_iterations
                )

            elapsed = time.time() - start_time

            # Learning rate step
            if self.scheduler:
                self.scheduler.step()

            # Best model selected by VALIDATION loss
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"{'[BEST]' if is_best else ''} | "
                  f"{elapsed:.1f}s")

            # Early stopping on validation loss
            if self.patience_counter >= patience:
                print(f"\n[Early Stopping] No val improvement for {patience} epochs.")
                break

        print(f"\n{'='*60}")
        print(f"  Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
