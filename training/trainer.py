"""
Unified training loop for both Siamese and Prototypical networks.
Driven by YAML configs.
"""

import os
import time
import yaml
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models.siamese import SiameseNetwork
from models.prototypical import PrototypicalNetwork
from losses.losses import ContrastiveLoss, PrototypicalLoss, BinaryCrossEntropyLoss
from data.samplers import PairSampler, EpisodeSampler
from data.augmentations import get_augmentation
from utils import get_device


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

        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
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

    def train_siamese_epoch(self, sampler, dataset, num_iterations=100):
        """Train one epoch for Siamese network."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_pairs = 0

        for i in range(num_iterations):
            batch = sampler.sample_batch()

            images1, images2, labels = [], [], []
            for path1, path2, label in batch:
                img1 = dataset.load_image(path1)
                img2 = dataset.load_image(path2)
                images1.append(img1)
                images2.append(img2)
                labels.append(label)

            images1 = torch.FloatTensor(np.stack(images1)).to(self.device)
            images2 = torch.FloatTensor(np.stack(images2)).to(self.device)
            labels = torch.FloatTensor(labels).to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images1, images2)

            if isinstance(self.criterion, ContrastiveLoss):
                loss = self.criterion(output['distance'], labels)
            else:
                loss = self.criterion(output['similarity'], labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Accuracy
            with torch.no_grad():
                preds = (output['similarity'] > 0.5).float()
                total_correct += (preds == labels).sum().item()
                total_pairs += len(labels)

        avg_loss = total_loss / num_iterations
        accuracy = total_correct / total_pairs if total_pairs > 0 else 0
        return avg_loss, accuracy

    def train_prototypical_epoch(self, sampler, dataset, num_episodes=100):
        """Train one epoch for Prototypical network."""
        self.model.train()
        total_loss = 0
        total_accuracy = 0

        for i in range(num_episodes):
            support_paths, query_paths = sampler.sample_episode()

            # Load support images
            support_images = []
            support_labels = []
            for path, class_idx in support_paths:
                img = dataset.load_image(path)
                support_images.append(img)
                support_labels.append(class_idx)

            # Load query images
            query_images = []
            query_labels = []
            for path, class_idx in query_paths:
                img = dataset.load_image(path)
                query_images.append(img)
                query_labels.append(class_idx)

            support_images = torch.FloatTensor(np.stack(support_images)).to(self.device)
            support_labels = torch.LongTensor(support_labels).to(self.device)
            query_images = torch.FloatTensor(np.stack(query_images)).to(self.device)
            query_labels = torch.LongTensor(query_labels).to(self.device)

            self.optimizer.zero_grad()
            output = self.model(support_images, support_labels, query_images)
            loss, acc = self.criterion(output['logits'], query_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_accuracy += acc

        avg_loss = total_loss / num_episodes
        avg_accuracy = total_accuracy / num_episodes
        return avg_loss, avg_accuracy

    def save_checkpoint(self, filename=None, is_best=False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pth"

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
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
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.model.to(self.device)
        print(f"[Trainer] Loaded checkpoint from epoch {self.epoch}")

    def train(self, dataset, val_dataset=None):
        """
        Full training loop.
        
        Args:
            dataset: BiometricDataset for training
            val_dataset: Optional validation dataset
        """
        model_type = self.config['model']['type']
        epochs = self.config['training'].get('epochs', 100)
        patience = self.config['training'].get('patience', 15)
        iterations = self.config['training'].get('iterations_per_epoch', 100)

        # Create sampler
        train_data, val_data, _ = dataset.split_subjects()

        if model_type == 'siamese':
            batch_size = self.config['training'].get('batch_size', 32)
            sampler = PairSampler(train_data, batch_size=batch_size)
        else:
            n_way = self.config['training'].get('n_way', 5)
            k_shot = self.config['training'].get('k_shot', 5)
            q_query = self.config['training'].get('q_query', 5)
            sampler = EpisodeSampler(
                train_data, n_way=n_way, k_shot=k_shot, q_query=q_query
            )

        print(f"\n{'='*60}")
        print(f"  Training {model_type.upper()} Network")
        print(f"  Epochs: {epochs} | Patience: {patience}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            start_time = time.time()

            if model_type == 'siamese':
                loss, acc = self.train_siamese_epoch(
                    sampler, dataset, num_iterations=iterations
                )
            else:
                loss, acc = self.train_prototypical_epoch(
                    sampler, dataset, num_episodes=iterations
                )

            elapsed = time.time() - start_time

            # Learning rate step
            if self.scheduler:
                self.scheduler.step()

            # Check best model
            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Acc: {acc:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"{'[BEST]' if is_best else ''} | "
                  f"{elapsed:.1f}s")

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs.")
                break

        print(f"\n{'='*60}")
        print(f"  Training complete. Best loss: {self.best_loss:.4f}")
        print(f"{'='*60}")
