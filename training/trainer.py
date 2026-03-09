"""
Unified training loop for both Siamese and Prototypical networks.
Driven by YAML configs.
"""

import os
import sys
import csv
import time
import yaml
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
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

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False


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

    def __init__(self, config_path, config_override=None):
        """
        Args:
            config_path: Path to YAML config file
            config_override: Optional pre-built config dict. If provided,
                             used instead of reading from config_path.
        """
        if config_override is not None:
            self.config = config_override
        else:
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
        os.makedirs(os.path.join(self.results_dir, 'figures'), exist_ok=True)

        # ── Mixed-precision (AMP) ────────────────────────────────────────
        self.use_amp = (
            self.config.get('training', {}).get('amp', False)
            and self.device.type == 'cuda'
        )
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.config.get('training', {}).get('amp', False):
            if self.use_amp:
                print("[AMP] Mixed precision enabled")
            else:
                print(f"[AMP] Disabled (non-CUDA device: {self.device})")

        # ── Gradient accumulation ────────────────────────────────────────
        self.accumulation_steps = self.config.get('training', {}).get(
            'accumulation_steps', 1
        )

        # ── CSV training log ─────────────────────────────────────────────
        self._csv_log_path = os.path.join(
            self.results_dir, 'logs', 'training_log.csv'
        )
        self._csv_columns = [
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'lr', 'is_best', 'elapsed_s', 'timestamp',
        ]
        self._init_csv_log()

        # ── TensorBoard ──────────────────────────────────────────────────
        self._tb_writer = None
        if self.config.get('training', {}).get('tensorboard', False):
            self._init_tensorboard()

    # ── Logging helpers ───────────────────────────────────────────────────

    def _init_csv_log(self):
        """Create CSV log file with header if it doesn't exist."""
        if not os.path.exists(self._csv_log_path):
            with open(self._csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self._csv_columns)

    def _init_tensorboard(self):
        """Initialize TensorBoard SummaryWriter with import guard."""
        if not _TENSORBOARD_AVAILABLE:
            print("[TensorBoard] WARNING: tensorboard not installed. "
                  "Run: pip install tensorboard")
            return
        tb_dir = os.path.join(self.results_dir, 'logs', 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=tb_dir)
        print(f"[TensorBoard] Logging to {tb_dir}")

    def _log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc,
                   lr, is_best, elapsed):
        """Append one row to CSV log and write to TensorBoard."""
        # CSV
        with open(self._csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}",
                f"{lr:.8f}", int(is_best), f"{elapsed:.2f}",
                datetime.now().isoformat(timespec='seconds'),
            ])

        # TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.add_scalar('Loss/train', train_loss, epoch)
            self._tb_writer.add_scalar('Loss/val', val_loss, epoch)
            self._tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
            self._tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
            self._tb_writer.add_scalar('LR', lr, epoch)

    def _plot_training_curves(self):
        """Read CSV log and generate training curve plot."""
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt

        # Read CSV
        epochs, t_loss, v_loss, t_acc, v_acc, best_epoch = [], [], [], [], [], None
        try:
            with open(self._csv_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row['epoch']))
                    t_loss.append(float(row['train_loss']))
                    v_loss.append(float(row['val_loss']))
                    t_acc.append(float(row['train_acc']))
                    v_acc.append(float(row['val_acc']))
                    if int(row['is_best']):
                        best_epoch = int(row['epoch'])
        except (FileNotFoundError, KeyError):
            print("[Plot] Could not read CSV log, skipping training curves.")
            return

        if len(epochs) < 2:
            print("[Plot] < 2 epochs logged, skipping training curves.")
            return

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Loss (left axis)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(epochs, t_loss, 'r-', alpha=0.7, label='Train Loss')
        ax1.plot(epochs, v_loss, 'r--', alpha=0.7, label='Val Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Accuracy (right axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(epochs, t_acc, 'b-', alpha=0.7, label='Train Acc')
        ax2.plot(epochs, v_acc, 'b--', alpha=0.7, label='Val Acc')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0, 1.05)

        # Best epoch marker
        if best_epoch is not None:
            ax1.axvline(x=best_epoch, color='green', linestyle=':',
                        alpha=0.8, label=f'Best (epoch {best_epoch})')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        fig.suptitle('Training Curves', fontsize=14)
        fig.tight_layout()

        plot_path = os.path.join(self.results_dir, 'figures',
                                 'training_curves.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[Plot] Training curves saved to {plot_path}")

    # ── Model / optimizer / scheduler / criterion builders ────────────────

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
            training: if True, run backward (optimizer step handled by caller)
        """
        autocast_ctx = (
            torch.amp.autocast('cuda') if self.use_amp
            else _nullcontext()
        )

        with autocast_ctx:
            output = self.model(images1, images2)

            if isinstance(self.criterion, ContrastiveLoss):
                loss = self.criterion(output['distance'], labels)
            else:
                loss = self.criterion(output['similarity'], labels)

        if training:
            scaled_loss = loss / self.accumulation_steps
            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

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
        via DataLoader for parallel I/O.  Supports gradient accumulation.
        """
        if training:
            self.model.train()
            self.optimizer.zero_grad()
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
            for batch_idx, (images1, images2, labels) in enumerate(loader):
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

                # Gradient accumulation: step every N batches
                if training and (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()

        # Flush any remaining accumulated gradients
        if training and num_batches % self.accumulation_steps != 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()

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
            training:       if True, run backward (optimizer step handled by caller)
        """
        autocast_ctx = (
            torch.amp.autocast('cuda') if self.use_amp
            else _nullcontext()
        )

        with autocast_ctx:
            output = self.model(support_images, support_labels, query_images)
            loss, acc = self.criterion(output['logits'], query_labels)

        if training:
            scaled_loss = loss / self.accumulation_steps
            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        return loss.item(), acc

    def _run_prototypical_epoch(self, sampler, dataset, num_episodes, training):
        """Shared Prototypical epoch logic for train and validate.

        Flattens all episodes into a single DataLoader for efficient
        parallel loading, then reconstructs per-episode support/query
        boundaries from the flat output.  Supports gradient accumulation.
        """
        if training:
            self.model.train()
            self.optimizer.zero_grad()
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

                # Gradient accumulation: step every N episodes
                if training and (ep_idx + 1) % self.accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()

        # Flush any remaining accumulated gradients
        if training and num_episodes % self.accumulation_steps != 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()

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

        # Save AMP scaler state for seamless resume
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save scheduler state for correct LR on resume
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save early stopping state
        checkpoint['patience_counter'] = self.patience_counter

        path = os.path.join(self.results_dir, 'checkpoints', filename)
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.results_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss',
                                              checkpoint.get('best_loss', float('inf')))

        # Restore AMP scaler state if available
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore scheduler state for correct LR on resume
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore early stopping state
        self.patience_counter = checkpoint.get('patience_counter', 0)

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

        start_epoch = self.epoch + 1  # resume-safe: starts at 1 if fresh, N+1 if resumed

        for epoch in range(start_epoch, epochs + 1):
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

            # Structured logging (CSV + TensorBoard)
            self._log_epoch(
                epoch, train_loss, train_acc, val_loss, val_acc,
                lr, is_best, elapsed,
            )

            # Early stopping on validation loss
            if self.patience_counter >= patience:
                print(f"\n[Early Stopping] No val improvement for {patience} epochs.")
                break

        # ── Post-training ────────────────────────────────────────────────
        # Generate training curve plot
        self._plot_training_curves()

        # Close TensorBoard writer
        if self._tb_writer is not None:
            self._tb_writer.close()

        print(f"\n{'='*60}")
        print(f"  Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
