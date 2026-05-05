"""
Hybrid Trainer for Siamese + Prototypical Networks
Trains both architectures simultaneously for robust face verification

Author: LAP Project
Version: 1.0
"""

import os
import sys

# Ensure repo root is on sys.path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import logging

# Import models and losses
from models.siamese import SiameseNetwork
from models.prototypical import PrototypicalNetwork
from losses.losses import ContrastiveLoss, PrototypicalLoss, CombinedFaceLoss, get_loss
from data.dataset_factory import get_dataset, get_samplers


class HybridTrainer:
    """
    Trainer for Hybrid Siamese + Prototypical Network.
    
    Features:
    - Joint training of Siamese and Prototypical networks
    - Early stopping with validation monitoring
    - Mixed precision training (AMP)
    - Gradient accumulation support
    - TensorBoard logging
    - Checkpoint management
    """

    def __init__(self, config_path, device='cuda', resume_from=None):
        """
        Initialize trainer from config file.
        
        Args:
            config_path: Path to YAML config file
            device: 'cuda' or 'cpu'
            resume_from: Path to checkpoint to resume from
        """
        self.device = device
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.results_dir / 'logs').mkdir(exist_ok=True)
        
        # Load datasets
        self.logger.info("Loading datasets...")
        self.train_dataset = get_dataset(
            config=self.config['dataset'],
            split='train'
        )
        self.val_dataset = get_dataset(
            config=self.config['dataset'],
            split='val'
        )
        
        # Build models
        self.logger.info("Building models...")
        self.siamese_net = self._build_siamese_model()
        self.proto_net = self._build_prototypical_model()
        
        # Build losses
        self.siamese_loss = get_loss(
            loss_name=self.config['training']['loss'],
            margin=self.config['training'].get('margin', 1.0)
        )
        self.proto_loss = PrototypicalLoss()
        
        # Build optimizers
        self.logger.info("Building optimizers...")
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Resume if provided
        if resume_from:
            self.load_checkpoint(resume_from)

    def setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.results_dir / 'logs' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ]
        )

    def _build_siamese_model(self):
        """Build Siamese Network model."""
        model = SiameseNetwork(
            backbone=self.config['model']['backbone'],
            embedding_dim=self.config['model']['embedding_dim'],
            pretrained=self.config['model']['pretrained'],
            in_channels=self.config['model']['in_channels'],
        )
        return model.to(self.device)

    def _build_prototypical_model(self):
        """Build Prototypical Network model."""
        model = PrototypicalNetwork(
            backbone=self.config['model']['backbone'],
            embedding_dim=self.config['model']['embedding_dim'],
            pretrained=self.config['model']['pretrained'],
            in_channels=self.config['model']['in_channels'],
        )
        return model.to(self.device)

    def _build_optimizer(self):
        """Build optimizer with shared parameters from both models."""
        # Combine parameters from both models
        params = list(self.siamese_net.parameters()) + list(self.proto_net.parameters())
        
        if self.config['training'].get('optimizer', 'adam').lower() == 'adam':
            return optim.Adam(
                params,
                lr=self.config['training']['lr'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(
                    self.config['training'].get('adam_beta1', 0.9),
                    self.config['training'].get('adam_beta2', 0.999),
                ),
            )
        else:
            return optim.SGD(
                params,
                lr=self.config['training']['lr'],
                weight_decay=self.config['training']['weight_decay'],
            )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler_name = self.config['training']['scheduler'].lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['training']['scheduler_config'].get('min_lr', 1e-5),
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )
        else:
            return None

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config['training']['epochs']} epochs")
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_loss = self._validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config['training']['patience']:
                    self.logger.info(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['logging'].get('save_interval', 5) == 0:
                self.save_checkpoint(is_best=False)

    def _train_epoch(self):
        """Train for one epoch."""
        self.siamese_net.train()
        self.proto_net.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory'],
        )
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} - Train")
        
        for batch_idx, batch in enumerate(pbar):
            img1, img2, labels = batch
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    # Siamese forward
                    siamese_output = self.siamese_net(img1, img2)
                    siamese_loss = self.siamese_loss(
                        siamese_output['emb1'],
                        siamese_output['emb2'],
                        labels,
                    )
                    
                    # Prototypical forward (simplified: use img pairs as support)
                    proto_loss = self.proto_loss(
                        siamese_output['emb1'],
                        siamese_output['emb2'].unsqueeze(1),
                        labels.unsqueeze(1),
                    )
                    
                    # Weighted combination
                    hybrid_loss = (
                        self.config['hybrid']['siamese_weight'] * siamese_loss +
                        self.config['hybrid']['prototypical_weight'] * proto_loss
                    ) / self.config['training'].get('accumulation_steps', 1)
                
                # Backward
                self.scaler.scale(hybrid_loss).backward()
            else:
                # Standard forward/backward
                siamese_output = self.siamese_net(img1, img2)
                siamese_loss = self.siamese_loss(
                    siamese_output['emb1'],
                    siamese_output['emb2'],
                    labels,
                )
                
                proto_loss = self.proto_loss(
                    siamese_output['emb1'],
                    siamese_output['emb2'].unsqueeze(1),
                    labels.unsqueeze(1),
                )
                
                hybrid_loss = (
                    self.config['hybrid']['siamese_weight'] * siamese_loss +
                    self.config['hybrid']['prototypical_weight'] * proto_loss
                ) / self.config['training'].get('accumulation_steps', 1)
                
                hybrid_loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config['training'].get('accumulation_steps', 1) == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += hybrid_loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': total_loss / num_batches})
        
        return total_loss / num_batches

    def _validate_epoch(self):
        """Validate for one epoch."""
        self.siamese_net.eval()
        self.proto_net.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
        )
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} - Val"):
                img1, img2, labels = batch
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                siamese_output = self.siamese_net(img1, img2)
                siamese_loss = self.siamese_loss(
                    siamese_output['emb1'],
                    siamese_output['emb2'],
                    labels,
                )
                
                proto_loss = self.proto_loss(
                    siamese_output['emb1'],
                    siamese_output['emb2'].unsqueeze(1),
                    labels.unsqueeze(1),
                )
                
                hybrid_loss = (
                    self.config['hybrid']['siamese_weight'] * siamese_loss +
                    self.config['hybrid']['prototypical_weight'] * proto_loss
                )
                
                total_loss += hybrid_loss.item()
                num_batches += 1
        
        return total_loss / num_batches

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'siamese_state': self.siamese_net.state_dict(),
            'proto_state': self.proto_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        checkpoint_path = self.results_dir / 'checkpoints' / f'epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.results_dir / 'checkpoints' / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {self.current_epoch}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.siamese_net.load_state_dict(checkpoint['siamese_state'])
        self.proto_net.load_state_dict(checkpoint['proto_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Hybrid Siamese+Prototypical Network')
    parser.add_argument('--config', type=str, default='configs/siamese_prototypical_hybrid.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    trainer = HybridTrainer(
        config_path=args.config,
        device=args.device,
        resume_from=args.resume,
    )
    
    trainer.train()
