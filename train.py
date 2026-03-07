"""
Training entry point.
Usage: python train.py --config configs/siamese_signature.yaml
"""

import argparse
import yaml
import os
import sys

from training.trainer import Trainer
from data.signature_loader import CEDARDataset, BHSig260Dataset
from data.face_loader import ATTFaceDataset, LFWDataset
from data.fingerprint_loader import SOCOFingDataset
from data.augmentations import get_augmentation


def get_dataset(config):
    """Create dataset based on config."""
    modality = config['dataset']['modality']
    name = config['dataset']['name']
    root_dir = config['dataset']['root_dir']

    transform = get_augmentation(modality, training=True)

    if modality == 'signature':
        if name == 'cedar':
            return CEDARDataset(root_dir, transform=transform)
        elif name == 'bhsig260':
            script = config['dataset'].get('script', 'Bengali')
            return BHSig260Dataset(root_dir, script=script, transform=transform)
    elif modality == 'face':
        if name == 'att':
            return ATTFaceDataset(root_dir, transform=transform)
        elif name == 'lfw':
            min_imgs = config['dataset'].get('min_images', 5)
            return LFWDataset(root_dir, min_images=min_imgs, transform=transform)
    elif modality == 'fingerprint':
        if name == 'socofing':
            return SOCOFingDataset(root_dir, transform=transform)

    raise ValueError(f"Unknown dataset: {modality}/{name}")


def main():
    parser = argparse.ArgumentParser(description="Train biometric verification model")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"  Biometric Few-Shot Verification — Training")
    print(f"  Model:    {config['model']['type']}")
    print(f"  Dataset:  {config['dataset']['modality']}/{config['dataset']['name']}")
    print(f"  Config:   {args.config}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = get_dataset(config)

    # Create trainer
    trainer = Trainer(args.config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(dataset)


if __name__ == "__main__":
    main()
