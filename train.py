"""
Training entry point.
Usage: python train.py --config configs/siamese_signature.yaml
"""

import argparse
import yaml
import os
import sys

from training.trainer import Trainer
from data.dataset_factory import get_dataset


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
