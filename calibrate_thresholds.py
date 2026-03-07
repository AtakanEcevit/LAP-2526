#!/usr/bin/env python3
"""
Threshold Calibration Script

Runs evaluation on test data for each trained model and computes the
EER-optimal decision threshold. Writes results to
inference/calibrated_thresholds.json so the inference engine can
use accurate thresholds instead of the default 0.5.

Usage:
    python calibrate_thresholds.py
    python calibrate_thresholds.py --device cpu
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from tqdm import tqdm

from inference.engine import VerificationEngine
from inference.config import MODEL_REGISTRY, _PROJECT_ROOT
from evaluation.metrics import compute_eer, compute_all_metrics

# Reuse dataset loaders from the training pipeline
from data.dataset_factory import get_dataset as _get_dataset_base
from utils import get_device


# ── Dataset factory ───────────────────────────────────────────────────

def _get_dataset(modality, config_data):
    """Create dataset without augmentation for evaluation."""
    import yaml
    with open(config_data, "r") as f:
        config = yaml.safe_load(f)

    # Make root_dir absolute relative to project root
    root_dir = config["dataset"]["root_dir"]
    if not os.path.isabs(root_dir):
        config["dataset"]["root_dir"] = os.path.join(_PROJECT_ROOT, root_dir)

    return _get_dataset_base(config, training=False)


# ── Score collection ──────────────────────────────────────────────────

def _collect_scores_siamese(engine, dataset, test_data, device, num_pairs=500):
    """Generate genuine/impostor similarity scores for Siamese models."""
    from data.samplers import PairSampler

    genuine_scores = []
    impostor_scores = []

    sampler = PairSampler(test_data, batch_size=num_pairs, neg_ratio=0.5)
    batch = sampler.sample_batch()

    with torch.no_grad():
        for path1, path2, label in batch:
            img1 = torch.FloatTensor(dataset.load_image(path1)).unsqueeze(0).to(device)
            img2 = torch.FloatTensor(dataset.load_image(path2)).unsqueeze(0).to(device)

            output = engine.model(img1, img2)
            score = output["similarity"].item()

            if label == 1:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    return genuine_scores, impostor_scores


def _collect_scores_prototypical(engine, dataset, test_data, device, k_shot=5):
    """Generate genuine/impostor similarity scores for Prototypical models."""
    genuine_scores = []
    impostor_scores = []
    subjects = list(test_data.keys())

    for subj in subjects:
        genuine = test_data[subj]["genuine"]
        forgery = test_data[subj].get("forgery", [])

        if len(genuine) < k_shot + 1:
            continue

        # Build support set
        support_imgs = []
        for p in genuine[:k_shot]:
            img = torch.FloatTensor(dataset.load_image(p))
            support_imgs.append(img)
        support = torch.stack(support_imgs).to(device)

        with torch.no_grad():
            support_emb = engine.model.encoder(support)
            prototype = support_emb.mean(dim=0, keepdim=True)

            # Test remaining genuine
            for p in genuine[k_shot:]:
                img = torch.FloatTensor(dataset.load_image(p)).unsqueeze(0).to(device)
                query_emb = engine.model.encoder(img)
                dist = torch.sqrt(((query_emb - prototype) ** 2).sum(dim=1) + 1e-8)
                score = 1.0 / (1.0 + dist.item())
                genuine_scores.append(score)

            # Test forgeries
            for p in forgery:
                img = torch.FloatTensor(dataset.load_image(p)).unsqueeze(0).to(device)
                query_emb = engine.model.encoder(img)
                dist = torch.sqrt(((query_emb - prototype) ** 2).sum(dim=1) + 1e-8)
                score = 1.0 / (1.0 + dist.item())
                impostor_scores.append(score)

            # Cross-subject negatives if no forgeries
            if len(forgery) == 0:
                other_subjects = [s for s in subjects if s != subj]
                for other_subj in np.random.choice(
                    other_subjects, min(5, len(other_subjects)), replace=False
                ):
                    other_genuine = test_data[other_subj]["genuine"]
                    if other_genuine:
                        p = np.random.choice(other_genuine)
                        img = torch.FloatTensor(dataset.load_image(p)).unsqueeze(0).to(device)
                        query_emb = engine.model.encoder(img)
                        dist = torch.sqrt(
                            ((query_emb - prototype) ** 2).sum(dim=1) + 1e-8
                        )
                        score = 1.0 / (1.0 + dist.item())
                        impostor_scores.append(score)

    return genuine_scores, impostor_scores


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Calibrate EER thresholds for all models")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    args = parser.parse_args()

    device = args.device
    if device:
        device = torch.device(device)
    else:
        device = get_device()

    results = {}
    summary_rows = []

    print(f"\n{'='*70}")
    print(f"  Threshold Calibration -- Computing EER for all 6 models")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    for (modality, model_type), entry in MODEL_REGISTRY.items():
        key_str = f"{modality}/{model_type}"
        config_path = entry["config"]
        checkpoint_path = entry["checkpoint"]

        if not os.path.exists(checkpoint_path):
            print(f"  [SKIP] {key_str} -- checkpoint not found: {checkpoint_path}")
            continue

        print(f"  Calibrating {key_str}...")

        # Load model
        engine = VerificationEngine()
        engine.load(modality, model_type, device=device)

        # Load dataset and get test split
        try:
            dataset = _get_dataset(modality, config_path)
            _, _, test_data = dataset.split_subjects()
        except Exception as e:
            print(f"  [SKIP] {key_str} -- dataset error: {e}")
            continue

        if not test_data:
            print(f"  [SKIP] {key_str} -- no test data available")
            continue

        # Collect scores
        if model_type == "siamese":
            genuine, impostor = _collect_scores_siamese(
                engine, dataset, test_data, device, num_pairs=500
            )
        else:
            genuine, impostor = _collect_scores_prototypical(
                engine, dataset, test_data, device, k_shot=5
            )

        if len(genuine) < 5 or len(impostor) < 5:
            print(f"  [SKIP] {key_str} -- insufficient scores "
                  f"(genuine={len(genuine)}, impostor={len(impostor)})")
            continue

        # Compute EER
        metrics = compute_all_metrics(genuine, impostor)
        eer = metrics["eer"]
        eer_threshold = metrics["eer_threshold"]
        accuracy = metrics["accuracy"]
        auc = metrics["auc"]

        results[f"{modality},{model_type}"] = round(float(eer_threshold), 6)

        summary_rows.append({
            "key": key_str,
            "eer": eer,
            "threshold": eer_threshold,
            "accuracy": accuracy,
            "auc": auc,
            "genuine_n": len(genuine),
            "impostor_n": len(impostor),
        })

        print(f"    EER={eer:.4f}  Threshold={eer_threshold:.6f}  "
              f"Acc={accuracy:.4f}  AUC={auc:.4f}  "
              f"(G={len(genuine)}, I={len(impostor)})")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  CALIBRATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<30s} {'EER':>8s} {'Threshold':>12s} "
          f"{'Acc':>8s} {'AUC':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*8} {'-'*8}")
    for row in summary_rows:
        print(f"  {row['key']:<30s} {row['eer']:>8.4f} "
              f"{row['threshold']:>12.6f} {row['accuracy']:>8.4f} "
              f"{row['auc']:>8.4f}")
    print(f"{'='*70}\n")

    # Write calibrated thresholds
    output_path = os.path.join(_PROJECT_ROOT, "inference", "calibrated_thresholds.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Calibrated thresholds saved to: {output_path}")
    print(f"  These will be auto-loaded by inference/config.py on next import.\n")


if __name__ == "__main__":
    main()
