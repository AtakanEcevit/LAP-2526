"""
Evaluation entry point.
Loads a trained model and runs full biometric evaluation.

Usage: python evaluate.py --config configs/siamese_signature.yaml --checkpoint results/checkpoints/best.pth
"""

import argparse
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm

from models.siamese import SiameseNetwork
from models.prototypical import PrototypicalNetwork
from data.signature_loader import CEDARDataset, BHSig260Dataset
from data.face_loader import ATTFaceDataset, LFWDataset
from data.fingerprint_loader import SOCOFingDataset
from data.augmentations import get_augmentation
from data.samplers import PairSampler, EpisodeSampler
from evaluation.metrics import compute_all_metrics
from evaluation.visualize import (
    plot_roc_curve, plot_det_curve, plot_score_distribution, plot_tsne
)
from utils import get_device


def get_dataset(config):
    """Create dataset (no augmentation for evaluation)."""
    modality = config['dataset']['modality']
    name = config['dataset']['name']
    root_dir = config['dataset']['root_dir']
    transform = get_augmentation(modality, training=False)

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
            return LFWDataset(root_dir, min_images=5, transform=transform)
    elif modality == 'fingerprint':
        if name == 'socofing':
            return SOCOFingDataset(root_dir, transform=transform)
    raise ValueError(f"Unknown dataset: {modality}/{name}")


def load_model(config, checkpoint_path, device):
    """Load trained model from checkpoint."""
    model_type = config['model']['type']
    backbone = config['model'].get('backbone', 'resnet')
    emb_dim = config['model'].get('embedding_dim', 128)
    in_channels = config['model'].get('in_channels', 1)

    if model_type == 'siamese':
        model = SiameseNetwork(
            backbone=backbone, embedding_dim=emb_dim,
            pretrained=False, in_channels=in_channels
        )
    else:
        distance = config['model'].get('distance', 'euclidean')
        model = PrototypicalNetwork(
            backbone=backbone, embedding_dim=emb_dim,
            pretrained=False, in_channels=in_channels, distance=distance
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def evaluate_siamese(model, dataset, test_data, device, k_shot=5,
                     num_pairs=500):
    """
    Evaluate Siamese network with k-shot enrollment per subject.
    
    For each subject, uses k genuine samples as support.
    Compares remaining genuine and forgery queries against each
    support sample and averages the similarity scores.
    
    Returns genuine and impostor similarity scores.
    """
    genuine_scores = []
    impostor_scores = []
    subjects = list(test_data.keys())

    for subj in tqdm(subjects, desc=f"Evaluating Siamese (k={k_shot})"):
        genuine = test_data[subj]['genuine']
        forgery = test_data[subj].get('forgery', [])

        if len(genuine) < k_shot + 1:
            continue

        # Support: k genuine enrollment samples
        support_paths = genuine[:k_shot]

        with torch.no_grad():
            # Preload support images
            support_imgs = []
            for sp in support_paths:
                img = torch.FloatTensor(dataset.load_image(sp)).unsqueeze(0)
                support_imgs.append(img)

            # Test remaining genuine
            for query_path in genuine[k_shot:]:
                query = torch.FloatTensor(
                    dataset.load_image(query_path)
                ).unsqueeze(0).to(device)
                scores = []
                for simg in support_imgs:
                    output = model(simg.to(device), query)
                    scores.append(output['similarity'].item())
                genuine_scores.append(np.mean(scores))

            # Test forgery samples
            for query_path in forgery:
                query = torch.FloatTensor(
                    dataset.load_image(query_path)
                ).unsqueeze(0).to(device)
                scores = []
                for simg in support_imgs:
                    output = model(simg.to(device), query)
                    scores.append(output['similarity'].item())
                impostor_scores.append(np.mean(scores))

            # Cross-subject negatives if no forgeries
            if len(forgery) == 0:
                other_subjects = [s for s in subjects if s != subj]
                for other_subj in np.random.choice(
                    other_subjects, min(5, len(other_subjects)), replace=False
                ):
                    other_genuine = test_data[other_subj]['genuine']
                    if other_genuine:
                        p = np.random.choice(other_genuine)
                        query = torch.FloatTensor(
                            dataset.load_image(p)
                        ).unsqueeze(0).to(device)
                        scores = []
                        for simg in support_imgs:
                            output = model(simg.to(device), query)
                            scores.append(output['similarity'].item())
                        impostor_scores.append(np.mean(scores))

    return genuine_scores, impostor_scores


def _proto_score(query_emb, prototype, distance_type):
    """Compute similarity score for Prototypical evaluation."""
    if distance_type == 'cosine':
        # Cosine similarity (embeddings already L2-normalized)
        sim = torch.mm(query_emb, prototype.t()).squeeze().item()
        return (sim + 1.0) / 2.0  # map [-1, 1] → [0, 1]
    else:
        # Euclidean: closer = higher score
        dist = torch.sqrt(((query_emb - prototype) ** 2).sum(dim=1) + 1e-8)
        return 1.0 / (1.0 + dist.item())


def evaluate_prototypical(model, dataset, test_data, device, k_shot=5,
                          num_episodes=100):
    """
    Evaluate Prototypical network with k-shot verification episodes.
    Respects model.distance_type (cosine vs euclidean).
    """
    genuine_scores = []
    impostor_scores = []
    subjects = list(test_data.keys())
    distance_type = getattr(model, 'distance_type', 'euclidean')

    for subj in tqdm(subjects, desc=f"Evaluating Prototypical (k={k_shot})"):
        genuine = test_data[subj]['genuine']
        forgery = test_data[subj].get('forgery', [])

        if len(genuine) < k_shot + 1:
            continue

        # Support set: k genuine samples
        support_imgs = []
        for p in genuine[:k_shot]:
            img = torch.FloatTensor(dataset.load_image(p))
            support_imgs.append(img)
        support = torch.stack(support_imgs).to(device)

        with torch.no_grad():
            support_emb = model.encoder(support)
            prototype = support_emb.mean(dim=0, keepdim=True)

            # Test remaining genuine samples
            for p in genuine[k_shot:]:
                img = torch.FloatTensor(dataset.load_image(p)).unsqueeze(0).to(device)
                query_emb = model.encoder(img)
                score = _proto_score(query_emb, prototype, distance_type)
                genuine_scores.append(score)

            # Test forgery samples
            for p in forgery:
                img = torch.FloatTensor(dataset.load_image(p)).unsqueeze(0).to(device)
                query_emb = model.encoder(img)
                score = _proto_score(query_emb, prototype, distance_type)
                impostor_scores.append(score)

            # Cross-subject negatives (if not enough forgeries)
            if len(forgery) == 0:
                other_subjects = [s for s in subjects if s != subj]
                for other_subj in np.random.choice(other_subjects,
                                                    min(5, len(other_subjects)),
                                                    replace=False):
                    other_genuine = test_data[other_subj]['genuine']
                    if other_genuine:
                        p = np.random.choice(other_genuine)
                        img = torch.FloatTensor(dataset.load_image(p)).unsqueeze(0).to(device)
                        query_emb = model.encoder(img)
                        score = _proto_score(query_emb, prototype, distance_type)
                        impostor_scores.append(score)

    return genuine_scores, impostor_scores


def collect_embeddings(model, dataset, test_data, device, max_per_subject=5):
    """Collect embeddings for t-SNE visualization."""
    embeddings = []
    labels = []
    label_map = {}

    for idx, subj in enumerate(list(test_data.keys())[:20]):  # Max 20 subjects
        genuine = test_data[subj]['genuine'][:max_per_subject]
        label_map[idx] = subj

        for p in genuine:
            img = torch.FloatTensor(dataset.load_image(p)).unsqueeze(0).to(device)
            with torch.no_grad():
                if hasattr(model, 'get_embedding'):
                    emb = model.get_embedding(img)
                else:
                    emb = model.encoder(img)
            embeddings.append(emb.cpu().numpy().squeeze())
            labels.append(idx)

    return np.array(embeddings), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate biometric verification model")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k-shot', type=int, default=None,
                        help='Override k-shot from config')
    parser.add_argument('--num-pairs', type=int, default=500,
                        help='Number of test pairs for Siamese evaluation')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    model_type = config['model']['type']
    results_dir = config.get('results_dir', 'results')

    print(f"\n{'='*60}")
    print(f"  Biometric Verification — Evaluation")
    print(f"  Model:      {model_type}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    # Load model and dataset
    model = load_model(config, args.checkpoint, device)
    dataset = get_dataset(config)
    _, _, test_data = dataset.split_subjects()

    # Evaluate for each k-shot
    k_shots = [args.k_shot] if args.k_shot else config.get('evaluation', {}).get('k_shots', [5])

    for k in k_shots:
        print(f"\n{'—'*40}")
        print(f"  Evaluating with k_shot = {k}")
        print(f"{'—'*40}")

        if model_type == 'siamese':
            genuine, impostor = evaluate_siamese(
                model, dataset, test_data, device,
                k_shot=k, num_pairs=args.num_pairs
            )
        else:
            genuine, impostor = evaluate_prototypical(
                model, dataset, test_data, device, k_shot=k
            )

        if len(genuine) == 0 or len(impostor) == 0:
            print("  [WARNING] Not enough scores to compute metrics. Skipping.")
            continue

        # Compute metrics
        metrics = compute_all_metrics(genuine, impostor)

        print(f"\n  Results (k={k}):")
        print(f"    Accuracy:     {metrics['accuracy']:.4f}")
        print(f"    EER:          {metrics['eer']:.4f}")
        print(f"    FAR:          {metrics['far']:.4f}")
        print(f"    FRR:          {metrics['frr']:.4f}")
        print(f"    AUC:          {metrics['auc']:.4f}")
        print(f"    d-prime:      {metrics['d_prime']:.4f}")
        print(f"    FRR@FAR=0.1%: {metrics['frr_at_far_01']:.4f}")

        # Generate plots
        fig_dir = os.path.join(results_dir, 'figures', f'kshot_{k}')
        os.makedirs(fig_dir, exist_ok=True)

        plot_roc_curve(genuine, impostor,
                       title=f"ROC — {model_type} (k={k})",
                       save_path=os.path.join(fig_dir, 'roc.png'))
        plot_det_curve(genuine, impostor,
                       title=f"DET — {model_type} (k={k})",
                       save_path=os.path.join(fig_dir, 'det.png'))
        plot_score_distribution(genuine, impostor,
                                title=f"Scores — {model_type} (k={k})",
                                save_path=os.path.join(fig_dir, 'scores.png'))

        print(f"    Plots saved to: {fig_dir}")

    # t-SNE visualization
    print(f"\nGenerating t-SNE embeddings...")
    embeddings, labels = collect_embeddings(model, dataset, test_data, device)
    if len(embeddings) > 10:
        plot_tsne(embeddings, labels,
                  title=f"t-SNE — {model_type}",
                  save_path=os.path.join(results_dir, 'figures', 'tsne.png'))
        print(f"  t-SNE saved to: {results_dir}/figures/tsne.png")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
