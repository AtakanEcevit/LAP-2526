"""
Visualization tools for biometric verification results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, det_curve
from sklearn.manifold import TSNE


def plot_roc_curve(genuine_scores, impostor_scores, title="ROC Curve",
                   save_path=None):
    """
    Plot Receiver Operating Characteristic curve.
    """
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate (FAR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (1 - FRR)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_det_curve(genuine_scores, impostor_scores, title="DET Curve",
                   save_path=None):
    """
    Plot Detection Error Tradeoff curve.
    Standard in biometric evaluation (NIST).
    """
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, fnr, _ = det_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr * 100, fnr * 100, 'b-', linewidth=2)
    ax.set_xlabel('False Acceptance Rate (%)', fontsize=12)
    ax.set_ylabel('False Rejection Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_score_distribution(genuine_scores, impostor_scores,
                            title="Score Distribution", save_path=None):
    """
    Plot histogram of genuine vs impostor score distributions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(genuine_scores, bins=50, alpha=0.6, color='green',
            label='Genuine', density=True)
    ax.hist(impostor_scores, bins=50, alpha=0.6, color='red',
            label='Impostor', density=True)

    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_tsne(embeddings, labels, title="t-SNE Embeddings",
              save_path=None, n_classes_show=10):
    """
    Plot t-SNE visualization of embeddings.
    
    Args:
        embeddings: (N, embedding_dim) numpy array
        labels: (N,) class labels
        n_classes_show: max number of classes to visualize (for clarity)
    """
    # Limit classes for readability
    unique_labels = np.unique(labels)
    if len(unique_labels) > n_classes_show:
        selected = np.random.choice(unique_labels, n_classes_show, replace=False)
        mask = np.isin(labels, selected)
        embeddings = embeddings[mask]
        labels = labels[mask]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap='tab20', alpha=0.7, s=30
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    plt.colorbar(scatter, label='Class')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_comparison_bar(results_dict, metric='accuracy',
                        title=None, save_path=None):
    """
    Bar chart comparing models across configurations.
    
    Args:
        results_dict: dict of {config_name: {metric_name: value}}
        metric: which metric to plot
    """
    configs = list(results_dict.keys())
    values = [results_dict[c][metric] for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("husl", len(configs))
    bars = ax.bar(range(len(configs)), values, color=colors)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title or f"{metric.upper()} Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig
