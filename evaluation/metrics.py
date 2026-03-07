"""
Evaluation metrics for biometric verification.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score


def compute_eer(genuine_scores, impostor_scores):
    """
    Compute Equal Error Rate (EER).
    EER is the point where False Acceptance Rate = False Rejection Rate.
    
    Args:
        genuine_scores: similarity scores for genuine pairs (higher = more similar)
        impostor_scores: similarity scores for impostor pairs
        
    Returns:
        eer: Equal Error Rate (float)
        eer_threshold: threshold at EER
    """
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Find intersection point of FPR and FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return eer, eer_threshold


def compute_far_frr(genuine_scores, impostor_scores, threshold):
    """
    Compute FAR and FRR at a given threshold.
    
    Args:
        genuine_scores: similarity scores for genuine pairs
        impostor_scores: similarity scores for impostor pairs
        threshold: decision threshold (score >= threshold → accept)
        
    Returns:
        far: False Acceptance Rate (impostor accepted / total impostor)
        frr: False Rejection Rate (genuine rejected / total genuine)
    """
    far = np.mean(np.array(impostor_scores) >= threshold)
    frr = np.mean(np.array(genuine_scores) < threshold)
    return float(far), float(frr)


def compute_far_at_threshold(impostor_scores, target_far=0.001):
    """
    Find the threshold that achieves a target FAR.
    
    Args:
        impostor_scores: similarity scores for impostor pairs
        target_far: desired FAR (e.g., 0.001 = 0.1%)
        
    Returns:
        threshold: threshold achieving desired FAR
    """
    sorted_scores = np.sort(impostor_scores)[::-1]
    idx = int(len(sorted_scores) * target_far)
    idx = min(idx, len(sorted_scores) - 1)
    return sorted_scores[idx]


def compute_auc(genuine_scores, impostor_scores):
    """
    Compute Area Under ROC Curve.
    
    Returns:
        auc_score: float ∈ [0, 1], higher is better
    """
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])
    return roc_auc_score(labels, scores)


def compute_d_prime(genuine_scores, impostor_scores):
    """
    Compute d-prime (d') — a sensitivity measure.
    
    d' = |μ_genuine - μ_impostor| / sqrt(0.5 * (σ²_genuine + σ²_impostor))
    
    Higher d' → better separation between genuine and impostor distributions.
    """
    mu_g = np.mean(genuine_scores)
    mu_i = np.mean(impostor_scores)
    var_g = np.var(genuine_scores)
    var_i = np.var(impostor_scores)

    denominator = np.sqrt(0.5 * (var_g + var_i))
    if denominator < 1e-10:
        return 0.0

    return abs(mu_g - mu_i) / denominator


def compute_accuracy(genuine_scores, impostor_scores, threshold=None):
    """
    Compute overall classification accuracy.
    If no threshold given, uses the EER threshold.
    """
    if threshold is None:
        _, threshold = compute_eer(genuine_scores, impostor_scores)

    genuine_correct = np.sum(np.array(genuine_scores) >= threshold)
    impostor_correct = np.sum(np.array(impostor_scores) < threshold)
    total = len(genuine_scores) + len(impostor_scores)

    return (genuine_correct + impostor_correct) / total


def compute_all_metrics(genuine_scores, impostor_scores):
    """
    Compute all biometric verification metrics at once.
    
    Returns:
        dict with: accuracy, eer, far, frr, far_at_01, auc, d_prime
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    far, frr = compute_far_frr(genuine_scores, impostor_scores, eer_threshold)
    auc_score = compute_auc(genuine_scores, impostor_scores)
    d_prime = compute_d_prime(genuine_scores, impostor_scores)
    accuracy = compute_accuracy(genuine_scores, impostor_scores, eer_threshold)

    # FAR at 0.1% threshold
    threshold_01 = compute_far_at_threshold(impostor_scores, target_far=0.001)
    _, frr_at_01 = compute_far_frr(genuine_scores, impostor_scores, threshold_01)

    return {
        'accuracy': accuracy,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'far': far,
        'frr': frr,
        'frr_at_far_01': frr_at_01,
        'auc': auc_score,
        'd_prime': d_prime,
    }
