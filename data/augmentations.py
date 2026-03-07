"""
Per-modality augmentation pipelines using Albumentations.

Note: Uses translate_percent (not shift_limit) for Albumentations 2.0+ compatibility.
"""

import albumentations as A


def get_signature_augmentation(training=True):
    """
    Signature augmentation pipeline.
    Light augmentations to simulate natural writing variation
    without destroying signature structure.
    """
    if training:
        return A.Compose([
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1), rotate=(-5, 5),
                border_mode=0, p=0.5
            ),
            A.ElasticTransform(alpha=20, sigma=5, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
        ])
    return A.Compose([])  # No augmentation for val/test


def get_face_augmentation(training=True):
    """
    Face augmentation pipeline.
    Simulates lighting changes, slight pose variations, and noise.
    """
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05), rotate=(-10, 10),
                border_mode=0, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])
    return A.Compose([])


def get_fingerprint_augmentation(training=True):
    """
    Fingerprint augmentation pipeline.
    Conservative augmentations — fingerprint ridge patterns are fragile.
    """
    if training:
        return A.Compose([
            A.Affine(
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                scale=(0.95, 1.05), rotate=(-8, 8),
                border_mode=0, p=0.4
            ),
            A.ElasticTransform(alpha=15, sigma=4, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
        ])
    return A.Compose([])


def get_augmentation(modality, training=True):
    """Factory function to get augmentation pipeline by modality name."""
    pipelines = {
        'signature': get_signature_augmentation,
        'face': get_face_augmentation,
        'fingerprint': get_fingerprint_augmentation,
    }
    if modality not in pipelines:
        raise ValueError(f"Unknown modality: {modality}. "
                         f"Choose from: {list(pipelines.keys())}")
    return pipelines[modality](training)
