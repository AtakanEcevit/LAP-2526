"""
Per-modality augmentation pipelines using Albumentations.

Optimized for biometric verification tasks.

Includes improved signature augmentations to simulate
natural handwriting variations and scanning artifacts.
"""

import albumentations as A


# ---------------------------------------------------
# SIGNATURE AUGMENTATION
# ---------------------------------------------------

def get_signature_augmentation(training=True):
    """
    Signature augmentation pipeline.

    Simulates natural handwriting variation:
    - slight rotation
    - pen pressure variation
    - scanner noise
    - elastic distortion
    """

    if training:
        return A.Compose([

            # small geometric variation
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-5, 5),
                border_mode=0,
                p=0.5
            ),

            # handwriting deformation
            A.ElasticTransform(
                alpha=20,
                sigma=5,
                p=0.3
            ),

            # scanner blur
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2
            ),

            # scanner noise
            A.GaussNoise(
                var_limit=(5.0, 20.0),
                p=0.3
            ),

            # simulate broken ink strokes
            A.CoarseDropout(
                max_holes=3,
                max_height=5,
                max_width=5,
                p=0.2
            ),

            # lighting / scan contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
        ])

    # validation / test → no augmentation
    return A.Compose([])


# ---------------------------------------------------
# FACE AUGMENTATION
# ---------------------------------------------------

def get_face_augmentation(training=True):
    """
    Face augmentation pipeline.

    Simulates:
    - lighting changes
    - small pose shifts
    - camera noise
    """

    if training:
        return A.Compose([

            A.HorizontalFlip(p=0.5),

            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                border_mode=0,
                p=0.5
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2
            ),
        ])

    return A.Compose([])


# ---------------------------------------------------
# FINGERPRINT AUGMENTATION
# ---------------------------------------------------

def get_fingerprint_augmentation(training=True):
    """
    Fingerprint augmentation pipeline.

    Conservative augmentations because ridge structures
    are fragile and should not be heavily distorted.
    """

    if training:
        return A.Compose([

            A.Affine(
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                scale=(0.95, 1.05),
                rotate=(-8, 8),
                border_mode=0,
                p=0.4
            ),

            A.ElasticTransform(
                alpha=15,
                sigma=4,
                p=0.2
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
        ])

    return A.Compose([])


# ---------------------------------------------------
# FACTORY FUNCTION
# ---------------------------------------------------

def get_augmentation(modality, training=True):
    """
    Factory function to return augmentation pipeline
    based on biometric modality.
    """

    pipelines = {
        "signature": get_signature_augmentation,
        "face": get_face_augmentation,
        "fingerprint": get_fingerprint_augmentation,
    }

    if modality not in pipelines:
        raise ValueError(
            f"Unknown modality: {modality}. "
            f"Choose from: {list(pipelines.keys())}"
        )

    return pipelines[modality](training)
