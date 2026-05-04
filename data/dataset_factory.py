"""
Dataset factory — single source of truth for creating datasets.

Consolidates the duplicated get_dataset() functions that were in
train.py, evaluate.py, calibrate_thresholds.py, and colab_train.py.
"""

from data.signature_loader import CEDARDataset, BHSig260Dataset
from data.face_loader import ATTFaceDataset, LFWDataset, CasiaWebFaceDataset
from data.fingerprint_loader import SOCOFingDataset
from data.augmentations import get_augmentation


def get_dataset(config, training=True):
    """
    Create the appropriate dataset from a config dict.

    Args:
        config:   Parsed YAML config dict (must contain 'dataset' section)
        training: If True, apply training augmentations; if False, eval-only transforms

    Returns:
        BiometricDataset subclass instance
    """
    modality = config['dataset']['modality']
    name = config['dataset']['name']
    root_dir = config['dataset']['root_dir']
    transform = get_augmentation(modality, training=training)
    in_channels = config.get('model', {}).get('in_channels', 1)

    if modality == 'signature':
        if name == 'cedar':
            return CEDARDataset(root_dir, transform=transform)
        elif name == 'bhsig260':
            script = config['dataset'].get('script', 'Bengali')
            return BHSig260Dataset(root_dir, script=script, transform=transform)
    elif modality == 'face':
        if name == 'att':
            return ATTFaceDataset(root_dir, transform=transform, in_channels=in_channels)
        elif name == 'lfw':
            min_imgs = config['dataset'].get('min_images', 5)
            return LFWDataset(root_dir, min_images=min_imgs, transform=transform,
                              in_channels=in_channels)
        elif name == 'casia':
            rec_file = config['dataset'].get('rec_file', 'train.rec')
            idx_file = config['dataset'].get('idx_file', 'train.idx')
            min_imgs = config['dataset'].get('min_images', 1)
            return CasiaWebFaceDataset(
                root_dir, rec_file=rec_file, idx_file=idx_file,
                min_images=min_imgs, transform=transform,
            )
    elif modality == 'fingerprint':
        if name == 'socofing':
            return SOCOFingDataset(root_dir, transform=transform)

    raise ValueError(f"Unknown dataset: {modality}/{name}")
