"""
Signature dataset loaders for CEDAR and BHSig260.
"""

import os
import cv2
import numpy as np
from PIL import Image
from data.base_loader import BiometricDataset


class CEDARDataset(BiometricDataset):
    """
    CEDAR Signature Dataset.
    
    Expected directory structure:
        data/raw/signatures/CEDAR/
            full_org/       # Genuine signatures: original_1_1.png, original_1_2.png, ...
            full_forg/      # Forgeries: forgeries_1_1.png, forgeries_1_2.png, ...
    
    55 writers, 24 genuine + 24 forgery signatures each = 2,640 total images.
    """

    IMG_SIZE = (155, 220)  # H x W

    def _load_data(self):
        genuine_dir = os.path.join(self.root_dir, 'full_org')
        forgery_dir = os.path.join(self.root_dir, 'full_forg')

        # Parse genuine signatures
        if os.path.exists(genuine_dir):
            for fname in sorted(os.listdir(genuine_dir)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    continue
                # Parse: original_{writer}_{sample}.png
                parts = fname.replace('.', '_').split('_')
                try:
                    writer_id = int(parts[1])
                except (IndexError, ValueError):
                    continue

                if writer_id not in self.data:
                    self.data[writer_id] = {'genuine': [], 'forgery': []}
                self.data[writer_id]['genuine'].append(
                    os.path.join(genuine_dir, fname)
                )

        # Parse forged signatures
        if os.path.exists(forgery_dir):
            for fname in sorted(os.listdir(forgery_dir)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    continue
                parts = fname.replace('.', '_').split('_')
                try:
                    writer_id = int(parts[1])
                except (IndexError, ValueError):
                    continue

                if writer_id not in self.data:
                    self.data[writer_id] = {'genuine': [], 'forgery': []}
                self.data[writer_id]['forgery'].append(
                    os.path.join(forgery_dir, fname)
                )

        print(f"[CEDAR] Loaded {len(self.data)} writers, "
              f"{sum(len(v['genuine']) for v in self.data.values())} genuine, "
              f"{sum(len(v['forgery']) for v in self.data.values())} forgery")

    def _preprocess(self, image):
        """Grayscale → binary (Otsu) → resize to 155×220."""
        img = np.array(image, dtype=np.uint8)

        # Otsu binarization
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if background is black (we want white background, dark ink)
        if np.mean(img) < 127:
            img = 255 - img

        # Resize
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)

        return Image.fromarray(img)


class BHSig260Dataset(BiometricDataset):
    """
    BHSig260 Dataset (Bengali or Hindi).
    
    Expected directory structure:
        data/raw/signatures/BHSig260/Bengali/ (or Hindi/)
            001/        # Writer directories
                B-S-01-F-01.tif   # Forgery
                B-S-01-G-01.tif   # Genuine
                ...
    
    Bengali: 100 writers, Hindi: 160 writers.
    Each writer: 24 genuine + 30 forgery signatures.
    """

    IMG_SIZE = (155, 220)

    def __init__(self, root_dir, script="Bengali", **kwargs):
        self.script = script
        super().__init__(root_dir, **kwargs)

    def _load_data(self):
        script_dir = os.path.join(self.root_dir, self.script)
        if not os.path.exists(script_dir):
            script_dir = self.root_dir  # Fallback: maybe already in the right dir

        for writer_dir in sorted(os.listdir(script_dir)):
            writer_path = os.path.join(script_dir, writer_dir)
            if not os.path.isdir(writer_path):
                continue

            try:
                writer_id = int(writer_dir)
            except ValueError:
                writer_id = writer_dir

            self.data[writer_id] = {'genuine': [], 'forgery': []}

            for fname in sorted(os.listdir(writer_path)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    continue
                fpath = os.path.join(writer_path, fname)
                upper = fname.upper()
                if '-G-' in upper or '_G_' in upper:
                    self.data[writer_id]['genuine'].append(fpath)
                elif '-F-' in upper or '_F_' in upper:
                    self.data[writer_id]['forgery'].append(fpath)

        # Remove empty writers
        self.data = {k: v for k, v in self.data.items()
                     if len(v['genuine']) > 0}

        print(f"[BHSig260-{self.script}] Loaded {len(self.data)} writers, "
              f"{sum(len(v['genuine']) for v in self.data.values())} genuine, "
              f"{sum(len(v['forgery']) for v in self.data.values())} forgery")

    def _preprocess(self, image):
        """Same as CEDAR: Grayscale → Otsu → resize."""
        img = np.array(image, dtype=np.uint8)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(img) < 127:
            img = 255 - img
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)
