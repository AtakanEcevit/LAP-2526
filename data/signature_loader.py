"""
Signature dataset loaders for CEDAR and BHSig260.
"""

import os
import cv2
import numpy as np
from PIL import Image

from data.base_loader import BiometricDataset
from data.preprocessing import preprocess_signature, IMAGE_SIZES


class CEDARDataset(BiometricDataset):
    """
    CEDAR Signature Dataset.
    
    Expected directory structure:
        data/raw/signatures/CEDAR/
            full_org/
            full_forg/
    """

    IMG_SIZE = IMAGE_SIZES["signature"]

    def _load_data(self):

        genuine_dir = os.path.join(self.root_dir, "full_org")
        forgery_dir = os.path.join(self.root_dir, "full_forg")

        if os.path.exists(genuine_dir):
            for fname in sorted(os.listdir(genuine_dir)):

                if not fname.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".bmp")
                ):
                    continue

                parts = fname.replace(".", "_").split("_")

                try:
                    writer_id = int(parts[1])
                except (IndexError, ValueError):
                    continue

                if writer_id not in self.data:
                    self.data[writer_id] = {"genuine": [], "forgery": []}

                self.data[writer_id]["genuine"].append(
                    os.path.join(genuine_dir, fname)
                )

        if os.path.exists(forgery_dir):
            for fname in sorted(os.listdir(forgery_dir)):

                if not fname.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".bmp")
                ):
                    continue

                parts = fname.replace(".", "_").split("_")

                try:
                    writer_id = int(parts[1])
                except (IndexError, ValueError):
                    continue

                if writer_id not in self.data:
                    self.data[writer_id] = {"genuine": [], "forgery": []}

                self.data[writer_id]["forgery"].append(
                    os.path.join(forgery_dir, fname)
                )

        print(
            f"[CEDAR] Loaded {len(self.data)} writers, "
            f"{sum(len(v['genuine']) for v in self.data.values())} genuine, "
            f"{sum(len(v['forgery']) for v in self.data.values())} forgery"
        )

    def _preprocess(self, image):

        img = np.array(image)

        # güvenli grayscale dönüşüm
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # noise azaltma
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # CLAHE contrast enhancement
        img = preprocess_signature(img)

        # stroke sharpening
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        img = cv2.filter2D(img, -1, kernel)

        # resize (model input size)
        img = cv2.resize(
            img,
            (self.IMG_SIZE[1], self.IMG_SIZE[0]),
            interpolation=cv2.INTER_AREA
        )

        return Image.fromarray(img)


class BHSig260Dataset(BiometricDataset):
    """
    BHSig260 Dataset (Bengali or Hindi).
    """

    IMG_SIZE = IMAGE_SIZES["signature"]

    def __init__(self, root_dir, script="Bengali", **kwargs):
        self.script = script
        super().__init__(root_dir, **kwargs)

    def _load_data(self):

        script_dir = os.path.join(self.root_dir, self.script)

        if not os.path.exists(script_dir):
            script_dir = self.root_dir

        for writer_dir in sorted(os.listdir(script_dir)):

            writer_path = os.path.join(script_dir, writer_dir)

            if not os.path.isdir(writer_path):
                continue

            try:
                writer_id = int(writer_dir)
            except ValueError:
                writer_id = writer_dir

            self.data[writer_id] = {"genuine": [], "forgery": []}

            for fname in sorted(os.listdir(writer_path)):

                if not fname.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".bmp")
                ):
                    continue

                fpath = os.path.join(writer_path, fname)

                upper = fname.upper()

                if "-G-" in upper or "_G_" in upper:
                    self.data[writer_id]["genuine"].append(fpath)

                elif "-F-" in upper or "_F_" in upper:
                    self.data[writer_id]["forgery"].append(fpath)

        self.data = {
            k: v for k, v in self.data.items()
            if len(v["genuine"]) > 0
        }

        print(
            f"[BHSig260-{self.script}] Loaded {len(self.data)} writers, "
            f"{sum(len(v['genuine']) for v in self.data.values())} genuine, "
            f"{sum(len(v['forgery']) for v in self.data.values())} forgery"
        )

    def _preprocess(self, image):

        img = np.array(image)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, (3, 3), 0)

        img = preprocess_signature(img)

        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        img = cv2.filter2D(img, -1, kernel)

        img = cv2.resize(
            img,
            (self.IMG_SIZE[1], self.IMG_SIZE[0]),
            interpolation=cv2.INTER_AREA
        )

        return Image.fromarray(img)
