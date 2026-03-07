"""
Face dataset loaders for AT&T/ORL and LFW.
"""

import os
import cv2
import numpy as np
from PIL import Image
from data.base_loader import BiometricDataset
from data.preprocessing import preprocess_face, IMAGE_SIZES


class ATTFaceDataset(BiometricDataset):
    """
    AT&T (ORL) Face Database.
    
    Expected directory structure:
        data/raw/faces/att_faces/
            s1/     # Subject 1
                1.pgm
                2.pgm
                ...
                10.pgm
            s2/
            ...
            s40/
    
    40 subjects, 10 images each = 400 total images.
    Images are 92x112 pixels, grayscale.
    Ideal for few-shot experiments due to small size.
    """

    IMG_SIZE = IMAGE_SIZES["face"]  # Resize to square for CNN

    def _load_data(self):
        for subj_dir in sorted(os.listdir(self.root_dir)):
            subj_path = os.path.join(self.root_dir, subj_dir)
            if not os.path.isdir(subj_path):
                continue
            if not subj_dir.startswith('s'):
                continue

            try:
                subj_id = int(subj_dir[1:])
            except ValueError:
                subj_id = subj_dir

            self.data[subj_id] = {'genuine': [], 'forgery': []}

            for fname in sorted(os.listdir(subj_path)):
                if fname.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp')):
                    self.data[subj_id]['genuine'].append(
                        os.path.join(subj_path, fname)
                    )

        print(f"[AT&T/ORL] Loaded {len(self.data)} subjects, "
              f"{sum(len(v['genuine']) for v in self.data.values())} images")

    def _preprocess(self, image):
        """Grayscale → histogram equalization → resize to 105×105."""
        img = np.array(image, dtype=np.uint8)
        img = preprocess_face(img)

        # Resize to square
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)

        return Image.fromarray(img)


class LFWDataset(BiometricDataset):
    """
    Labeled Faces in the Wild (LFW) Dataset.
    
    Expected directory structure:
        data/raw/faces/lfw/
            Aaron_Eckhart/
                Aaron_Eckhart_0001.jpg
            ...
            (5,749 identity folders)
    
    For few-shot: we filter to identities with >= k_shot+2 images.
    """

    IMG_SIZE = IMAGE_SIZES["face"]

    def __init__(self, root_dir, min_images=5, **kwargs):
        """
        Args:
            min_images: Minimum images per identity to include (default 5).
                        Filters out identities with too few samples for few-shot.
        """
        self.min_images = min_images
        super().__init__(root_dir, **kwargs)

    def _load_data(self):
        for subj_dir in sorted(os.listdir(self.root_dir)):
            subj_path = os.path.join(self.root_dir, subj_dir)
            if not os.path.isdir(subj_path):
                continue

            images = []
            for fname in sorted(os.listdir(subj_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(subj_path, fname))

            if len(images) >= self.min_images:
                self.data[subj_dir] = {'genuine': images, 'forgery': []}

        print(f"[LFW] Loaded {len(self.data)} identities "
              f"(filtered >= {self.min_images} images), "
              f"{sum(len(v['genuine']) for v in self.data.values())} total images")

    def _preprocess(self, image):
        """Grayscale → histogram equalization → resize to 105×105."""
        img = np.array(image, dtype=np.uint8)
        img = preprocess_face(img)
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)
