"""
Face dataset loaders with RGB + grayscale support.
"""

import os
import cv2
import numpy as np
from PIL import Image
from data.base_loader import BiometricDataset
from data.preprocessing import preprocess_face, IMAGE_SIZES


class ATTFaceDataset(BiometricDataset):
    IMG_SIZE = IMAGE_SIZES["face"]

    def __init__(self, root_dir, color_mode='grayscale', **kwargs):
        self.color_mode = color_mode
        super().__init__(root_dir, **kwargs)

    def _load_data(self):
        for subj_dir in sorted(os.listdir(self.root_dir)):
            subj_path = os.path.join(self.root_dir, subj_dir)
            if not os.path.isdir(subj_path) or not subj_dir.startswith('s'):
                continue
            try:
                subj_id = int(subj_dir[1:])
            except ValueError:
                subj_id = subj_dir
            self.data[subj_id] = {'genuine': [], 'forgery': []}
            for fname in sorted(os.listdir(subj_path)):
                if fname.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp')):
                    self.data[subj_id]['genuine'].append(os.path.join(subj_path, fname))
        print(f"[AT&T] {len(self.data)} subjects, "
              f"{sum(len(v['genuine']) for v in self.data.values())} images "
              f"(color_mode={self.color_mode})")

    def _preprocess(self, image):
        img = np.array(image, dtype=np.uint8)
        if self.color_mode == 'rgb' and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = preprocess_face(img)
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)


class LFWDataset(BiometricDataset):
    IMG_SIZE = IMAGE_SIZES["face"]

    def __init__(self, root_dir, min_images=5, color_mode='rgb', **kwargs):
        self.min_images = min_images
        self.color_mode = color_mode
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
        print(f"[LFW] {len(self.data)} identities (>= {self.min_images} images), "
              f"{sum(len(v['genuine']) for v in self.data.values())} total "
              f"(color_mode={self.color_mode})")

    def _preprocess(self, image):
        if self.color_mode == 'rgb':
            image = image.convert('RGB')
            img = np.array(image, dtype=np.uint8)
        else:
            image = image.convert('L')
            img = np.array(image, dtype=np.uint8)
        img = preprocess_face(img)
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)
