"""
Fingerprint dataset loaders for SOCOFing.
"""

import os
import cv2
import numpy as np
from PIL import Image
from data.base_loader import BiometricDataset
from data.preprocessing import preprocess_fingerprint, IMAGE_SIZES


class SOCOFingDataset(BiometricDataset):
    """
    SOCOFing (Sokoto Coventry Fingerprint) Dataset.
    
    Expected directory structure:
        data/raw/fingerprints/SOCOFing/
            Real/
                1__M_Left_index_finger.BMP
                1__M_Left_little_finger.BMP
                ...
            Altered/
                Altered-Easy/
                    1__M_Left_index_finger_CR.BMP
                Altered-Medium/
                    1__M_Left_index_finger_Obl.BMP
                Altered-Hard/
                    1__M_Left_index_finger_Zcut.BMP
    
    600 subjects, 10 fingerprint images each (2 per finger × 5 fingers) = 6,000 real.
    Altered versions simulate forgery/spoofing attacks.
    """

    IMG_SIZE = IMAGE_SIZES["fingerprint"]

    def _load_data(self):
        real_dir = os.path.join(self.root_dir, 'Real')
        altered_dirs = [
            os.path.join(self.root_dir, 'Altered', 'Altered-Easy'),
            os.path.join(self.root_dir, 'Altered', 'Altered-Medium'),
            os.path.join(self.root_dir, 'Altered', 'Altered-Hard'),
        ]

        # Parse real fingerprints
        if os.path.exists(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if not fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')):
                    continue
                # Parse: {subject_id}__{gender}_{hand}_{finger}.BMP
                try:
                    subj_id = int(fname.split('__')[0])
                except (ValueError, IndexError):
                    continue

                if subj_id not in self.data:
                    self.data[subj_id] = {'genuine': [], 'forgery': []}
                self.data[subj_id]['genuine'].append(
                    os.path.join(real_dir, fname)
                )

        # Parse altered (forged) fingerprints
        for alt_dir in altered_dirs:
            if not os.path.exists(alt_dir):
                continue
            for fname in sorted(os.listdir(alt_dir)):
                if not fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')):
                    continue
                try:
                    subj_id = int(fname.split('__')[0])
                except (ValueError, IndexError):
                    continue

                if subj_id not in self.data:
                    self.data[subj_id] = {'genuine': [], 'forgery': []}
                self.data[subj_id]['forgery'].append(
                    os.path.join(alt_dir, fname)
                )

        print(f"[SOCOFing] Loaded {len(self.data)} subjects, "
              f"{sum(len(v['genuine']) for v in self.data.values())} genuine, "
              f"{sum(len(v['forgery']) for v in self.data.values())} forgery/altered")

    def _preprocess(self, image):
        """
        Grayscale → CLAHE enhancement → resize to 96×96.
        CLAHE (Contrast Limited Adaptive Histogram Equalization) is better
        than global histogram equalization for fingerprints because it
        preserves ridge/valley detail.
        """
        img = np.array(image, dtype=np.uint8)
        img = preprocess_fingerprint(img)

        # Resize
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)

        return Image.fromarray(img)
