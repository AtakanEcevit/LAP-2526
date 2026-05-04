"""
Face dataset loaders for AT&T/ORL, LFW, and CASIA-WebFace.
"""

import os
import io
import struct
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


class CasiaWebFaceDataset(BiometricDataset):
    """
    CASIA-WebFace dataset in MXNet RecordIO format.

    Expected files in root_dir:
        train.rec  — images stored as JPEG inside a RecordIO binary
        train.idx  — index: binary (uint64 key + uint64 offset) or text (id\\toffset)
        train.lst  — optional text list: id\\tlabel\\tpath per line

    10,575 identities, ~494,414 colour images (112×112 → resized to 105×105).
    Requires no mxnet installation — reads the binary format directly.
    """

    # MXNet RecordIO constants
    _MAGIC = 0xced7230a
    _IHEADER_SIZE = 24   # flag(int32) + label(float32) + id(uint64) + id2(uint64)

    IMG_SIZE = IMAGE_SIZES["face"]

    def __init__(self, root_dir, rec_file="train.rec", idx_file="train.idx",
                 min_images=1, **kwargs):
        """
        Args:
            rec_file:   RecordIO filename inside root_dir (default train.rec)
            idx_file:   Index filename inside root_dir (default train.idx)
            min_images: Drop identities with fewer than this many images
        """
        self._rec_file = rec_file
        self._idx_file = idx_file
        self._min_images = min_images
        self._offsets = {}   # {rec_id(int): byte_offset(int)}
        super().__init__(root_dir, **kwargs)

    # ── Data loading ──────────────────────────────────────────────────────

    def _load_data(self):
        rec_path = os.path.join(self.root_dir, self._rec_file)
        idx_path = os.path.join(self.root_dir, self._idx_file)
        lst_path = os.path.join(self.root_dir,
                                os.path.splitext(self._rec_file)[0] + ".lst")

        if not os.path.exists(rec_path):
            raise FileNotFoundError(f"[CASIA-WebFace] .rec file not found: {rec_path}")

        self._offsets = self._parse_idx(idx_path)

        if os.path.exists(lst_path):
            label_map = self._parse_lst(lst_path)
        else:
            label_map = self._scan_rec_headers(rec_path)

        for rec_id, label in label_map.items():
            if self._offsets and rec_id not in self._offsets:
                continue
            if label not in self.data:
                self.data[label] = {'genuine': [], 'forgery': []}
            self.data[label]['genuine'].append(rec_id)

        if self._min_images > 1:
            self.data = {
                k: v for k, v in self.data.items()
                if len(v['genuine']) >= self._min_images
            }

        n_images = sum(len(v['genuine']) for v in self.data.values())
        print(f"[CASIA-WebFace] Loaded {len(self.data)} identities, {n_images} images")

    def _parse_idx(self, idx_path):
        """Parse .idx file (binary or text) → {rec_id: byte_offset}."""
        if not os.path.exists(idx_path):
            return {}

        with open(idx_path, 'rb') as f:
            raw = f.read()

        # Try binary format: 16-byte entries (uint64 key + uint64 offset)
        if len(raw) >= 16 and len(raw) % 16 == 0:
            try:
                offsets = {}
                for i in range(0, len(raw), 16):
                    key, off = struct.unpack_from('<QQ', raw, i)
                    offsets[int(key)] = int(off)
                if offsets:
                    return offsets
            except struct.error:
                pass

        # Fall back to text format: id\toffset\n
        offsets = {}
        try:
            for line in raw.decode('utf-8').strip().splitlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    offsets[int(parts[0])] = int(parts[1])
        except (UnicodeDecodeError, ValueError):
            pass
        return offsets

    def _parse_lst(self, lst_path):
        """Parse .lst text file → {rec_id(int): label(int)}.

        Real CASIA-WebFace format (tab-separated):
            <flag>  <abs_path>  <x>  <y>  <w>  <h>  ...

        col-0 is a detection flag (always 0) — NOT the identity label.
        Identity is encoded in the parent directory of col-1:
            /raid5data/dplearn/CASIA-WebFace/0000001/001.jpg
                                              ^^^^^^^ → label 1

        rec_id = 0-based line number, matching the .rec record order.
        """
        label_map = {}
        with open(lst_path, 'r') as f:
            for rec_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                try:
                    # Extract identity from parent directory name in the path
                    identity_dir = parts[1].split('/')[-2]
                    label_map[rec_id] = int(identity_dir)
                except (ValueError, IndexError):
                    continue
        return label_map

    def _scan_rec_headers(self, rec_path):
        """Scan .rec headers to extract (rec_id → label) when .lst is missing."""
        label_map = {}
        with open(rec_path, 'rb') as f:
            rec_id = 0
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                magic, length = struct.unpack('<II', hdr)
                if magic != self._MAGIC:
                    break
                rest = f.read(length)
                if len(rest) < self._IHEADER_SIZE:
                    break
                _, label_f = struct.unpack('<if', rest[:8])
                label_map[rec_id] = int(label_f)
                rec_id += 1
        return label_map

    # ── Image loading (overrides base class to read from RecordIO) ────────

    def _read_jpeg(self, rec_id):
        """Extract raw JPEG bytes for rec_id from the .rec file."""
        rec_path = os.path.join(self.root_dir, self._rec_file)
        offset = self._offsets[rec_id]
        with open(rec_path, 'rb') as f:
            f.seek(offset)
            magic, length = struct.unpack('<II', f.read(8))
            rest = f.read(length)
        return rest[self._IHEADER_SIZE:]

    def load_image(self, rec_id):
        """Override: decode JPEG from RecordIO instead of opening a file path."""
        if rec_id in self._image_cache:
            img = self._image_cache[rec_id]
        else:
            jpeg_bytes = self._read_jpeg(rec_id)
            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"[CASIA-WebFace] Failed to decode rec_id={rec_id}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            pil_img = self._preprocess(Image.fromarray(img_rgb))
            img = np.array(pil_img, dtype=np.float32)
            if img.max() > 1.0:
                img /= 255.0
            # (H, W, C) → (C, H, W)
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)
            elif img.ndim == 2:
                img = np.expand_dims(img, 0)

            if len(self._image_cache) >= self._max_cache_size:
                del self._image_cache[next(iter(self._image_cache))]
            self._image_cache[rec_id] = img

        if self.transform:
            img_hwc = img.transpose(1, 2, 0) if img.ndim == 3 else img
            transformed = self.transform(image=img_hwc)
            img = transformed['image']
            if img.ndim == 2:
                img = np.expand_dims(img, 0)
            elif img.ndim == 3 and img.shape[-1] in (1, 3):
                img = img.transpose(2, 0, 1)

        return img

    def _preprocess(self, image):
        """Resize colour image to IMG_SIZE."""
        img = np.array(image, dtype=np.uint8)
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)
