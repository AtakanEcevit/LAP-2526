"""
Hybrid FaceNet-style adapter for FaceVerify Campus.

The checkpoint used by this adapter stores an InceptionResnetV1-like backbone
under ``model_state`` with keys prefixed by ``backbone.``. Verification uses the
512-dimensional backbone embedding; classifier/logits heads are ignored.
"""

import io
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


HYBRID_FACE_SIZE = (160, 160)
HYBRID_EMBEDDING_DIM = 512


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        return self.relu(out)


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat((self.branch0(x), self.branch1(x)), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        return self.relu(out)


class Block8(nn.Module):
    def __init__(self, scale=1.0, no_relu=False):
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu
        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not no_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat((self.branch0(x), self.branch1(x)), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.no_relu:
            return out
        return self.relu(out)


class Mixed6a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)


class Mixed7a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat(
            (self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)),
            1,
        )


class HybridFaceBackbone(nn.Module):
    """InceptionResnetV1-like embedding backbone matching the hybrid checkpoint."""

    def __init__(self, embedding_dim=HYBRID_EMBEDDING_DIM, num_classes=8631):
        super().__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(*[Block35(scale=0.17) for _ in range(5)])
        self.mixed_6a = Mixed6a()
        self.repeat_2 = nn.Sequential(*[Block17(scale=0.10) for _ in range(10)])
        self.mixed_7a = Mixed7a()
        self.repeat_3 = nn.Sequential(*[Block8(scale=0.20) for _ in range(5)])
        self.block8 = Block8(no_relu=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.6)
        self.last_linear = nn.Linear(1792, embedding_dim, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_dim, eps=0.001, momentum=0.1)
        self.logits = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        x = self.last_bn(x)
        return F.normalize(x, p=2, dim=1)


class HybridFaceModel(nn.Module):
    """Thin wrapper exposing get_embedding like the existing project models."""

    def __init__(self, embedding_dim=HYBRID_EMBEDDING_DIM, num_classes=8631):
        super().__init__()
        self.backbone = HybridFaceBackbone(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

    def get_embedding(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.get_embedding(x)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        state = checkpoint.get("model_state") or checkpoint.get("model_state_dict")
        if state is None:
            raise ValueError("Hybrid checkpoint must contain 'model_state'.")

        logits_weight = state.get("backbone.logits.weight")
        num_classes = logits_weight.shape[0] if logits_weight is not None else 8631
        model = cls(num_classes=num_classes)

        backbone_state = OrderedDict()
        for key, value in state.items():
            if not key.startswith("backbone."):
                continue
            clean_key = key[len("backbone."):]
            if clean_key.startswith("logits."):
                continue
            backbone_state[clean_key] = value

        missing, unexpected = model.backbone.load_state_dict(
            backbone_state,
            strict=False,
        )
        missing = [key for key in missing if not key.startswith("logits.")]
        if missing or unexpected:
            raise ValueError(
                "Hybrid checkpoint did not match adapter. "
                f"Missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        return model


def preprocess_hybrid_face(image_input) -> torch.Tensor:
    """Load an RGB face image into the tensor format expected by the hybrid model."""
    img = _load_rgb(image_input)
    img = cv2.resize(img, HYBRID_FACE_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).float()
    tensor = (tensor - 127.5) / 128.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def _load_rgb(image_input) -> np.ndarray:
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        if not image_input.strip():
            raise ValueError("Empty image path provided.")
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            return cv2.cvtColor(image_input.astype(np.uint8), cv2.COLOR_BGR2RGB)
        raise ValueError(f"Unexpected image array shape: {image_input.shape}.")
    else:
        raise ValueError(
            f"Unsupported image input type: {type(image_input).__name__}."
        )

    return np.array(pil_img.convert("RGB"), dtype=np.uint8)
