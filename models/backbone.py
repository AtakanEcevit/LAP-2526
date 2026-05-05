"""
Shared CNN backbone for both Siamese and Prototypical networks.
Supports ResNet-18 and ResNet-50 with CBAM attention and configurable input channels.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True, in_channels=1,
                 architecture='resnet18', use_attention=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.architecture = architecture

        if architecture == 'resnet18':
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feature_dim = 512
        elif architecture == 'resnet50':
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        if in_channels != 3:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    resnet.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.attention = CBAM(feature_dim) if use_attention else nn.Identity()

        hidden_dim = 512
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class LightCNNEncoder(nn.Module):
    def __init__(self, embedding_dim=512, in_channels=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(nn.Linear(256, embedding_dim))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
