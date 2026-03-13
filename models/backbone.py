"""
Shared CNN backbone for both Siamese and Prototypical networks.
Improved version for stronger biometric embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetEncoder(nn.Module):

    def __init__(self, embedding_dim=128, pretrained=True, in_channels=1):

        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # ---- grayscale input fix ----

        if in_channels != 3:

            original_conv = resnet.conv1

            resnet.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            if pretrained:
                with torch.no_grad():
                    resnet.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

        # ---- backbone features ----

        self.features = nn.Sequential(

            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # ---- global pooling ----

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- stronger embedding head ----

        self.embedding = nn.Sequential(

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.embedding(x)

        # ---- L2 normalize embeddings ----

        x = F.normalize(x, p=2, dim=1)

        return x


class LightCNNEncoder(nn.Module):

    def __init__(self, embedding_dim=128, in_channels=1):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = nn.Sequential(

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.embedding(x)

        x = F.normalize(x, p=2, dim=1)

        return x
