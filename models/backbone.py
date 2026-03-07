"""
Shared CNN backbone for both Siamese and Prototypical networks.
Uses a modified ResNet-18 that outputs 128-dim embeddings.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet-18 based encoder that produces 128-dimensional embeddings.
    
    Modifications from standard ResNet-18:
        1. First conv layer accepts 1-channel (grayscale) input instead of 3
        2. Final FC layer replaced with 128-dim embedding layer
        3. L2 normalization on output embeddings
    """

    def __init__(self, embedding_dim=128, pretrained=True, in_channels=1):
        """
        Args:
            embedding_dim: Dimension of the output embedding vector
            pretrained: Use ImageNet pretrained weights (adapted for 1-ch)
            in_channels: Number of input channels (1 for grayscale)
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load base ResNet-18
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Modify first conv layer for grayscale input
        if in_channels != 3:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Average the pretrained RGB weights into single channel
                with torch.no_grad():
                    resnet.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

        # Extract feature layers (everything except final FC)
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



        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, H, W)
        Returns:
            L2-normalized embedding of shape (batch, embedding_dim)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)

        # L2 normalize embeddings
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class LightCNNEncoder(nn.Module):
    """
    Lightweight CNN encoder for faster training and lower memory usage.
    Good for initial experiments and smoke tests.
    
    Architecture:
        4 conv blocks → global avg pool → 128-dim embedding
    """

    def __init__(self, embedding_dim=128, in_channels=1):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            # Block 1: 1 → 32
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = nn.Sequential(
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
