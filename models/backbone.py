"""
backbone.py  —  Geliştirilmiş backbone mimarisi
Değişiklikler:
  - ResNet-18 → EfficientNet-B3 (veya ResNet-50 seçeneği)
  - 128-d → 512-d embedding
  - BatchNorm + Dropout pipeline
  - GeM (Generalized Mean Pooling) — ortalama yerine daha güçlü agregasyon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    Global average pooling'den daha iyi: p=1 → avg, p→∞ → max.
    Yüz tanımada özellikle yararlı.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1
        ).pow(1.0 / self.p)


class EmbeddingHead(nn.Module):
    """
    Backbone çıktısını sabit boyutlu embedding vektörüne dönüştürür.
    BN → Dropout → Linear → BN → L2Norm
    """
    def __init__(self, in_features, embedding_dim=512, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        emb = self.head(x)
        return F.normalize(emb, p=2, dim=1)  # L2 normalizasyon


class FaceEfficientNet(nn.Module):
    """
    EfficientNet-B3 tabanlı yüz embedding modeli.
    in_channels=1 (grayscale) veya 3 (RGB) destekler.
    """
    def __init__(self, embedding_dim=512, dropout=0.4, pretrained=True, in_channels=1):
        super().__init__()

        # EfficientNet-B3 yükle
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        orig_conv = backbone.features[0][0]
        if in_channels != 3:
            # 3 kanaldan in_channels kanala dönüştür
            backbone.features[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False
            )
            if pretrained:
                with torch.no_grad():
                    backbone.features[0][0].weight.copy_(
                        orig_conv.weight.mean(dim=1, keepdim=True)
                    )

        # Classifier'ı kaldır, feature extractor tut
        self.backbone = backbone.features
        self.pool = GeM(p=3.0)
        in_features = 1536  # EfficientNet-B3 çıktı kanalı

        self.embedding = EmbeddingHead(in_features, embedding_dim, dropout)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        feat = self.backbone(x)          # [B, 1536, H, W]
        feat = self.pool(feat)            # [B, 1536, 1, 1]
        feat = feat.flatten(1)            # [B, 1536]
        emb = self.embedding(feat)        # [B, 512] — L2 normalize
        return emb


class FaceResNet50(nn.Module):
    """
    ResNet-50 tabanlı alternatif (EfficientNet yoksa).
    in_channels=1 (grayscale) veya 3 (RGB) destekler.
    """
    def __init__(self, embedding_dim=512, dropout=0.4, pretrained=True, in_channels=1):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        orig_conv = backbone.conv1
        if in_channels != 3:
            # 3 kanaldan in_channels kanala dönüştür
            backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False
            )
            if pretrained:
                with torch.no_grad():
                    backbone.conv1.weight.copy_(
                        orig_conv.weight.mean(dim=1, keepdim=True)
                    )

        # FC katmanını kaldır
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.pool = GeM(p=3.0)
        self.embedding = EmbeddingHead(in_features, embedding_dim, dropout)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # ResNet50 kendi avgpool'unu yapıyor, GeM için override et
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        feat = self.pool(x).flatten(1)
        emb = self.embedding(feat)
        return emb


class LightCNNEncoder(nn.Module):
    """
    Hafif alternatif — GPU yoksa veya hızlı deney için.
    Mevcut projeden korunuyor.
    """
    def __init__(self, embedding_dim=256, dropout=0.3, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = EmbeddingHead(256, embedding_dim, dropout)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.embedding(feat)


def build_backbone(config):
    """
    Config'e göre uygun backbone döner.
    config örnekleri:
      backbone: efficientnet   embedding_dim: 512
      backbone: resnet50       embedding_dim: 512
      backbone: light          embedding_dim: 256
    """
    name = config.get('backbone', 'efficientnet').lower()
    emb_dim = config.get('embedding_dim', 512)
    dropout = config.get('dropout', 0.4)
    pretrained = config.get('pretrained', True)
    in_channels = config.get('in_channels', 1)

    if name in ('efficientnet', 'efficientnet_b3'):
        return FaceEfficientNet(emb_dim, dropout, pretrained, in_channels)
    elif name in ('resnet50', 'resnet'):
        return FaceResNet50(emb_dim, dropout, pretrained, in_channels)
    elif name in ('light', 'lightcnn'):
        return LightCNNEncoder(min(emb_dim, 256), dropout, in_channels)
    else:
        raise ValueError(f"Bilinmeyen backbone: {name}")
