"""
backbone.py — Backbone mimarisi

Desteklenen modeller:
  - FaceResNet50   : ResNet-50 + GeM Pooling + EmbeddingHead
  - FaceEfficientNet: EfficientNet-B3 + GeM Pooling + EmbeddingHead
  - LightCNNEncoder : Hafif CNN (GPU yoksa)

in_channels her zaman config'den okunur; hiçbir sınıfta hardcoded değer yoktur.
FaceResNet50 kuralı:
  - in_channels == 3 → conv1'e dokunma (pretrained ağırlıklar korunur)
  - in_channels == 1 → conv1'i tek kanallı yeni katmanla değiştir,
                       pretrained ağırlıklar RGB kanallarının ortalaması alınarak aktarılır
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Yardımcı katmanlar
# ---------------------------------------------------------------------------

class GeM(nn.Module):
    """Generalized Mean Pooling — p=1 → avg pooling, p→∞ → max pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), output_size=1
        ).pow(1.0 / self.p)


class EmbeddingHead(nn.Module):
    """BN → Dropout → Linear → BN → L2-norm."""

    def __init__(self, in_features: int, embedding_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(x), p=2, dim=1)


# ---------------------------------------------------------------------------
# FaceResNet50
# ---------------------------------------------------------------------------

class FaceResNet50(nn.Module):
    """
    ResNet-50 tabanlı yüz embedding modeli.

    conv1 kuralı (config'den gelen in_channels değerine göre):
      - in_channels == 3 : conv1'e dokunulmaz; ImageNet pretrained ağırlıklar aynen korunur.
      - in_channels == 1 : conv1, tek kanallı yeni bir Conv2d ile değiştirilir.
                           pretrained=True ise orijinal 3-kanallı ağırlıkların kanal
                           ortalaması alınarak yeni katmana kopyalanır.

    in_channels değeri her zaman dışarıdan verilmek zorundadır; varsayılan yoktur.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.4,
        pretrained: bool = True,
        in_channels: int = None,
    ):
        super().__init__()

        if in_channels is None:
            raise ValueError(
                "FaceResNet50: 'in_channels' zorunludur. "
                "YAML config'e 'in_channels: 1' veya 'in_channels: 3' ekleyin."
            )
        if in_channels not in (1, 3):
            raise ValueError(
                f"FaceResNet50: in_channels yalnızca 1 veya 3 olabilir, {in_channels!r} verildi."
            )

        self.in_channels = in_channels

        # ResNet-50'yi pretrained ağırlıklarla yükle
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        if self.in_channels == 1:
            # conv1'i tek kanallı katmanla değiştir
            orig_conv = backbone.conv1          # (64, 3, 7, 7)
            backbone.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            if pretrained:
                # Pretrained ağırlıkları 3 kanalın ortalamasını alarak aktar
                with torch.no_grad():
                    backbone.conv1.weight.copy_(
                        orig_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                    )
        # in_channels == 3 → conv1'e dokunulmaz

        # FC katmanını kaldır; embedding head kendimiz koyacağız
        in_features = backbone.fc.in_features   # 2048
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.pool = GeM(p=3.0)
        self.embedding = EmbeddingHead(in_features, embedding_dim, dropout)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet'in kendi avgpool'unu bypass ederek GeM kullanıyoruz
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        feat = self.pool(x).flatten(1)      # [B, 2048]
        return self.embedding(feat)          # [B, embedding_dim], L2-norm


# ---------------------------------------------------------------------------
# FaceEfficientNet
# ---------------------------------------------------------------------------

class FaceEfficientNet(nn.Module):
    """
    EfficientNet-B3 tabanlı yüz embedding modeli.

    FaceResNet50 ile aynı in_channels kuralını uygular:
      - in_channels == 3 : ilk conv'a dokunulmaz.
      - in_channels == 1 : ilk conv tek kanallı ile değiştirilir.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.4,
        pretrained: bool = True,
        in_channels: int = None,
    ):
        super().__init__()

        if in_channels is None:
            raise ValueError(
                "FaceEfficientNet: 'in_channels' zorunludur. "
                "YAML config'e 'in_channels: 1' veya 'in_channels: 3' ekleyin."
            )
        if in_channels not in (1, 3):
            raise ValueError(
                f"FaceEfficientNet: in_channels yalnızca 1 veya 3 olabilir, {in_channels!r} verildi."
            )

        self.in_channels = in_channels

        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        if self.in_channels == 1:
            orig_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels=1,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    backbone.features[0][0].weight.copy_(
                        orig_conv.weight.mean(dim=1, keepdim=True)  # (C_out, 1, K, K)
                    )
        # in_channels == 3 → ilk conv'a dokunulmaz

        self.backbone = backbone.features
        self.pool = GeM(p=3.0)
        self.embedding = EmbeddingHead(1536, embedding_dim, dropout)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # [B, 1536, H, W]
        feat = self.pool(feat).flatten(1) # [B, 1536]
        return self.embedding(feat)       # [B, embedding_dim], L2-norm


# ---------------------------------------------------------------------------
# LightCNNEncoder
# ---------------------------------------------------------------------------

class LightCNNEncoder(nn.Module):
    """Hafif CNN — GPU yoksa veya hızlı deney için."""

    def __init__(
        self,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        in_channels: int = None,
    ):
        super().__init__()

        if in_channels is None:
            raise ValueError(
                "LightCNNEncoder: 'in_channels' zorunludur. "
                "YAML config'e 'in_channels: 1' veya 'in_channels: 3' ekleyin."
            )

        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),          nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),         nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = EmbeddingHead(256, embedding_dim, dropout)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.features(x).flatten(1))


# ---------------------------------------------------------------------------
# build_backbone — config'den backbone oluştur
# ---------------------------------------------------------------------------

def build_backbone(config: dict) -> nn.Module:
    """
    Config dict'e göre backbone döner.

    Zorunlu anahtar: 'in_channels' (int: 1 veya 3)
    İsteğe bağlı  : 'backbone' (str), 'embedding_dim' (int),
                    'dropout' (float), 'pretrained' (bool)

    Örnek YAML:
        model:
          backbone: resnet50
          embedding_dim: 512
          in_channels: 3
    """
    if "in_channels" not in config:
        raise ValueError(
            "build_backbone(): config'de 'in_channels' anahtarı zorunludur. "
            "YAML model bölümüne 'in_channels: 1' veya 'in_channels: 3' ekleyin."
        )

    name        = config.get("backbone", "resnet50").lower()
    emb_dim     = config.get("embedding_dim", 512)
    dropout     = config.get("dropout", 0.4)
    pretrained  = config.get("pretrained", True)
    in_channels = config["in_channels"]

    if name in ("resnet50", "resnet"):
        return FaceResNet50(emb_dim, dropout, pretrained, in_channels)
    elif name in ("efficientnet", "efficientnet_b3"):
        return FaceEfficientNet(emb_dim, dropout, pretrained, in_channels)
    elif name in ("light", "lightcnn"):
        return LightCNNEncoder(min(emb_dim, 256), dropout, in_channels)
    else:
        raise ValueError(
            f"build_backbone(): bilinmeyen backbone '{name}'. "
            "Geçerli değerler: 'resnet50', 'efficientnet', 'light'"
        )
