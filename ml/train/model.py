from __future__ import annotations

import torch
import torch.nn as nn
import timm


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 with accessible penultimate embeddings."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.num_features = self.backbone.num_features
        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.classifier(feats)


def build_model(num_classes: int = 4) -> EfficientNetEncoder:
    return EfficientNetEncoder(num_classes=num_classes)
