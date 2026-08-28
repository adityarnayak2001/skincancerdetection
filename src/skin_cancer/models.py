from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(architecture: str, num_classes: int, pretrained: bool = True, dropout: float = 0.25) -> nn.Module:
    """Create a transfer-learning classifier with a correctly sized output head."""
    architecture = architecture.lower()
    weights = "DEFAULT" if pretrained else None
    if architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
    elif architecture == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError("architecture must be 'mobilenet_v2' or 'resnet18'")
    return model


def gradcam_layer(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "mobilenet_v2":
        return model.features[-1]
    if architecture == "resnet18":
        return model.layer4[-1]
    raise ValueError(f"No Grad-CAM layer configured for {architecture}")
