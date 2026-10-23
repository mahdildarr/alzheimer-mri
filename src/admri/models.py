"""Classifiers: ImageNet-pretrained ResNet-18 (default), ResNet-18 from scratch, and a small CNN for CPU runs."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def small_cnn(n_classes: int) -> nn.Module:
    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))

    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, n_classes))


def build_model(name: str, n_classes: int) -> nn.Module:
    if name == "smallcnn":
        return small_cnn(n_classes)
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if name == "resnet18" else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional stage, the usual Grad-CAM target."""
    return model.layer4 if hasattr(model, "layer4") else model[3]
