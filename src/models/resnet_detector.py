"""ResNet-50 architecture compatible with train_advanced.py checkpoints."""
from torch import nn
from torchvision import models


def build_model(pretrained: bool = False, dropout: float = 0.4):
    """Build the binary ResNet-50 used by train_advanced.py.

    Class indices follow ImageFolder / UniversalDeepfakeDataset: fake=0, real=1.
    Set pretrained=False for inference when loading a local trained checkpoint.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 2))
    return model
