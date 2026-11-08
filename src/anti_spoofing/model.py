import torch
import torch.nn as nn
import torchvision.models as models

from src.config import Config


class AntiSpoofNet(nn.Module):
    """
    Binary Liveness Classifier:
    Outputs a single logit: Sigmoid(logit) = P(Sample is Live)
    """
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained_backbone else None)

        # 64x64 resolution adapted stem
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            base.bn1,
            base.relu,
            base.layer1,  # 64 channels
            base.layer2,  # 128 channels
            base.layer3,  # 256 channels
            base.layer4,  # 512 channels
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1)  # Single raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes explicit live probability in [0, 1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
