import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import Config


class EmotionNet(nn.Module):
    """
    Multi-Class Facial Expression Classifier (7 Basic Emotions).
    Optimized for low-latency inference (< 30ms requirement).
    """
    def __init__(self, num_classes: int = Config.NUM_EMOTIONS, pretrained_backbone: bool = False):
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs normalized class probabilities across the 7 emotions."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)