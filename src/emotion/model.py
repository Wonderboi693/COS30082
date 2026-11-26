import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import Config


class EmotionNet(nn.Module):
    """
    Multi-Class Facial Expression Classifier (9 emotion classes).
    Supports 'convnext_tiny' and 'resnet18' backbones adapted for 64x64 faces.
    """
    def __init__(
        self,
        num_classes: int = Config.NUM_EMOTIONS,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
        backbone: str = "resnet18",
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "convnext_tiny":
            base = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_backbone else None
            )
            # Adapt stem from stride=4 to stride=2 for 64x64 input (keeps 4x4 final spatial map)
            stem_norm = base.features[0][1]
            base.features[0] = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=2, stride=2),
                stem_norm,
            )
            self.features = nn.Sequential(
                *list(base.features.children()),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            in_features = 768
            self.pretrained_indices = tuple(range(1, len(base.features)))
            self.fresh_indices = (0,)
        elif backbone == "resnet18":
            base = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
            )
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                base.bn1,
                base.relu,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            in_features = 512
            self.pretrained_indices = (3, 4, 5, 6)
            self.fresh_indices = (0, 1)
        elif backbone == "resnet34":
            base = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained_backbone else None
            )
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                base.bn1,
                base.relu,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            in_features = 512
            self.pretrained_indices = (3, 4, 5, 6)
            self.fresh_indices = (0, 1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs normalized class probabilities across the emotion classes."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)
