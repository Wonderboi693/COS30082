import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.config import Config

class ClassificationFaceNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = Config.EMBEDDING_DIM):
        super().__init__()
        
        # Backbone adapted for small 64x64 faces
        base = models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu

        self.layer1 = base.layer1  # 64 channels,  64x64
        self.layer2 = base.layer2  # 128 channels, 32x32
        self.layer3 = base.layer3  # 256 channels, 16x16
        self.layer4 = base.layer4  # 512 channels, 8x8

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Penultimate feature embedding layer
        self.fc_embedding = nn.Linear(512, embedding_dim, bias=False)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)

        # Softmax classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts L2-normalized feature representation from the penultimate layer.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        embedding = self.bn_embedding(self.fc_embedding(x))
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x: torch.Tensor):
        embedding = self.extract_embedding(x)
        logits = self.classifier(embedding)
        return logits, embedding
