import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from src.config import Config

ARCHITECTURES = ("resnet18", "resnet50", "mobilenet_v3_small")


class EmbeddingNetwork(nn.Module):
    """Classifier-free network; returns normalized or raw penultimate features."""
    def __init__(self, features: nn.Module, embedding: nn.Module, normalize: bool = True):
        super().__init__()
        self.features = features
        self.embedding = embedding
        self.normalize = normalize

    def forward(self, images):
        x = self.features(images)
        x = torch.flatten(x, 1)
        embeddings = self.embedding(x)
        return F.normalize(embeddings, p=2, dim=1) if self.normalize else embeddings


class ClassificationFaceNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=Config.EMBEDDING_DIM, architecture=getattr(Config, "ARCHITECTURE", "resnet18"), pretrained=False):
        super().__init__()
        if num_classes < 2 or embedding_dim < 2:
            raise ValueError("num_classes and embedding_dim must be >= 2")

        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        if architecture == "resnet18":
            base = models.resnet18(weights=None)
            # Custom stem preserving 64x64 spatial resolution
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                base.bn1,
                base.relu,
                base.layer1,  # 64 channels, 64x64
                base.layer2,  # 128 channels, 32x32
                base.layer3,  # 256 channels, 16x16
                base.layer4,  # 512 channels, 8x8
                nn.AdaptiveAvgPool2d((1, 1))
            )
            feature_dim = 512
        elif architecture == "resnet50":
            base = models.resnet50(weights=None)
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                base.bn1,
                base.relu,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            feature_dim = 2048
        elif architecture == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=None)
            self.features = nn.Sequential(base.features, base.avgpool)
            feature_dim = base.classifier[0].in_features
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Linear bottleneck with BatchNorm (NO ReLU, preserving complete hypersphere)
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.config = {
            "num_classes": num_classes,
            "embedding_dim": embedding_dim,
            "architecture": architecture
        }

    def forward_embeddings(self, images, normalize=False):
        x = self.features(images)
        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        return F.normalize(emb, p=2, dim=1) if normalize else emb

    def extract_embedding(self, images):
        """Returns L2-normalized penultimate features for face verification."""
        return self.forward_embeddings(images, normalize=True)

    def forward(self, images):
        x = self.features(images)
        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        logits = self.classifier(self.dropout(emb))
        return logits, F.normalize(emb, p=2, dim=1)

    def without_classifier(self, normalize=True):
        return EmbeddingNetwork(self.features, self.embedding, normalize=normalize)


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Ensure torch with CUDA support is installed.")
    return device
