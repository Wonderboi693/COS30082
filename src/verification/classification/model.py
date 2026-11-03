import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from src.config import Config

ARCHITECTURES = ("resnet18", "resnet50", "mobilenet_v3_small")


class EmbeddingNetwork(nn.Module):
    """Classifier-free network; returns raw penultimate features by default."""
    def __init__(self, features, embedding, normalize=False):
        super().__init__()
        self.features = features
        self.embedding = embedding
        self.normalize = normalize

    def forward(self, images):
        embeddings = self.embedding(self.features(images).flatten(1))
        return F.normalize(embeddings, p=2, dim=1) if self.normalize else embeddings


class ClassificationFaceNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=Config.EMBEDDING_DIM, architecture=Config.ARCHITECTURE, pretrained=True):
        super().__init__()
        if num_classes < 2 or embedding_dim < 2:
            raise ValueError("num_classes and embedding_dim must be >= 2")
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        weights = None
        if architecture.startswith("resnet"):
            if pretrained:
                weights = (models.ResNet50_Weights.DEFAULT if architecture == "resnet50"
                           else models.ResNet18_Weights.DEFAULT)
            backbone = getattr(models, architecture)(weights=weights)
            feature_dim = backbone.fc.in_features
            self.features = nn.Sequential(*list(backbone.children())[:-1])
        else:
            if pretrained:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
            backbone = models.mobilenet_v3_small(weights=weights)
            feature_dim = backbone.classifier[0].in_features
            self.features = nn.Sequential(backbone.features, backbone.avgpool)
        self.embedding = nn.Sequential(nn.Linear(feature_dim, embedding_dim), nn.ReLU())
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.config = {"num_classes": num_classes, "embedding_dim": embedding_dim,
                       "architecture": architecture}

    def forward_embeddings(self, images, normalize=False):
        embeddings = self.embedding(self.features(images).flatten(1))
        return F.normalize(embeddings, p=2, dim=1) if normalize else embeddings

    def extract_embedding(self, images):
        """Return normalized penultimate features for face comparison."""
        return self.forward_embeddings(images, normalize=True)

    def forward(self, images):
        # Classify raw features; normalize the returned verification embedding.
        embedding = self.forward_embeddings(images)
        logits = self.classifier(self.dropout(embedding))
        return logits, F.normalize(embedding, p=2, dim=1)

    def without_classifier(self, normalize=False):
        return EmbeddingNetwork(self.features, self.embedding, normalize)


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install CUDA-enabled PyTorch on the NVIDIA PC.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is unavailable; use --device cpu")
    return device
