from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import Config


class MetricFaceNet(nn.Module):
    """
    Siamese embedding network: projects face crops directly to unit L2 hypersphere.
    """
    def __init__(self, embedding_dim: int = Config.EMBEDDING_DIM):
        super().__init__()
        base = models.resnet18(weights=None)

        # 64x64 resolution adapted stem
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

        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        return F.normalize(emb, p=2, dim=1) if normalize else emb

    def load_pretrained_backbone(self, checkpoint_path: str):
        """
        Loads backbone and embedding bottleneck weights from US 1.1 supervised checkpoint,
        ignoring the 4000-class classification head.
        """
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))

        # Filter out classifier weights
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("classifier")
        }

        missing, unexpected = self.load_state_dict(filtered_dict, strict=False)
        print(f"[*] Pretrained weights loaded from {path.name}")
        if missing:
            print(f"    Missing keys (expected for non-classifier model): {missing}")
        if unexpected:
            print(f"    Unexpected keys ignored: {unexpected}")