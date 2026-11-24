"""FaceEmbedder: ResNet18 (ImageNet init, partial fine-tuning) -> 512-d face embedding,
plus the training heads: SoftmaxHead (baseline) and ArcFaceHead (additive angular margin).

load_embedder() reads both checkpoint formats:
  - exported checkpoints (checkpoints/best_softmax.pt, best_arcface.pt): "model_state" + "model_config"
  - training run checkpoints (checkpoints/arcface_runs/<run>/best.pt): "model" + "config"
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from src.config import Config

EMB_DIM = Config.ARCFACE_EMBEDDING_DIM

FREEZE_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4"]


class FaceEmbedder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, pretrained=True, freeze="none"):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Embedding neck: dropout + linear + BN gives a compact, well-scaled embedding.
        self.neck = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, emb_dim, bias=False), nn.BatchNorm1d(emb_dim))
        self.emb_dim = emb_dim
        self.freeze = freeze
        self._apply_freeze()

    def _frozen_modules(self):
        if self.freeze in (None, "none"):
            return []
        if self.freeze not in FREEZE_ORDER:
            raise ValueError(f"freeze must be 'none' or one of {FREEZE_ORDER}")
        return [getattr(self, n) for n in FREEZE_ORDER[: FREEZE_ORDER.index(self.freeze) + 1]]

    def _apply_freeze(self):
        for m in self._frozen_modules():
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for m in self._frozen_modules():  # keep frozen BN statistics fixed
            m.eval()
        return self

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return self.neck(torch.flatten(self.pool(x), 1))


def load_embedder(path, device="cpu") -> FaceEmbedder:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    if "model_state" in ckpt:   # exported format
        emb_dim = ckpt.get("model_config", {}).get("embedding_dim", EMB_DIM)
        state = ckpt["model_state"]
    else:                        # training run format
        emb_dim = ckpt.get("config", {}).get("emb_dim", EMB_DIM)
        state = ckpt["model"]
    model = FaceEmbedder(emb_dim=emb_dim, pretrained=False, freeze="none")
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def embed(model, images, normalize=False):
    """Embeddings for a batch; normalize=True returns unit-length vectors (for cosine)."""
    out = model(images)
    return F.normalize(out, dim=1) if normalize else out


# ---------------------------------------------------------------- training heads
class SoftmaxHead(nn.Module):
    """Plain linear classifier trained with cross-entropy (baseline)."""

    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, emb, labels=None):
        return self.fc(emb)


class ArcFaceHead(nn.Module):
    """Additive angular margin (Deng et al., 2019): logit_y = s * cos(theta_y + m).

    Non-target logits stay s * cos(theta_j). Without labels (inference) no margin is added.
    """

    def __init__(self, emb_dim, num_classes, s=32.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.set_margin(m)

    def set_margin(self, m):
        """Allows a margin warm-up schedule during training."""
        self.m = m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)          # beyond this, theta + m > pi
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels=None):
        # Always fp32: under fp16 autocast the clamp 1 - 1e-7 rounds to 1.0, so sqrt(1 - cos^2)
        # can hit 0 and its gradient explodes.
        with torch.autocast(emb.device.type, enabled=False):
            cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)
            self.last_cos = cos.detach()   # margin-free scores, used for the training accuracy
            if labels is None:
                return self.s * cos
            sin = torch.sqrt(1.0 - cos * cos)
            cos_margin = cos * self.cos_m - sin * self.sin_m
            # Keep the target logit monotonic when theta + m would exceed pi.
            cos_margin = torch.where(cos > self.th, cos_margin, cos - self.mm)
            one_hot = F.one_hot(labels, cos.size(1)).bool()
            return self.s * torch.where(one_hot, cos_margin, cos)


def build_head(loss, emb_dim, num_classes, s=32.0, m=0.5):
    if loss == "softmax":
        return SoftmaxHead(emb_dim, num_classes)
    if loss == "arcface":
        return ArcFaceHead(emb_dim, num_classes, s=s, m=m)
    raise ValueError(f"unknown loss {loss!r}")
