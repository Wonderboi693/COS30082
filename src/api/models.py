"""Registry of face-embedding models the app can use.

Thresholds are cosine thresholds at the EER point on verification_pairs_val.txt
(8,805 pairs), measured on 2026-11-24. Validation data only.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class EmbeddingModel:
    key: str
    label: str
    owner: str
    checkpoint: str              # file name inside the checkpoint directory
    loader: str                  # "group" (FaceEmbeddingExtractor) or "arcface" (ArcFaceExtractor)
    threshold: Optional[float]   # None = not calibrated yet
    auc: Optional[float]


# Order = preference when no model is chosen (highest validation AUC first).
MODELS = [
    EmbeddingModel("triplet", "Triplet (metric learning)", "group", "best_metric.pth", "group", 0.3333, 0.9652),
    EmbeddingModel("softmax", "Softmax (classification)", "Member A", "best_softmax.pt", "arcface", 0.2007, 0.8994),
    EmbeddingModel("arcface", "ArcFace (classification)", "Member A", "best_arcface.pt", "arcface", 0.2391, 0.8729),
    EmbeddingModel("classification", "Classification (group)", "group", "best.pt", "group", None, None),
]
MODELS_BY_KEY = {m.key: m for m in MODELS}
UNCALIBRATED_THRESHOLD = 0.30


def checkpoint_path(model: EmbeddingModel, checkpoint_dir: Path) -> Path:
    return Path(checkpoint_dir) / model.checkpoint


def available(checkpoint_dir: Path):
    return [m for m in MODELS if checkpoint_path(m, checkpoint_dir).is_file()]


def load_extractor(model: EmbeddingModel, checkpoint_dir: Path, device="auto"):
    """Both extractors expose get_embedding(image) and extract(images) -> unit-norm vectors."""
    path = checkpoint_path(model, checkpoint_dir)
    if model.loader == "arcface":
        from src.verification.arcface.extractor import ArcFaceExtractor
        return ArcFaceExtractor(path, device=device)
    from src.verification.extractor import FaceEmbeddingExtractor
    return FaceEmbeddingExtractor(str(path), device=device, normalize=True)
