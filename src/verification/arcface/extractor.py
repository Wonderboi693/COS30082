"""Inference for the softmax / ArcFace embedding models, with the same interface as
src.verification.extractor.FaceEmbeddingExtractor: get_embedding(image) and extract(images).

    from src.verification.arcface.extractor import ArcFaceExtractor
    ext = ArcFaceExtractor()                                  # Config.SOFTMAX_CHECKPOINT
    ext = ArcFaceExtractor(Config.ARCFACE_CHECKPOINT)
    emb = ext.get_embedding(face_crop)                        # unit-norm 512-d, cosine = dot product
    same = float(emb_a @ emb_b) >= ext.threshold
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from src.config import Config

from .dataset import IMG_SIZE, MEAN, STD
from .model import load_embedder

# Training images are 64x64 crops upsampled to 112. Downsampling live crops to 64
# first keeps webcam faces close to the training distribution.
TRAIN_NATIVE_SIZE = 64


class ArcFaceExtractor:
    def __init__(self, checkpoint_path=None, device="auto", match_train_resolution=True):
        path = Path(checkpoint_path or Config.SOFTMAX_CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.checkpoint_path = path.resolve()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = load_embedder(self.checkpoint_path, device=self.device)
        self.embedding_dim = self.model.emb_dim
        meta = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.threshold = meta.get("threshold")   # validation EER threshold (exported checkpoints)
        steps = [T.Resize((TRAIN_NATIVE_SIZE, TRAIN_NATIVE_SIZE))] if match_train_resolution else []
        self.transform = T.Compose(steps + [T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

    @staticmethod
    def _to_pil(img, color_order="RGB"):
        if isinstance(img, (str, Path)):
            with Image.open(img) as opened:
                return opened.convert("RGB")
        if isinstance(img, np.ndarray):
            if color_order == "BGR" and img.ndim == 3 and img.shape[2] == 3:
                img = img[..., ::-1]
            return Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        raise TypeError(f"Unsupported image format: {type(img)}")

    def get_embedding(self, image_input, color_order="RGB") -> np.ndarray:
        """Returns a 1D float32 unit-norm embedding."""
        return self.extract([image_input], color_order=color_order)[0]

    @torch.inference_mode()
    def extract(self, images, batch_size=32, color_order="RGB") -> np.ndarray:
        """Returns [N, 512] float32 unit-norm embeddings."""
        images = list(images)
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        out = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(self._to_pil(i, color_order))
                                 for i in images[start:start + batch_size]]).to(self.device)
            out.append(F.normalize(self.model(batch).float(), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)
