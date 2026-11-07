import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import Config
from .classification.dataset import build_transform, get_preprocessing_config
from .classification.model import ClassificationFaceNet, resolve_device


class FaceEmbeddingExtractor:
    """Extracts unit L2-normalized 256-d embeddings from cropped face inputs."""
    def __init__(self, checkpoint_path=None, device="auto", normalize=True):
        self.device = resolve_device(device)
        path = Path(checkpoint_path if checkpoint_path is not None else Path(Config.CHECKPOINT_DIR) / "best.pt").expanduser()

        if path.is_dir():
            path = next((path / name for name in ("embeddings.pt", "best.pt") if (path / name).is_file()), path / "embeddings.pt")

        if not path.is_file():
            # Check fallback to best_supervised.pth
            fallback = Path(Config.CHECKPOINT_DIR) / "best_supervised.pth"
            if fallback.is_file():
                path = fallback
            else:
                raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.checkpoint_path = path.resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        # Resolve model parameters and preprocessing
        self.model_config = checkpoint.get("model_config", {
            "num_classes": checkpoint.get("num_classes", 4000),
            "embedding_dim": checkpoint.get("embedding_dim", Config.EMBEDDING_DIM),
            "architecture": "resnet18"
        })
        self.image_size = checkpoint.get("image_size", Config.IMAGE_SIZE[0])
        self.embedding_dim = self.model_config.get("embedding_dim", Config.EMBEDDING_DIM)
        self.normalize = normalize

        settings = checkpoint.get("preprocessing", get_preprocessing_config(self.image_size))
        self.preprocessing = {
            "image_size": self.image_size,
            "mean": list(settings.get("mean", [0.5, 0.5, 0.5])),
            "std": list(settings.get("std", [0.5, 0.5, 0.5]))
        }

        classifier = ClassificationFaceNet(**self.model_config, pretrained=False)
        state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))

        if checkpoint.get("kind") == "embeddings":
            self.model = classifier.without_classifier(normalize=normalize)
            self.model.load_state_dict(state_dict)
        else:
            classifier.load_state_dict(state_dict, strict=False)
            self.model = classifier.without_classifier(normalize=normalize)

        self.model.to(self.device).eval()
        self.transform = build_transform(training=False, **self.preprocessing)

    def get_embedding(self, image_input, color_order="RGB") -> np.ndarray:
        """Returns a 1D NumPy float32 embedding vector."""
        return self.extract([image_input], color_order=color_order)[0]

    def extract(self, images, batch_size=32, color_order="RGB") -> np.ndarray:
        """Extracts [N, embedding_dim] float32 embeddings for a batch of images."""
        images = list(images)
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        results = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                tensors = []
                for img in images[start:start + batch_size]:
                    if isinstance(img, (str, Path)):
                        with Image.open(img) as opened:
                            img = opened.convert("RGB")
                    elif isinstance(img, np.ndarray):
                        if color_order == "BGR" and img.ndim == 3 and img.shape[2] == 3:
                            img = img[..., ::-1].copy()
                        img = Image.fromarray(img)
                    elif not isinstance(img, Image.Image):
                        raise TypeError(f"Unsupported image format: {type(img)}")

                    tensors.append(self.transform(img))

                batch = torch.stack(tensors).to(self.device)
                results.append(self.model(batch).float().cpu().numpy())

        return np.concatenate(results, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from cropped face images.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", default="embeddings.npy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    extractor = FaceEmbeddingExtractor(args.checkpoint, args.device, normalize=not args.raw)
    embeddings = extractor.extract(args.images)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as handle:
        np.save(handle, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {output}")


if __name__ == "__main__":
    main()