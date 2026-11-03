import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.config import Config
from .classification.dataset import build_transform, get_preprocessing_config
from .classification.model import ClassificationFaceNet, resolve_device


class FaceEmbeddingExtractor:
    """Load best.pt, embeddings.pt, or a run folder containing either file.

    A run folder prefers the smaller embeddings.pt export. For OpenCV face crops,
    call get_embedding(face_crop, color_order="BGR"). Inputs must be face crops.
    """
    def __init__(self, checkpoint_path=None, device=Config.DEVICE, normalize=True):
        self.device = resolve_device(device)
        path = Path(checkpoint_path if checkpoint_path is not None else Config.SUPERVISED_CHECKPOINT).expanduser()
        if path.is_dir():
            path = next((path / name for name in ("embeddings.pt", "best.pt")
                         if (path / name).is_file()), path / "embeddings.pt")
        if not path.is_file():
            raise FileNotFoundError(
                f"Trained checkpoint not found: {path}. Pass checkpoint_path as a best.pt, "
                "embeddings.pt, or run folder, or set COS30082_SUPERVISED_CHECKPOINT.")
        self.checkpoint_path = path.resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not {"model_config", "image_size", "model_state"} <= checkpoint.keys():
            raise ValueError("Checkpoint is missing model/preprocessing metadata; use this training pipeline's checkpoint. "
                             "Checkpoints from the previous architecture cannot be loaded into this model.")
        if checkpoint.get("training_steps", 0) < 1:
            raise ValueError("Embedding extraction requires a locally trained/fine-tuned checkpoint")
        self.model_config = dict(checkpoint["model_config"])
        self.image_size = checkpoint["image_size"]
        self.embedding_dim = self.model_config["embedding_dim"]
        self.normalize = normalize
        self.class_to_idx = dict(checkpoint.get("class_to_idx", {}))
        # Earlier checkpoints used these same fixed settings without storing them.
        settings = checkpoint.get("preprocessing", get_preprocessing_config(self.image_size))
        if settings.get("image_size") != self.image_size:
            raise ValueError("Checkpoint image size and preprocessing metadata disagree")
        self.preprocessing = {"image_size": self.image_size,
                              "mean": list(settings["mean"]), "std": list(settings["std"])}
        classifier = ClassificationFaceNet(**self.model_config, pretrained=False)
        if checkpoint.get("kind") == "embeddings":
            self.model = classifier.without_classifier(normalize=normalize)
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            classifier.load_state_dict(checkpoint["model_state"])
            self.model = classifier.without_classifier(normalize=normalize)
        self.model.to(self.device).eval()
        self.transform = build_transform(training=False, **self.preprocessing)

    def get_embedding(self, image_input, color_order="RGB"):
        """Return one face embedding as a 1D NumPy array."""
        return self.extract([image_input], color_order=color_order)[0]

    def extract(self, images, batch_size=32, color_order="RGB"):
        """Return [N, embedding_dim] float32 NumPy values.

        Accept paths, PIL images or uint8 NumPy arrays. OpenCV callers must pass
        color_order='BGR'; ordinary RGB/PIL/path input uses the default.
        """
        if batch_size < 1 or color_order not in {"RGB", "BGR"}:
            raise ValueError("batch_size must be positive and color_order must be RGB or BGR")
        images = list(images)
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        results = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                tensors = []
                for image in images[start:start + batch_size]:
                    if isinstance(image, (str, Path)):
                        with Image.open(image) as opened:
                            image = ImageOps.exif_transpose(opened).convert("RGB")
                    elif isinstance(image, np.ndarray):
                        if image.dtype != np.uint8:
                            raise ValueError("NumPy face images must be uint8 in [0, 255]")
                        if color_order == "BGR" and image.ndim == 3 and image.shape[2] == 3:
                            image = image[..., ::-1].copy()
                        image = Image.fromarray(image)
                    if not isinstance(image, Image.Image):
                        raise TypeError("Each image must be a path, PIL image, or uint8 NumPy array")
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    tensors.append(self.transform(image))
                batch = torch.stack(tensors).to(self.device)
                results.append(self.model(batch).float().cpu().numpy())
        embeddings = np.concatenate(results, axis=0)
        if not np.isfinite(embeddings).all():
            raise ValueError("Model produced non-finite embeddings")
        return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from cropped face images.")
    parser.add_argument("--checkpoint", help="best.pt, embeddings.pt, or a run folder; defaults to the configured checkpoint")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", default="embeddings.npy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--raw", action="store_true", help="Return raw penultimate values without L2 normalization")
    args = parser.parse_args()
    try:
        extractor = FaceEmbeddingExtractor(args.checkpoint, args.device, normalize=not args.raw)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    embeddings = extractor.extract(args.images, args.batch_size)
    output = Path(args.output)
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.save(handle, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {output}")


if __name__ == "__main__":
    main()
