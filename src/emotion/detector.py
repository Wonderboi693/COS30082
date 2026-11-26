import time
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.config import Config
from src.emotion.model import EmotionNet


class EmotionDetector:
    """
    Inference module for 7-class facial emotion recognition.
    Meets the real-time inference latency requirement (< 30ms).
    """
    def __init__(self, checkpoint_path: str = None, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        ckpt_path = Path(checkpoint_path or Config.EMOTION_CHECKPOINT).resolve()

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Emotion checkpoint not found at: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.classes = ckpt.get("classes", Config.EMOTION_CLASSES)
        self.model = EmotionNet(num_classes=len(self.classes), pretrained_backbone=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _to_pil(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as opened:
                return opened.convert("RGB")
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                image_input = image_input[..., ::-1]
            return Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _probs_from_pil(self, img: Image.Image) -> np.ndarray:
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def predict_emotion(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Predicts the dominant emotion and full probability distribution.
        Measures execution latency in milliseconds.
        """
        start_time = time.perf_counter()
        img = self._to_pil(image_input)
        probs = self._probs_from_pil(img)

        pred_idx = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "dominant_emotion": self.classes[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "distribution": {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes, probs)},
            "latency_ms": round(latency_ms, 2),
            "latency_acceptable": latency_ms < 30.0
        }

    @torch.inference_mode()
    def predict_emotion_with_tta(self, image_input: Union[str, Path, np.ndarray, Image.Image],
                                 n_tta: int = 5) -> Dict[str, Any]:
        """
        Test-time augmentation for more stable emotion predictions.

        Averages softmax probabilities over:
          - original crop
          - horizontally flipped crop
          - small rotations (±5°, ±10°)

        Returns the same dict as predict_emotion but with smoothed / more robust probabilities.
        """
        start_time = time.perf_counter()
        img = self._to_pil(image_input)

        variants = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
        for deg in (-10, -5, 5, 10):
            variants.append(img.rotate(deg, resample=Image.BICUBIC, fillcolor=(0, 0, 0)))

        probs_sum = np.zeros(len(self.classes), dtype=np.float64)
        for v in variants[: max(1, min(n_tta, len(variants)))]:
            probs_sum += self._probs_from_pil(v)
        probs = probs_sum / len(variants[: max(1, min(n_tta, len(variants)))])

        pred_idx = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "dominant_emotion": self.classes[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "distribution": {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes, probs)},
            "latency_ms": round(latency_ms, 2),
            "latency_acceptable": latency_ms < 30.0,
            "tta": True
        }