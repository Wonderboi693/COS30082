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

    @torch.inference_mode()
    def predict_emotion(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Predicts the dominant emotion and full probability distribution.
        Measures execution latency in milliseconds.
        """
        start_time = time.perf_counter()

        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as opened:
                img = opened.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                image_input = image_input[..., ::-1]
            img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "dominant_emotion": self.classes[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "distribution": {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes, probs)},
            "latency_ms": round(latency_ms, 2),
            "latency_acceptable": latency_ms < 30.0
        }