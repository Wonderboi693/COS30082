from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.config import Config
from src.anti_spoofing.model import AntiSpoofNet


class LivenessDetector:
    """
    Inline Gatekeeper module for intercepting presentation attacks.
    """
    def __init__(self, checkpoint_path: str = None, threshold: float = None, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        ckpt_path = Path(checkpoint_path or Config.ANTISPOOF_CHECKPOINT).resolve()

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Anti-spoof checkpoint not found at: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = AntiSpoofNet(pretrained_backbone=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        # Operational threshold
        self.threshold = threshold if threshold is not None else ckpt.get("threshold", Config.LIVENESS_THRESHOLD)

        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.inference_mode()
    def check_liveness(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[bool, float]:
        """
        Evaluates face crop presentation.
        
        Returns:
            is_live (bool): True if probability >= threshold, False otherwise.
            score (float): Live probability in range [0.0, 1.0].
        """
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as opened:
                img = opened.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Convert BGR (OpenCV) to RGB
            if img_input.ndim == 3 and img_input.shape[2] == 3:
                img_input = img_input[..., ::-1]
            img = Image.fromarray(img_input)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logit = self.model(tensor)
        score = float(torch.sigmoid(logit).item())
        is_live = bool(score >= self.threshold)

        return is_live, score
