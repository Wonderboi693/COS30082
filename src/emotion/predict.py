import os

import cv2
import numpy as np
import torch

from .model import EmotionNet
from src.config import Config

WEIGHTS_PATH = Config.EMOTION_CHECKPOINT
IMG_SIZE = Config.IMAGE_SIZE[0]  # 64
EMOTION_LABELS = [c.lower() for c in Config.EMOTION_CLASSES]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_classes = None


def _load_model():
    global _model, _classes
    if _model is None:
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        classes = ckpt.get("classes", Config.EMOTION_CLASSES)
        _classes = [c.lower() for c in classes]
        backbone = ckpt.get("backbone", ckpt.get("train_args", {}).get("backbone", "resnet18"))

        model = EmotionNet(num_classes=len(_classes), pretrained_backbone=False, backbone=backbone)
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model, _classes


def _preprocess(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        rgb = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image

    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def predict_emotion(image, return_all=False):
    model, classes = _load_model()
    x = _preprocess(image)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    label = classes[idx]
    confidence = float(probs[idx])

    if return_all:
        all_probs = {name: float(p) for name, p in zip(classes, probs)}
        return label, confidence, all_probs
    return label, confidence
