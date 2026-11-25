import os

import cv2
import numpy as np
import torch

from .model import EmotionNet
from src.config import Config

WEIGHTS_PATH = Config.EMOTION_CHECKPOINT
IMG_SIZE = Config.IMAGE_SIZE[0]  # 64
EMOTION_LABELS = [c.lower() for c in Config.EMOTION_CLASSES]  # 9 lớp, lowercase để tương thích API cũ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_classes = None


def _load_model():
    global _model, _classes
    if _model is None:
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        classes = ckpt.get("classes", Config.EMOTION_CLASSES)
        _classes = [c.lower() for c in classes]

        model = EmotionNet(num_classes=len(_classes), pretrained_backbone=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model, _classes


def _preprocess(image):
    """Convert a face crop (BGR/RGB/gray numpy array or PIL image) to a tensor.

    Matches the training/detector pipeline: RGB 64x64, normalized to [-1, 1].
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if image.ndim == 3 and image.shape[2] == 3:
        # OpenCV mặc định là BGR -> chuyển sang RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        rgb = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image

    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5  # cùng chuẩn hóa với EmotionDetector

    # HWC -> CHW -> NCHW
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def predict_emotion(image, return_all=False):
    """
    Predict the emotion of a single cropped face.

    Args:
        image: face crop as a numpy array (OpenCV BGR or grayscale) or PIL image.
        return_all: if True, also return a dict of probabilities for every class.

    Returns:
        (label, confidence)            e.g. ("happy", 0.82)
        or (label, confidence, probs)  if return_all=True
    """
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


def _default_haar_path() -> str:
    """Tìm file Haar Cascade frontal face mặc định của OpenCV (nếu có)."""
    # Ưu tiên file đã đóng gói trong dự án (không phụ thuộc vào cv2.data)
    project_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
        "common",
        "haarcascade_frontalface_default.xml",
    )

    candidates = [
        project_path,
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]  # fallback để CascadeClassifier ném lỗi rõ ràng nếu không tìm thấy


if __name__ == "__main__":
    # Quick webcam test: press q to quit.
    cap = cv2.VideoCapture(0)
    haar_path = _default_haar_path()
    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        print(f"[!] Không load được Haar Cascade từ: {haar_path}")
        print("    Cài đặt: pip install opencv-python (không phải opencv-python-headless)")
        cap.release()
        raise SystemExit(1)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.3, 5):
            face = frame[y:y + h, x:x + w]
            label, conf = predict_emotion(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("emotion test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()