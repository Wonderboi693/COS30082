import os

import cv2
import numpy as np
import torch

from .model import EmotionCNN  

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "weights", "emotion_best.pth"
)
IMG_SIZE = 48
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


def _load_model():
    global _model
    if _model is None:
        model = EmotionCNN(num_classes=len(EMOTION_LABELS))
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


def _preprocess(image):
    """Convert a face crop (BGR/RGB/gray numpy array or PIL image) to a tensor."""
    if not isinstance(image, np.ndarray):
        image = np.array(image)  

    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = image

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    gray = (gray - 0.5) / 0.5

    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  
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
    model = _load_model()
    x = _preprocess(image)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    label = EMOTION_LABELS[idx]
    confidence = float(probs[idx])

    if return_all:
        all_probs = {name: float(p) for name, p in zip(EMOTION_LABELS, probs)}
        return label, confidence, all_probs
    return label, confidence


if __name__ == "__main__":
    # Quick webcam test: press q to quit.
    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in detector.detectMultiScale(gray, 1.3, 5):
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