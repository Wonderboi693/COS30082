from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


def _resolve_cascade_path(filename: str = "haarcascade_frontalface_default.xml") -> str:
    local_path = Path("data/haarcascades") / filename
    if local_path.exists():
        return str(local_path)

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidate = Path(cv2.data.haarcascades) / filename
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Could not locate {filename} in data/haarcascades/ or cv2.data.")

class FaceDetector:
    """
    Fast real-time face detector & padded cropper for live webcam frames (Task 4.2.2).
    """
    def __init__(
        self,
        scale_factor: float = 1.15,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (80, 80),
        margin_ratio: float = 0.20
    ):
        cascade_path = _resolve_cascade_path("haarcascade_frontalface_default.xml")
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.margin_ratio = margin_ratio

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """
        Detects faces in a BGR frame and returns a list of (x1, y1, x2, y2, padded_crop_bgr)
        sorted largest-face-first.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        if len(boxes) == 0:
            return []

        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        results = []

        for (x, y, bw, bh) in boxes:
            pad_x = int(bw * self.margin_ratio)
            pad_y = int(bh * self.margin_ratio)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            crop = frame_bgr[y1:y2, x1:x2].copy()
            if crop.size > 0:
                results.append((x1, y1, x2, y2, crop))

        return results
