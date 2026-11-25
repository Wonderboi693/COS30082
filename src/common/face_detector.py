"""
Common face detection & alignment wrapper.

Supports MTCNN (facenet-pytorch) as primary detector, with MediaPipe and
OpenCV cascade fallbacks.

Returns bounding boxes and cropped PIL Images matching Config.IMAGE_SIZE.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.config import Config

BBox = Tuple[int, int, int, int, float]  # (x1, y1, x2, y2, confidence)
CropResult = Tuple[Image.Image, BBox, float]


def _try_mtcnn(device: str):
    """Lazy-import MTCNN from facenet-pytorch."""
    try:
        import torch
        from facenet_pytorch import MTCNN

        dev = device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        mtcnn = MTCNN(
            image_size=Config.IMAGE_SIZE[0],
            margin=10,
            min_face_size=20,
            thresholds=[0.5, 0.6, 0.6],
            keep_all=True,
            post_process=False,
            device=dev,
            select_largest=False,
        )
        return mtcnn, "mtcnn"
    except Exception as e:
        return None, None


def _try_mediapipe():
    """Lazy-import MediaPipe Face Detection if available."""
    try:
        import mediapipe as mp

        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
        return detector, "mediapipe"
    except Exception:
        return None, None


def _find_haar_cascade() -> str:
    """Tìm file haarcascade_frontalface_default.xml của OpenCV (nếu có).

    Trả về đường dẫn đầu tiên tồn tại, hoặc đường dẫn mặc định để để caller
    tự xử lý trường hợp không tìm thấy.
    """
    import cv2
    import os

    # Ưu tiên file đã đóng gói trong dự án (không phụ thuộc vào cv2.data)
    project_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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
    return candidates[0]


def _try_opencv():
    """OpenCV cascade fallback."""
    try:
        import cv2

        path = _find_haar_cascade()
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return None, None
        return cascade, "opencv"
    except Exception:
        return None, None


class FaceDetector:
    """
    Unified face detection & crop interface.
    """

    def __init__(
        self,
        backend: str = "auto",
        target_size: Tuple[int, int] = Config.IMAGE_SIZE,
        min_face_size: int = 40,
        device: str = "auto",
    ):
        self.target_size = target_size
        self.min_face_size = min_face_size
        self._backend_name: str = ""
        self._detector = None

        def _resolve_device(d: str) -> str:
            """Resolve 'auto' to a concrete torch device (mirrors torch/facenet-pytorch)."""
            if d == "auto":
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                return "cpu"
            return d

        resolved_device = _resolve_device(device)

        if backend == "auto":
            # facenet-pytorch MTCNN crashes on Apple Silicon (MPS) because
            # adaptive_avg_pool2d requires input sizes divisible by output sizes
            # (pytorch/pytorch#96056). Run MTCNN on CPU instead so real face
            # detection still works on Mac; MediaPipe / OpenCV are fallbacks.
            mtcnn_device = "cpu" if resolved_device == "mps" else resolved_device
            self._detector, self._backend_name = _try_mtcnn(mtcnn_device)
            if resolved_device == "mps" and self._backend_name == "mtcnn":
                print("[FaceDetector] MPS incompatible with MTCNN; running MTCNN on CPU.")
            if self._detector is None:
                self._detector, self._backend_name = _try_mediapipe()
            if self._detector is None:
                self._detector, self._backend_name = _try_opencv()
        elif backend == "mtcnn":
            self._detector, self._backend_name = _try_mtcnn(resolved_device)
        elif backend == "mediapipe":
            self._detector, self._backend_name = _try_mediapipe()
        elif backend == "opencv":
            self._detector, self._backend_name = _try_opencv()

        if self._detector is None:
            # Fallback simple dummy / center crop if no detector available
            warnings.warn("No face detection backend available; falling back to full-image pass-through.")
            self._backend_name = "passthrough"

        print(f"[FaceDetector] Backend: {self._backend_name.upper()} | Target size: {self.target_size}")

    def detect(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        color_order: str = "BGR",
    ) -> List[BBox]:
        pil_img = self._to_pil(image, color_order)
        return self._run_detection(pil_img)

    def detect_and_crop(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        color_order: str = "BGR",
        margin: float = 0.1,
    ) -> List[CropResult]:
        pil_img = self._to_pil(image, color_order)
        boxes = self._run_detection(pil_img)

        results: List[CropResult] = []
        img_w, img_h = pil_img.size

        if not boxes and self._backend_name == "passthrough":
            crop = pil_img.resize(self.target_size, Image.BILINEAR)
            return [(crop, (0, 0, img_w, img_h, 1.0), 1.0)]

        for x1, y1, x2, y2, conf in boxes:
            w, h = x2 - x1, y2 - y1
            if w < self.min_face_size or h < self.min_face_size:
                continue

            mx = int(w * margin)
            my = int(h * margin)
            cx1 = max(0, x1 - mx)
            cy1 = max(0, y1 - my)
            cx2 = min(img_w, x2 + mx)
            cy2 = min(img_h, y2 + my)

            crop = pil_img.crop((cx1, cy1, cx2, cy2)).resize(
                self.target_size, Image.BILINEAR
            )
            results.append((crop, (x1, y1, x2, y2, conf), conf))

        results.sort(key=lambda t: t[2], reverse=True)
        return results

    def largest_face(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        color_order: str = "BGR",
    ) -> Optional[Tuple[Image.Image, BBox]]:
        crops = self.detect_and_crop(image, color_order=color_order)
        if not crops:
            return None
        crops.sort(key=lambda t: (t[1][2] - t[1][0]) * (t[1][3] - t[1][1]), reverse=True)
        crop, bbox, _ = crops[0]
        return crop, bbox

    @property
    def backend(self) -> str:
        return self._backend_name

    @staticmethod
    def _to_pil(
        image: Union[str, Path, np.ndarray, Image.Image],
        color_order: str = "BGR",
    ) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return opened.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3 and image.shape[2] == 3 and color_order == "BGR":
                image = image[..., ::-1].copy()
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _run_detection(self, pil_img: Image.Image) -> List[BBox]:
        if self._backend_name == "mtcnn":
            return self._detect_mtcnn(pil_img)
        if self._backend_name == "mediapipe":
            return self._detect_mediapipe(pil_img)
        if self._backend_name == "opencv":
            return self._detect_opencv(pil_img)
        return []

    def _detect_mtcnn(self, pil_img: Image.Image) -> List[BBox]:
        try:
            boxes_raw, probs = self._detector.detect(pil_img)
        except Exception as e:  # e.g. MPS adaptive_pool divisibility errors
            warnings.warn(f"MTCNN detection failed ({e}); treating frame as no-face")
            return []
        if boxes_raw is None:
            return []
        results = []
        w_img, h_img = pil_img.size
        for box, prob in zip(boxes_raw, probs):
            if prob is None or prob < 0.3:
                continue
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w_img, int(box[2]))
            y2 = min(h_img, int(box[3]))
            if x2 > x1 and y2 > y1:
                results.append((x1, y1, x2, y2, float(prob)))
        return results

    def _detect_mediapipe(self, pil_img: Image.Image) -> List[BBox]:
        img_arr = np.array(pil_img)
        h, w = img_arr.shape[:2]
        result = self._detector.process(img_arr)
        if not result.detections:
            return []
        boxes = []
        for det in result.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            conf = det.score[0] if det.score else 0.0
            boxes.append((x1, y1, x2, y2, float(conf)))
        return boxes

    def _detect_opencv(self, pil_img: Image.Image) -> List[BBox]:
        import cv2

        gray = np.array(pil_img.convert("L"))
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(faces) == 0:
            return []
        return [
            (int(x), int(y), int(x + w), int(y + h), 0.9)
            for x, y, w, h in faces
        ]
