"""Face analysis engine: detect -> liveness -> embedding + 1:N match -> emotion.

Uses the group's modules wherever they exist:
  - src.anti_spoofing.detector.LivenessDetector   (checkpoints/best_antispoof.pt)
  - src.emotion.detector.EmotionDetector           (checkpoints/best_emotion.pt)
  - src.innovations.adaptive_fusion.AdaptiveSecurityFusion (temporal smoothing, live stream)
  - any embedding model in src/api/models.py (group triplet/classification, Member A softmax/ArcFace)
Face detection uses OpenCV's Haar cascade (src/common/face_detector.py is still empty).
"""
import io
import logging
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.innovations.adaptive_fusion import AdaptiveSecurityFusion

from . import models as registry
from .gallery import AttendanceLog, Gallery
from .settings import Settings

log = logging.getLogger("face_api")

# Images at most this size with no detected face are treated as already-cropped faces
# (e.g. the 64x64 dataset images).
PRECROPPED_MAX_SIDE = 160


class HaarFaceDetector:
    """OpenCV Haar cascade; ships with opencv-python, no download needed."""

    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")

    def detect(self, rgb: np.ndarray):
        gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        min_side = max(40, min(gray.shape[:2]) // 10)
        boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
        return [tuple(int(v) for v in b) for b in boxes]  # (x, y, w, h)


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> RGB uint8 array, honouring EXIF orientation (phone photos)."""
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def expand_box(box, shape, margin=0.2):
    """Haar boxes are tight; the training crops include some context around the face."""
    x, y, w, h = box
    H, W = shape[:2]
    dx, dy = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    return x0, y0, min(W, x + w + dx) - x0, min(H, y + h + dy) - y0


class FaceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = HaarFaceDetector()
        self.gallery = Gallery(settings.gallery_dir)
        self.attendance = AttendanceLog(settings.attendance_dir, settings.attendance_cooldown_s)
        self.fusion = AdaptiveSecurityFusion()
        self._lock = threading.RLock()
        self._extractors = {}

        self.liveness = self._optional("liveness", settings.antispoof_checkpoint)
        self.emotion = self._optional("emotion", settings.emotion_checkpoint)
        self.liveness_enabled = self.liveness is not None
        self.emotion_enabled = self.emotion is not None

        avail = registry.available(settings.checkpoint_dir)
        if not avail:
            raise RuntimeError(f"No embedding checkpoint found in {settings.checkpoint_dir} "
                               f"(expected one of {[m.checkpoint for m in registry.MODELS]})")
        key = settings.model or avail[0].key
        self.set_model(key, threshold=float(settings.threshold) if settings.threshold else None)

    # ------------------------------------------------------------------ setup
    def _optional(self, kind, ckpt):
        if not ckpt.is_file():
            log.warning("%s disabled: checkpoint not found at %s", kind, ckpt)
            return None
        try:
            if kind == "liveness":
                from src.anti_spoofing.detector import LivenessDetector
                return LivenessDetector(str(ckpt), threshold=self.settings.liveness_threshold, device=self.settings.device)
            from src.emotion.detector import EmotionDetector
            return EmotionDetector(str(ckpt), device=self.settings.device)
        except Exception as exc:  # an incompatible checkpoint must not stop the app
            log.warning("%s disabled: %s", kind, exc)
            return None

    def set_model(self, key, threshold=None):
        model = registry.MODELS_BY_KEY.get(key)
        if model is None:
            raise ValueError(f"Unknown model '{key}'")
        if not registry.checkpoint_path(model, self.settings.checkpoint_dir).is_file():
            raise ValueError(f"Checkpoint for '{key}' not found: {model.checkpoint}")
        with self._lock:
            if key not in self._extractors:
                ext = registry.load_extractor(model, self.settings.checkpoint_dir, self.settings.device)
                ext.dim = int(ext.get_embedding(Image.new("RGB", (64, 64))).shape[0])   # warm-up
                self._extractors[key] = ext
            self.model = model
            self.extractor = self._extractors[key]
            self.threshold = threshold if threshold is not None else (
                model.threshold if model.threshold is not None else registry.UNCALIBRATED_THRESHOLD)
            reembedded = self.gallery.ensure_model(key, self.extractor.extract)
        log.info("Active model %s (threshold %.4f); re-embedded %d people", key, self.threshold, reembedded)
        return reembedded

    def models(self):
        avail = {m.key for m in registry.available(self.settings.checkpoint_dir)}
        return [{"key": m.key, "label": m.label, "owner": m.owner, "checkpoint": m.checkpoint,
                 "available": m.key in avail, "active": m.key == self.model.key,
                 "threshold": m.threshold, "calibrated": m.threshold is not None, "val_auc": m.auc}
                for m in registry.MODELS]

    def status(self):
        return {
            "model": self.model.key, "model_label": self.model.label, "threshold": self.threshold,
            "embedding_dim": self.extractor.dim,
            "device": str(getattr(self.extractor, "device", "")), "detector": "opencv-haar",
            "liveness": {"available": self.liveness is not None, "enabled": self.liveness_enabled,
                         "threshold": getattr(self.liveness, "threshold", None)},
            "emotion": {"available": self.emotion is not None, "enabled": self.emotion_enabled},
            "enrolled": len(self.gallery),
        }

    # ------------------------------------------------------------------ helpers
    def faces(self, rgb: np.ndarray, largest_only=False):
        """[(box, crop, precropped)]; small images without a detection count as one face."""
        boxes = self.detector.detect(rgb)
        if not boxes:
            if max(rgb.shape[:2]) <= PRECROPPED_MAX_SIDE:
                h, w = rgb.shape[:2]
                return [((0, 0, w, h), Image.fromarray(rgb), True)]
            return []
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[: 1 if largest_only else None]
        out = []
        for b in boxes:
            x, y, w, h = expand_box(b, rgb.shape)
            out.append((b, Image.fromarray(np.ascontiguousarray(rgb[y:y + h, x:x + w])), False))
        return out

    # ------------------------------------------------------------------ actions
    def analyze(self, rgb: np.ndarray, log_attendance=True, stream=False):
        """stream=True: frames of one live camera; liveness/emotion are smoothed over
        recent frames (AdaptiveSecurityFusion) while exactly one face is visible."""
        t0 = time.perf_counter()
        full_frame = Image.fromarray(rgb)
        with self._lock:
            found = self.faces(rgb)
            smooth = stream and len(found) == 1
            if stream and not smooth:
                self.fusion.reset_session()
            results = [self._analyze_face(box, crop, pre, smooth, log_attendance, full_frame=full_frame)
                       for box, crop, pre in found]
        h, w = rgb.shape[:2]
        return {"faces": results, "image_size": [w, h], "model": self.model.key, "threshold": self.threshold,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def _analyze_face(self, box, crop, precropped, smooth, log_attendance, full_frame=None):
        item = {"box": list(box), "precropped": precropped, "liveness": None, "emotion": None,
                "identity": None, "access_granted": False, "rejection_reason": None, "checked_in": False,
                "template_adapted": False}
        if self.liveness_enabled:
            liveness_input = full_frame if full_frame is not None else crop
            live, score = self.liveness.check_liveness(liveness_input)
            item["liveness"] = {"passed": live, "score": round(score, 4), "smoothed": False}
            if smooth:
                s = self.fusion.stabilize_liveness(score, self.liveness.threshold)
                item["liveness"] = {"passed": s["passed"], "score": s["smoothed_score"], "raw_score": s["raw_score"],
                                    "smoothed": True, "frames": s["frames_buffered"]}
            if not item["liveness"]["passed"]:
                item["rejection_reason"] = "SPOOF_ATTACK_DETECTED"
                return item

        emb = self.extractor.get_embedding(crop)
        item["identity"] = self.gallery.identify(emb, self.model.key, self.threshold)
        item["access_granted"] = item["identity"]["matched"]
        if not item["access_granted"]:
            item["rejection_reason"] = "IDENTITY_UNRECOGNIZED"
        elif smooth and item["liveness"]:
            stored = self.gallery.get_template(item["identity"]["id"], self.model.key)
            if stored is not None:
                adapted = self.fusion.compute_adapted_template(
                    stored, emb, item["liveness"]["score"], item["identity"]["similarity"]
                )
                if adapted is not None:
                    self.gallery.update_template(item["identity"]["id"], self.model.key, adapted)
                    item["template_adapted"] = True

        if self.emotion_enabled:
            e = self.emotion.predict_emotion(crop)
            item["emotion"] = {"dominant": e["dominant_emotion"], "confidence": e["confidence"],
                               "distribution": e["distribution"], "smoothed": False}
            if smooth:
                s = self.fusion.stabilize_emotion(e["distribution"])
                item["emotion"] = {"dominant": s["dominant_emotion"], "confidence": s["confidence"],
                                   "distribution": s["smoothed_distribution"], "smoothed": True}

        if log_attendance and item["access_granted"]:
            ident = item["identity"]
            item["checked_in"] = self.attendance.record(
                ident["id"], ident["name"], self.model.key, ident["similarity"],
                item["liveness"]["score"] if item["liveness"] else None,
                item["emotion"]["dominant"] if item["emotion"] else None)
        return item

    def enroll(self, name, images, person_id=None):
        crops, skipped = [], 0
        for rgb in images:
            found = self.faces(rgb, largest_only=True)
            if found:
                crops.append(found[0][1])
            else:
                skipped += 1
        if not crops:
            return None, skipped
        with self._lock:
            embs = self.extractor.extract(crops)
            person = self.gallery.enroll(name, crops, self.model.key, embs, person_id)
        return person, skipped

    def verify(self, rgb_a, rgb_b):
        fa, fb = self.faces(rgb_a, largest_only=True), self.faces(rgb_b, largest_only=True)
        if not fa or not fb:
            return None
        with self._lock:
            ea, eb = self.extractor.extract([fa[0][1], fb[0][1]])
            sim = float(np.dot(ea, eb))
            return {"similarity": round(sim, 4), "threshold": self.threshold, "match": sim >= self.threshold,
                    "model": self.model.key, "boxes": [list(fa[0][0]), list(fb[0][0])]}
