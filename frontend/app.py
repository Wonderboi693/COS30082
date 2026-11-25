import argparse
import sys
import threading
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.common.face_detector import FaceDetector
from src.anti_spoofing.detector import LivenessDetector
from src.emotion.detector import EmotionDetector
from src.face_recognition.registry import FaceRegistry
from src.utils import AttendanceLogger

# ----------------------------------------------------------------------------- style
WINDOW = "Facial Recognition Attendance System - COS30082"
PANEL_W = 330
MIN_HEIGHT = 700
TARGET_FPS = 15

BG = (24, 24, 28)
PANEL = (36, 36, 42)
CHIP = (44, 44, 52)
TEXT = (235, 235, 235)
MUTED = (150, 150, 158)
GREEN = (90, 200, 90)
RED = (70, 70, 235)
AMBER = (0, 190, 255)
ACCENT = (235, 145, 60)
DARK = (20, 20, 20)
FEATURE = (25, 25, 25)

EMOTION_COLORS = {
    "angry": (70, 70, 225),
    "confused": (140, 200, 100),
    "disgust": (90, 170, 90),
    "fear": (190, 120, 170),
    "happy": (60, 205, 245),
    "neutral": (175, 175, 175),
    "sad": (205, 135, 75),
    "shy": (200, 140, 210),
    "surprise": (245, 175, 70),
    "uncertain": (130, 130, 138),
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).is_file()), None)
_font_cache: Dict[int, ImageFont.FreeTypeFont] = {}


def get_font(size: int):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


def fold(s: str) -> str:
    """Strip accents for the ASCII-only fallback renderer."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def text_width(s: str, size: int) -> int:
    if FONT_PATH:
        return int(get_font(size).getlength(s))
    return cv2.getTextSize(fold(s), cv2.FONT_HERSHEY_SIMPLEX, size / 32, 1)[0][0]


class Canvas:
    """Queues text and draws all of it in one PIL pass (handles Unicode, e.g. Vietnamese names)."""

    def __init__(self, img: np.ndarray):
        self.img = img
        self._items: List[tuple] = []

    def text(self, s, xy, size=15, color=TEXT, anchor="la"):
        self._items.append((str(s), (int(xy[0]), int(xy[1])), size, color, anchor))

    def flush(self):
        if not self._items:
            return
        if FONT_PATH:
            pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for s, xy, size, color, anchor in self._items:
                draw.text(xy, s, font=get_font(size), fill=(color[2], color[1], color[0]), anchor=anchor)
            self.img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        else:
            for s, (x, y), size, color, anchor in self._items:
                s = fold(s)
                scale = size / 32
                w = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                if anchor[0] == "r":
                    x -= w
                elif anchor[0] == "m":
                    x -= w // 2
                cv2.putText(self.img, s, (x, y + int(size * 0.85)), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 1, cv2.LINE_AA)
        self._items.clear()


# ----------------------------------------------------------------------------- drawing helpers
def shade(img, rect, color, alpha):
    x1, y1, x2, y2 = rect
    sub = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1 - alpha, np.full_like(sub, color), alpha, 0)


def draw_brackets(img, box, color, t=2):
    x1, y1, x2, y2 = box
    length = max(8, int(0.22 * min(x2 - x1, y2 - y1)))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * length, py), color, t, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * length), color, t, cv2.LINE_AA)


def draw_emotion_icon(img, center, r, emotion: str):
    """Vector face icon for one of the 7 emotions (no emoji font needed)."""
    key = emotion.lower()
    cx, cy = int(center[0]), int(center[1])
    face = EMOTION_COLORS.get(key, (175, 175, 175))
    t = max(2, r // 9)
    cv2.circle(img, (cx, cy), r, face, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)

    def P(dx, dy):
        return (int(cx + dx * r), int(cy + dy * r))

    def dot(dx, dy, rad):
        cv2.circle(img, P(dx, dy), max(2, int(rad * r)), FEATURE, -1, cv2.LINE_AA)

    def arc(dx, dy, ax, ay, a0, a1):
        cv2.ellipse(img, P(dx, dy), (max(2, int(ax * r)), max(2, int(ay * r))), 0, a0, a1,
                    FEATURE, t, cv2.LINE_AA)

    def line(p, q):
        cv2.line(img, P(*p), P(*q), FEATURE, t, cv2.LINE_AA)

    if key == "happy":
        dot(-0.35, -0.25, 0.09); dot(0.35, -0.25, 0.09)
        arc(0, 0.05, 0.5, 0.42, 15, 165)
    elif key == "sad":
        dot(-0.35, -0.2, 0.09); dot(0.35, -0.2, 0.09)
        arc(0, 0.62, 0.4, 0.28, 195, 345)
        cv2.circle(img, P(0.5, 0.05), max(2, int(0.08 * r)), (235, 205, 130), -1, cv2.LINE_AA)
    elif key == "angry":
        dot(-0.33, -0.12, 0.09); dot(0.33, -0.12, 0.09)
        line((-0.6, -0.5), (-0.12, -0.28)); line((0.6, -0.5), (0.12, -0.28))
        arc(0, 0.6, 0.32, 0.2, 195, 345)
    elif key == "surprise":
        dot(-0.35, -0.25, 0.12); dot(0.35, -0.25, 0.12)
        cv2.circle(img, P(0, 0.42), max(3, int(0.2 * r)), FEATURE, t, cv2.LINE_AA)
    elif key == "fear":
        dot(-0.35, -0.2, 0.11); dot(0.35, -0.2, 0.11)
        line((-0.6, -0.4), (-0.15, -0.58)); line((0.6, -0.4), (0.15, -0.58))
        cv2.ellipse(img, P(0, 0.5), (max(2, int(0.22 * r)), max(2, int(0.15 * r))), 0, 0, 360,
                    FEATURE, t, cv2.LINE_AA)
    elif key == "disgust":
        line((-0.5, -0.22), (-0.2, -0.22)); line((0.2, -0.22), (0.5, -0.22))
        pts = np.array([P(-0.42, 0.5), P(-0.2, 0.36), P(0.0, 0.52), P(0.2, 0.36), P(0.42, 0.52)], np.int32)
        cv2.polylines(img, [pts], False, FEATURE, t, cv2.LINE_AA)
    elif key == "confused":
        dot(-0.35, -0.22, 0.08); dot(0.35, -0.28, 0.1)
        line((-0.55, -0.38), (-0.15, -0.42)); line((0.15, -0.48), (0.55, -0.36))
        arc(0, 0.45, 0.35, 0.25, 20, 160)
    elif key == "shy":
        dot(-0.35, -0.2, 0.08); dot(0.35, -0.2, 0.08)
        cv2.circle(img, P(-0.55, 0.05), max(2, int(0.12 * r)), (180, 150, 245), -1, cv2.LINE_AA)
        cv2.circle(img, P(0.55, 0.05), max(2, int(0.12 * r)), (180, 150, 245), -1, cv2.LINE_AA)
        arc(0, 0.3, 0.28, 0.2, 20, 160)
    elif key == "uncertain":
        cv2.putText(img, "?", (cx - int(0.3 * r), cy + int(0.4 * r)), cv2.FONT_HERSHEY_SIMPLEX,
                    r / 20.0, FEATURE, t, cv2.LINE_AA)
    else:  # neutral / unknown
        dot(-0.35, -0.25, 0.09); dot(0.35, -0.25, 0.09)
        line((-0.35, 0.42), (0.35, 0.42))


# ----------------------------------------------------------------------------- camera thread
class FrameGrabber(threading.Thread):
    """Reads the camera continuously and keeps only the newest frame (no queue build-up)."""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap = cap
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._idx = 0
        self._halt = threading.Event()
        self.failed = False
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def run(self):
        fails = 0
        while not self._halt.is_set():
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    self.failed = True
                    return
                time.sleep(0.01)
                continue
            fails = 0
            with self._lock:
                self._frame, self._idx = frame, self._idx + 1

    def latest(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._frame, self._idx

    def stop(self):
        self._halt.set()
        self.join(timeout=1.0)


# ----------------------------------------------------------------------------- tracking
@dataclass
class FaceView:
    """Immutable snapshot of one tracked face, safe to read from the UI thread."""
    tid: int
    box: Tuple[int, int, int, int]
    is_live: bool
    live_score: float
    name: str
    conf: float
    emo: str
    emo_conf: float
    dist: Dict[str, float]
    emo_note: str = ""   # why there is no emotion yet (face too small, ...)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class Track:
    tid: int
    box: Tuple[int, int, int, int]
    last_seen: float
    crop: Optional[Image.Image] = None
    live_hist: Deque[bool] = field(default_factory=lambda: deque(maxlen=5))
    live_score: float = 1.0
    emo_hist: Deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=5))
    id_hist: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=5))
    n_obs: int = 0
    spoof_logged_at: float = 0.0
    logged_unknown: bool = False
    emo_note: str = ""

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def is_live(self) -> bool:
        if not self.live_hist:
            return True
        return sum(self.live_hist) * 2 > len(self.live_hist)  # strict majority

    def identity(self) -> Tuple[str, float]:
        if not self.id_hist:
            return "Unknown", 0.0
        names = [n for n, _ in self.id_hist]
        top, count = Counter(names).most_common(1)[0]
        if count * 2 <= len(names):
            return "Unknown", 0.0
        return top, float(np.mean([c for n, c in self.id_hist if n == top]))

    def emotion(self) -> Tuple[str, float, Dict[str, float]]:
        if not self.emo_hist:
            return "N/A", 0.0, {}
        acc: Dict[str, float] = {}
        for dist in list(self.emo_hist):
            for k, v in dist.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        total = sum(acc.values()) or 1.0
        avg = {k: v / total for k, v in acc.items()}
        label = max(avg, key=avg.get)
        return label, avg[label], avg

    def emotion_label(self, thresh: float) -> Tuple[str, float, Dict[str, float]]:
        """Smoothed emotion; always returns the model's top label (no 'Uncertain' fallback)."""
        emo, econf, dist = self.emotion()
        return emo, econf, dist

    def view(self, emo_thresh: float = 0.0) -> FaceView:
        name, conf = self.identity()
        emo, econf, dist = self.emotion_label(emo_thresh)
        return FaceView(self.tid, self.box, self.is_live, self.live_score, name, conf,
                        emo, econf, dist, self.emo_note)


# ----------------------------------------------------------------------------- application
class AttendanceApp:
    def __init__(
        self,
        camera_id: int = 0,
        liveness_thresh: float = Config.LIVENESS_THRESHOLD,
        verification_thresh: float = getattr(Config, "VERIFICATION_THRESHOLD", 0.3332),
        metric: str = "cosine",
        device: str = "auto",
        ai_fps: float = 10.0,
        mirror: bool = True,
        terminal_name: bool = False,
        min_face_size: int = 60,
        min_det_conf: float = 0.5,
        emotion_conf_thresh: float = 0.45,
        emotion_margin: float = 0.25,
        use_tta: bool = True,
        detector_backend: str = "auto",
    ):
        self.camera_id = camera_id
        self.liveness_thresh = liveness_thresh
        self.verification_thresh = verification_thresh
        self.metric = metric
        self.device = device
        self.ai_fps = ai_fps
        self.mirror = mirror
        self.terminal_name = terminal_name
        self.enforce_liveness = True
        self.min_face_size = min_face_size
        self.min_det_conf = min_det_conf
        self.emotion_conf_thresh = emotion_conf_thresh
        self.emotion_margin = emotion_margin
        self.use_tta = use_tta

        print("[*] Initializing Attendance System Modules...")
        self.face_detector = FaceDetector(backend=detector_backend, device=device)

        self.liveness_model: Optional[LivenessDetector] = None
        try:
            self.liveness_model = LivenessDetector(threshold=liveness_thresh, device=device)
            print("[✓] Anti-Spoofing Liveness Gate loaded.")
        except Exception as e:
            print(f"[!] Liveness model NOT loaded: {e}")

        self.emotion_model: Optional[EmotionDetector] = None
        try:
            self.emotion_model = EmotionDetector(device=device)
            print("[✓] Emotion Recognition Engine loaded.")
        except Exception as e:
            print(f"[!] Emotion model NOT loaded: {e}")
        self._tta_fn = getattr(self.emotion_model, "predict_emotion_with_tta", None)
        if self.use_tta and self.emotion_model is not None and self._tta_fn is None:
            print("[!] EmotionDetector has no predict_emotion_with_tta(): using plain predict_emotion().")

        self.registry = FaceRegistry(
            gallery_dir=getattr(Config, "GALLERY_DIR", "data/gallery"),
            threshold=verification_thresh, metric=metric, device=device,
        )
        print(f"[✓] Face Registry loaded: {len(self.registry.list_identities())} enrolled identities.")
        self.logger = AttendanceLogger(cooldown_seconds=30.0)

        self.emotions: List[str] = list(self.emotion_model.classes) if self.emotion_model else list(Config.EMOTION_CLASSES)

        # shared state ------------------------------------------------------------
        self.lock = threading.RLock()
        self._halt = threading.Event()
        self._reset = False
        self.grabber: Optional[FrameGrabber] = None
        self.tracks: List[Track] = []          # owned by the analysis thread
        self._next_tid = 1
        self.views: List[FaceView] = []        # published snapshot (atomic reference swap)
        self.ai_ms = 0.0
        self.ai_hz = 0.0
        self._warned: set = set()
        self.recent: Deque[str] = deque(maxlen=6)
        self.last_canvas: Optional[np.ndarray] = None

        # enrollment state machine: idle -> typing -> capture -> saving -> idle
        self.mode = "idle"
        self.typed = ""
        self.enroll_name = ""
        self.samples: List[Image.Image] = []
        self.target_samples = 5
        self.last_sample_t = 0.0
        self.msg = ""
        self.msg_until = 0.0

        self.show_analytics = False
        self.analytics: dict = {}
        self.analytics_t = 0.0
        self.show_help = False
        self.show_coords = True
        self.buttons: Dict[str, Tuple[int, int, int, int]] = {}

    # ------------------------------------------------------------------ utils
    @property
    def liveness_active(self) -> bool:
        return self.liveness_model is not None and self.enforce_liveness

    def _warn_once(self, key: str, err: Exception):
        if key not in self._warned:
            self._warned.add(key)
            print(f"[!] {key} error (further errors suppressed): {err}")

    def _flash(self, text: str, secs: float = 3.0):
        with self.lock:
            self.msg, self.msg_until = text, time.time() + secs

    def _open_camera(self, target_id: int) -> Tuple[Optional[cv2.VideoCapture], int]:
        for cam in (target_id, 1 if target_id == 0 else 0):
            cap = cv2.VideoCapture(cam)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"[✓] Opened camera {cam}")
                    return cap, cam
                cap.release()
            print(f"[!] Camera {cam} unavailable")
        return None, target_id

    # ------------------------------------------------------------------ analysis thread
    def _analysis_loop(self):
        last_idx, last_t = -1, 0.0
        min_dt = 1.0 / self.ai_fps if self.ai_fps > 0 else 0.0
        while not self._halt.is_set():
            grabber = self.grabber
            frame, idx = grabber.latest() if grabber is not None else (None, -1)
            if frame is None or idx == last_idx or time.time() - last_t < min_dt:
                time.sleep(0.003)
                continue
            last_idx, t0 = idx, time.perf_counter()
            period = time.time() - last_t if last_t else 1.0
            last_t = time.time()
            try:
                self.analyze(frame, last_t)
            except Exception as e:
                self._warn_once("analysis", e)
            self.ai_ms = 0.8 * self.ai_ms + 0.2 * (time.perf_counter() - t0) * 1000
            self.ai_hz = 0.8 * self.ai_hz + 0.2 / max(period, 1e-3)

    def analyze(self, frame: np.ndarray, now: float):
        if self._reset:
            self.tracks, self._reset = [], False
        detections = self.face_detector.detect_and_crop(frame, color_order="BGR")
        used = set()
        for crop, bbox, _ in detections:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
            det_conf = float(bbox[4]) if len(bbox) > 4 else 1.0
            size = max(x2 - x1, y2 - y1, 1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best, best_d = None, 0.75 * size
            for t in self.tracks:
                if t.tid in used:
                    continue
                tx, ty = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
                d = float(np.hypot(cx - tx, cy - ty))
                if d < best_d:
                    best, best_d = t, d
            if best is None:
                best = Track(self._next_tid, (x1, y1, x2, y2), now)
                self._next_tid += 1
                self.tracks.append(best)
            used.add(best.tid)
            best.box, best.last_seen, best.crop = (x1, y1, x2, y2), now, crop
            best.n_obs += 1
            self._run_models(best, frame, det_conf)
            self._maybe_log(best, now)
        self.tracks = [t for t in self.tracks if now - t.last_seen < 1.0]
        self.views = [t.view(self.emotion_conf_thresh) for t in self.tracks]  # publish to the UI thread
        self._enrollment_step(now)

    def _emotion_crop(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[Image.Image]:
        """Crop for the emotion model only: the box opened by `emotion_margin` (forehead / chin
        like the training images), as a PIL RGB image. Other models keep the detector's own crop."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = int((x2 - x1) * self.emotion_margin), int((y2 - y1) * self.emotion_margin)
        x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
        if x2 <= x1 or y2 <= y1:
            return None
        return Image.fromarray(np.ascontiguousarray(frame[y1:y2, x1:x2, ::-1]))  # BGR -> RGB

    def _run_models(self, t: Track, frame: np.ndarray, det_conf: float):
        crop = t.crop
        x1, y1, x2, y2 = t.box
        live, score = True, 1.0
        if self.liveness_active:
            try:
                live, score = self.liveness_model.check_liveness(crop)
                live, score = bool(live), float(score)
            except Exception as e:
                self._warn_once("liveness", e)
        t.live_hist.append(live)
        t.live_score = score

        if self.emotion_model is not None:
            t.emo_note = ""
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                t.emo_note = "move closer"
            elif det_conf < self.min_det_conf:
                t.emo_note = "low detection confidence"
            else:
                ecrop = self._emotion_crop(frame, t.box)
                if ecrop is not None:
                    try:
                        if self.use_tta and self._tta_fn is not None:
                            res = self._tta_fn(ecrop, n_tta=5)
                        else:
                            res = self.emotion_model.predict_emotion(ecrop)
                        t.emo_hist.append(dict(res["distribution"]))
                    except Exception as e:
                        self._warn_once("emotion", e)

        if live:
            try:
                m = self.registry.identify(crop, metric=self.metric)
                t.id_hist.append((str(m["name"]), float(m["confidence"])))
            except Exception as e:
                self._warn_once("identity", e)

    def _maybe_log(self, t: Track, now: float):
        if self.mode != "idle" or t.n_obs < 3:
            return
        emo, econf, _ = t.emotion_label(self.emotion_conf_thresh)
        if emo == "N/A":
            emo, econf = "N/A", 0.0
        rec = None
        if t.is_live:
            name, conf = t.identity()
            if name != "Unknown":
                rec = self.logger.log_checkin(name, t.live_score, True, conf, emo, econf)
            elif not t.logged_unknown:
                t.logged_unknown = True
                rec = self.logger.log_checkin("Unknown", t.live_score, True, 0.0, emo, econf)
        elif now - t.spoof_logged_at > 30.0:
            t.spoof_logged_at = now
            rec = self.logger.log_checkin("Spoof Attack", t.live_score, False, 0.0, emo, econf)
        if rec:
            self.recent.appendleft(f"{rec['time']}  {rec['employee_name']}  ({rec['checkin_status']})")

    def _enrollment_step(self, now: float):
        with self.lock:
            if self.mode != "capture":
                return
            live_tracks = [t for t in self.tracks if t.is_live and t.crop is not None and t.n_obs >= 2]
            if not live_tracks:
                return
            t = max(live_tracks, key=lambda x: x.area)
            if now - self.last_sample_t >= 0.5:
                self.samples.append(t.crop.copy())
                self.last_sample_t = now
            if len(self.samples) < self.target_samples:
                return
            name, images = self.enroll_name, list(self.samples)
            self.mode = "saving"
        # embedding extraction + saving may take a while: do it outside the lock
        try:
            self.registry.register(name=name, images=images)
            self._flash(f"Registered '{name}' successfully", 4.0)
        except Exception as e:
            self._flash(f"Registration failed: {e}", 5.0)
        with self.lock:
            self.mode, self.samples, self.enroll_name = "idle", [], ""

    # ------------------------------------------------------------------ rendering (main thread)
    def render(self, frame: np.ndarray, fps: float) -> np.ndarray:
        H, W = frame.shape[:2]
        s = min(1.0, 540 / H, 820 / W)
        disp = cv2.resize(frame, (int(W * s), int(H * s))) if s < 1 else frame.copy()
        if self.mirror:
            disp = disp[:, ::-1].copy()
        dh, dw = disp.shape[:2]
        ch = max(dh, MIN_HEIGHT)
        img = np.full((ch, dw + PANEL_W, 3), BG, np.uint8)
        img[:dh, :dw] = disp
        cv = Canvas(img)

        def mapbox(b):
            x1, y1, x2, y2 = (v * s for v in b)
            if self.mirror:
                x1, x2 = W * s - x2, W * s - x1
            return int(x1), int(y1), int(x2), int(y2)

        views = self.views
        for v in views:
            self._draw_face(img, cv, v, mapbox(v.box), dw, dh)

        self._draw_panel(img, cv, views, dw, ch, fps)
        self._draw_banners(img, cv, dw, dh)
        if self.show_analytics:
            self._draw_analytics(img, cv, dw, dh)
        if self.show_help:
            self._draw_help(img, cv, dw, dh)
        self._draw_modal(img, cv, dw, dh)
        cv.flush()
        self.last_canvas = img
        return img

    def _face_state(self, v: FaceView):
        if not v.is_live:
            return RED, "SPOOF ATTACK"
        if v.name != "Unknown":
            return GREEN, v.name
        return AMBER, "UNREGISTERED"

    def _draw_face(self, img, cv, v: FaceView, box, dw: int, dh: int):
        color, label = self._face_state(v)
        x1, y1, x2, y2 = box
        draw_brackets(img, box, color)

        size, h = 14, 24
        # liveness badge: REAL (green) / SPOOF (red); grey when the gate is off or missing
        if not self.liveness_active:
            live_txt, live_col = "UNCHECKED", (120, 120, 128)
        else:
            live_txt, live_col = ("REAL", GREEN) if v.is_live else ("SPOOF", RED)
        if v.emo == "N/A":
            emo_txt = v.emo_note or "no emotion"
        elif v.emo == "Uncertain":
            emo_txt = "Uncertain"
        else:
            emo_txt = f"{v.emo} {v.emo_conf * 100:.0f}%"
        w_name = text_width(label, size) + 14
        w_live = text_width(live_txt, size) + 14
        w_emo = text_width(emo_txt, size) + 40
        gap = 4
        total = w_name + w_live + w_emo + 2 * gap
        x = min(x1, max(0, dw - total))
        y = max(0, y1 - h - 4)

        cv2.rectangle(img, (x, y), (x + w_name, y + h), color, -1)
        cv.text(label, (x + 7, y + 4), size, DARK)
        x += w_name + gap
        cv2.rectangle(img, (x, y), (x + w_live, y + h), live_col, -1)
        cv.text(live_txt, (x + 7, y + 4), size, DARK)
        x += w_live + gap
        cv2.rectangle(img, (x, y), (x + w_emo, y + h), CHIP, -1)
        cv2.rectangle(img, (x, y), (x + w_emo, y + h), EMOTION_COLORS.get(v.emo.lower(), MUTED), 1)
        draw_emotion_icon(img, (x + 14, y + h // 2), 10, v.emo if v.emo != "N/A" else "uncertain")
        cv.text(emo_txt, (x + 30, y + 4), size, TEXT)

        ty = min(y2 + 4, dh - 34)
        if self.show_coords:
            ox1, oy1, ox2, oy2 = v.box
            cv.text(f"box ({ox1},{oy1}) - ({ox2},{oy2})", (x1, ty), 11, color)
            ty += 14
        if v.is_live and v.name == "Unknown":
            cv.text("Not enrolled - press R or click Register", (x1, ty), 12, AMBER)

    def _bar(self, img, x, y, w, h, frac, color, tick=None):
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 68), -1)
        cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), color, -1)
        if tick is not None:
            tx = x + int(w * tick)
            cv2.line(img, (tx, y - 3), (tx, y + h + 3), (255, 255, 255), 1)

    def _button(self, img, cv, key, rect, label, enabled=True, fill=ACCENT):
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), fill if enabled else (70, 70, 78), -1)
        cv.text(label, ((x1 + x2) // 2, (y1 + y2) // 2 - 8), 14, (255, 255, 255) if enabled else MUTED, "ma")
        self.buttons[key] = rect

    def _draw_panel(self, img, cv, views: List[FaceView], x0: int, ch: int, fps: float):
        pad = 16
        x = x0 + pad
        inner = PANEL_W - 2 * pad
        cv2.rectangle(img, (x0, 0), (x0 + PANEL_W, ch), PANEL, -1)
        cv.text("Attendance System", (x, 12), 20)
        cv.text("COS30082 · face · liveness · emotion", (x, 38), 12, MUTED)

        if self.liveness_model is None:
            live_txt, live_col = "Liveness: MODEL MISSING", RED
        elif not self.enforce_liveness:
            live_txt, live_col = "Liveness gate: OFF (bypass)", AMBER
        else:
            live_txt, live_col = "Liveness gate: ON", GREEN
        emo_txt, emo_col = ("Emotion model: OK", GREEN) if self.emotion_model else ("Emotion: MODEL MISSING", RED)
        y = 62
        for txt, col in ((live_txt, live_col), (emo_txt, emo_col)):
            cv2.circle(img, (x + 5, y + 8), 5, col, -1, cv2.LINE_AA)
            cv.text(txt, (x + 18, y), 13)
            y += 20
        n_enrolled = len(self.registry.list_identities())
        cv.text(f"Metric: {self.metric.upper()}  ·  Enrolled: {n_enrolled}", (x, y), 13, MUTED)
        cv.text(f"Video {fps:4.1f} FPS", (x, y + 18), 13, GREEN if fps >= TARGET_FPS else RED)
        cv.text(f"AI {self.ai_hz:3.1f}/s · {self.ai_ms:3.0f} ms", (x + 110, y + 18), 13, MUTED)
        y += 48

        cv2.line(img, (x, y), (x + inner, y), (70, 70, 78), 1)
        y += 10
        if not views:
            cv.text("No face detected", (x, y + 8), 16, MUTED)
            y += 40
        else:
            v = max(views, key=lambda z: z.area)
            color, _ = self._face_state(v)
            if not v.is_live:
                title, sub, subcol = "SPOOF ATTACK", "Access blocked: fake face", RED
            elif v.name == "Unknown":
                title, sub, subcol = "Unregistered", "Not enrolled · press R or click Register", MUTED
            else:
                title, sub, subcol = v.name, f"Similarity {v.conf:.2f} ({self.metric})", MUTED
            cv.text(title, (x, y), 24, color)
            cv.text(sub, (x, y + 30), 12, subcol)
            y += 56

            cv.text("Liveness", (x, y), 13, MUTED)
            cv.text("REAL" if v.is_live else "SPOOF", (x + inner, y), 13, GREEN if v.is_live else RED, "ra")
            self._bar(img, x, y + 22, inner, 10, v.live_score, GREEN if v.is_live else RED, self.liveness_thresh)
            y += 46

            cv.text("Emotion", (x, y), 13, MUTED)
            y += 20
            draw_emotion_icon(img, (x + 34, y + 34), 32, v.emo if v.emo != "N/A" else "uncertain")
            cv.text(v.emo, (x + 82, y + 14), 22)
            if v.emo == "N/A":
                sub_emo = v.emo_note or "no emotion yet"
            elif v.emo == "Uncertain":
                sub_emo = f"low confidence ({v.emo_conf * 100:.0f}%)"
            else:
                sub_emo = f"{v.emo_conf * 100:.0f}% confidence"
            cv.text(sub_emo, (x + 82, y + 44), 12, MUTED)
            y += 80
            for c in self.emotions:
                p = v.dist.get(c, v.dist.get(c.lower(), 0.0))
                cv.text(c, (x, y - 2), 11, MUTED)
                self._bar(img, x + 66, y + 1, inner - 100, 8, p, EMOTION_COLORS.get(c.lower(), MUTED))
                cv.text(f"{p * 100:2.0f}%", (x + inner, y - 2), 11, MUTED, "ra")
                y += 16
            y += 6

        cv2.line(img, (x, y), (x + inner, y), (70, 70, 78), 1)
        cv.text("Recent check-ins", (x, y + 8), 13, MUTED)
        yy = y + 30
        for line in list(self.recent)[:5]:
            if yy > ch - 112:
                break
            cv.text(line, (x, yy), 11)
            yy += 15

        # clickable buttons
        idle = self.mode == "idle"
        self._button(img, cv, "register", (x, ch - 102, x + inner, ch - 72), "Register New Employee", idle)
        half = (inner - 8) // 2
        self._button(img, cv, "analytics", (x, ch - 66, x + half, ch - 42), "Analytics (A)", True, (90, 90, 100))
        self._button(img, cv, "help", (x + half + 8, ch - 66, x + inner, ch - 42), "Help (H)", True, (90, 90, 100))
        cv.text("R register  S liveness  T metric  B coords", (x, ch - 36), 11, MUTED)
        cv.text("P snapshot  C camera  Q quit", (x, ch - 20), 11, MUTED)

    def _draw_banners(self, img, cv, dw: int, dh: int):
        with self.lock:
            idle = self.mode == "idle"
            msg, until = self.msg, self.msg_until
        if idle and msg and time.time() < until:
            shade(img, (0, dh - 36, dw, dh), (30, 30, 30), 0.8)
            ok = not msg.lower().startswith(("registration failed", "no face", "name"))
            cv.text(msg, (14, dh - 29), 16, GREEN if ok else AMBER)

    def _draw_modal(self, img, cv, dw: int, dh: int):
        with self.lock:
            mode, typed, name = self.mode, self.typed, self.enroll_name
            n = len(self.samples)
        if mode == "idle":
            return
        shade(img, (0, 0, dw, dh), (10, 10, 12), 0.55)
        mw, mh = min(460, dw - 40), 190
        x1, y1 = (dw - mw) // 2, (dh - mh) // 2
        x2, y2 = x1 + mw, y1 + mh
        cv2.rectangle(img, (x1, y1), (x2, y2), (44, 44, 52), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), ACCENT, 2)
        cv.text("Register New Employee", (x1 + 20, y1 + 14), 20)
        if mode == "typing":
            cv.text("Employee name / ID", (x1 + 20, y1 + 54), 13, MUTED)
            cv2.rectangle(img, (x1 + 20, y1 + 76), (x2 - 20, y1 + 110), (26, 26, 30), -1)
            cv2.rectangle(img, (x1 + 20, y1 + 76), (x2 - 20, y1 + 110), (110, 110, 120), 1)
            cv.text(typed + "_", (x1 + 30, y1 + 83), 18, AMBER)
            cv.text("Enter = start capture     Esc = cancel", (x1 + 20, y1 + 126), 13, MUTED)
            cv.text("Face the camera. Only live faces are accepted.", (x1 + 20, y1 + 150), 13, MUTED)
        elif mode == "capture":
            views = self.views
            if not views:
                hint, hc = "No face detected", AMBER
            elif not any(v.is_live for v in views):
                hint, hc = "Live face required (spoof detected)", RED
            else:
                hint, hc = "Keep facing the camera, vary your angle slightly", GREEN
            cv.text(f"Capturing reference frames for '{name}'", (x1 + 20, y1 + 54), 14)
            self._bar(img, x1 + 20, y1 + 88, mw - 40, 14, n / self.target_samples, ACCENT)
            cv.text(f"{n} / {self.target_samples} samples", (x1 + 20, y1 + 112), 13, MUTED)
            cv.text(hint, (x1 + 20, y1 + 138), 14, hc)
            cv.text("Esc = cancel", (x1 + 20, y1 + 160), 13, MUTED)
        else:  # saving
            cv.text(f"Extracting embeddings and saving '{name}' ...", (x1 + 20, y1 + 70), 14)
            self._bar(img, x1 + 20, y1 + 104, mw - 40, 14, 1.0, ACCENT)

    def _draw_analytics(self, img, cv, dw: int, dh: int):
        if time.time() - self.analytics_t > 5:
            self.analytics, self.analytics_t = self.logger.get_daily_analytics(), time.time()
        a = self.analytics
        shade(img, (20, 20, dw - 20, dh - 20), (20, 20, 24), 0.88)
        cv.text("Daily attendance & workplace sentiment", (40, 34), 20)
        names = a.get("verified_employees", a.get("verified_present", []))
        rows = [
            ("Date", a.get("date", "-")),
            ("Scan events", a.get("total_events", 0)),
            ("Unique employees", a.get("unique_employees_count", a.get("unique_employees", 0))),
            ("Verified present", ", ".join(names) if names else "None"),
            ("Spoof attacks blocked", a.get("spoof_attacks_blocked", a.get("spoof_attempts", 0))),
            ("Mood index", a.get("workplace_sentiment_index", "N/A")),
            ("Mood breakdown", a.get("mood_distribution", {})),
        ]
        y = 74
        for k, v in rows:
            cv.text(k, (40, y), 14, MUTED)
            cv.text(str(v), (230, y), 14)
            y += 28
        cv.text("Press A to close", (40, dh - 46), 12, MUTED)

    def _draw_help(self, img, cv, dw: int, dh: int):
        shade(img, (20, 20, dw - 20, dh - 20), (20, 20, 24), 0.9)
        cv.text("Controls", (40, 34), 20)
        items = [
            ("R / button", "Register new employee (type name / ID, live face required)"),
            ("A", "Daily attendance & mood analytics"),
            ("B", "Show / hide bounding-box coordinates"),
            ("S", "Toggle liveness gate (testing only)"),
            ("T", "Toggle cosine / euclidean matching"),
            ("L", "List enrolled employees in the terminal"),
            ("P", "Save snapshot to outputs/screenshots"),
            ("C / 0 / 1", "Switch camera"),
            ("H", "Close this help"),
            ("Q", "Quit"),
        ]
        y = 74
        for k, v in items:
            cv.text(k, (40, y), 14, AMBER)
            cv.text(v, (150, y), 14)
            y += 28

    # ------------------------------------------------------------------ input (main thread)
    def begin_registration(self):
        with self.lock:
            if self.mode != "idle":
                return
        if not self.views:
            self._flash("No face detected - position your face in front of the camera")
            return
        if self.terminal_name:
            try:
                name = input("Enter employee name / ID to register: ").strip()
            except Exception:
                name = ""
            if name and name.lower() != "unknown":
                self._start_capture(name)
            return
        with self.lock:
            self.mode, self.typed = "typing", ""

    def _start_capture(self, name: str):
        with self.lock:
            self.enroll_name, self.samples, self.last_sample_t = name, [], 0.0
            self.mode = "capture"

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for key, (x1, y1, x2, y2) in self.buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "register":
                    self.begin_registration()
                elif key == "analytics":
                    self.show_analytics, self.show_help = not self.show_analytics, False
                    self.analytics_t = 0.0
                elif key == "help":
                    self.show_help, self.show_analytics = not self.show_help, False
                return

    def handle_key(self, key: int):
        """Returns (quit, switch_camera_to or None)."""
        if key == 255:
            return False, None
        with self.lock:
            mode = self.mode
        if mode == "typing":
            with self.lock:
                if key in (13, 10):
                    name = self.typed.strip()
                    if not name or name.lower() == "unknown":
                        self.msg, self.msg_until = "Name must not be empty or 'Unknown'", time.time() + 3
                    else:
                        self.enroll_name, self.samples, self.last_sample_t = name, [], 0.0
                        self.mode = "capture"
                elif key == 27:
                    self.mode, self.typed = "idle", ""
                elif key in (8, 127):
                    self.typed = self.typed[:-1]
                elif 32 <= key <= 126 and len(self.typed) < 24:
                    self.typed += chr(key)
            return False, None
        if mode in ("capture", "saving"):
            if key == 27 and mode == "capture":
                with self.lock:
                    self.mode, self.samples = "idle", []
                self._flash("Registration cancelled")
            elif key in (ord("q"), ord("Q")):
                return True, None
            return False, None

        k = chr(key).lower() if 0 <= key < 128 else ""
        if k == "q":
            return True, None
        if k in ("0", "1") and int(k) != self.camera_id:
            return False, int(k)
        if k == "c":
            return False, 1 if self.camera_id == 0 else 0
        if k == "r":
            self.begin_registration()
        elif k == "a":
            self.show_analytics, self.show_help = not self.show_analytics, False
            self.analytics_t = 0.0
        elif k == "h":
            self.show_help, self.show_analytics = not self.show_help, False
        elif k == "b":
            self.show_coords = not self.show_coords
        elif k == "s":
            self.enforce_liveness = not self.enforce_liveness
            self._flash(f"Liveness gate {'ON' if self.enforce_liveness else 'OFF (testing)'}")
        elif k == "t":
            self.metric = "euclidean" if self.metric == "cosine" else "cosine"
            self._flash(f"Matching metric: {self.metric.upper()}")
        elif k == "l":
            print("\n--- ENROLLED EMPLOYEES ---")
            for item in self.registry.list_identities():
                print(f" • {item['name']:20} | Samples: {item['num_samples']} | Date: {item['created_at']}")
            self._flash("Enrolled list printed in terminal")
        elif k == "p" and self.last_canvas is not None:
            out = PROJECT_ROOT / "outputs" / "screenshots"
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(path), self.last_canvas)
            self._flash(f"Saved {path.name}")
        return False, None

    # ------------------------------------------------------------------ main loop
    def _start_grabber(self, cap: cv2.VideoCapture):
        self.grabber = FrameGrabber(cap)
        self.grabber.start()

    def _stop_grabber(self):
        if self.grabber is not None:
            self.grabber.stop()
            self.grabber.cap.release()
            self.grabber = None

    def run(self):
        cap, self.camera_id = self._open_camera(self.camera_id)
        if cap is None:
            print("[!] Unable to open a webcam (index 0 or 1). Check permissions / connection.")
            return
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self._start_grabber(cap)
        worker = threading.Thread(target=self._analysis_loop, daemon=True)
        worker.start()
        print("[*] Running (camera + AI in separate threads). Press H in the window for controls.")

        last_idx, fps, t_prev = -1, 0.0, time.time()
        try:
            while True:
                grabber = self.grabber
                if grabber is None or grabber.failed:
                    print(f"[!] Lost camera {self.camera_id}, reconnecting...")
                    self._stop_grabber()
                    time.sleep(0.5)
                    cap, self.camera_id = self._open_camera(self.camera_id)
                    if cap is None:
                        break
                    self._start_grabber(cap)
                    self._reset, last_idx = True, -1
                    continue

                frame, idx = grabber.latest()
                if frame is not None and idx != last_idx:
                    last_idx = idx
                    now = time.time()
                    fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-5)
                    t_prev = now
                    cv2.imshow(WINDOW, self.render(frame, fps))

                quit_flag, switch_to = self.handle_key(cv2.waitKey(1) & 0xFF)
                if quit_flag:
                    break
                if switch_to is not None:
                    self._stop_grabber()
                    new_cap, cam = self._open_camera(switch_to)
                    if new_cap is None:
                        new_cap, cam = self._open_camera(self.camera_id)
                    if new_cap is None:
                        break
                    self.camera_id = cam
                    self._start_grabber(new_cap)
                    self.views, self._reset, last_idx = [], True, -1
        finally:
            self._halt.set()
            worker.join(timeout=2.0)
            self._stop_grabber()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Face Recognition Attendance GUI.")
    parser.add_argument("--camera", type=int, default=0, choices=[0, 1])
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ai-fps", type=float, default=10.0,
                        help="Max analysis cycles per second (0 = as fast as possible). Video FPS is independent.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirrored (selfie) display")
    parser.add_argument("--terminal-name", action="store_true",
                        help="Type the employee name in the terminal (needed for accented characters)")
    parser.add_argument("--min-face-size", type=int, default=60,
                        help="Skip emotion when the face box is smaller than this many pixels")
    parser.add_argument("--min-det-conf", type=float, default=0.5,
                        help="Skip emotion when the detector confidence is below this (0 disables)")
    parser.add_argument("--emotion-conf-thresh", type=float, default=0.45,
                        help="Show 'Uncertain' when the smoothed emotion confidence is below this")
    parser.add_argument("--emotion-margin", type=float, default=0.25,
                        help="Expand the face box by this fraction for the emotion crop only (0 disables)")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation for emotion")
    parser.add_argument("--detector-backend", choices=["auto", "mtcnn", "mediapipe", "opencv"], default="auto",
                        help="Face detector backend (remove once the group-trained detector replaces it)")
    args = parser.parse_args()

    AttendanceApp(
        camera_id=args.camera, metric=args.metric, device=args.device, ai_fps=args.ai_fps,
        mirror=not args.no_mirror, terminal_name=args.terminal_name,
        min_face_size=args.min_face_size, min_det_conf=args.min_det_conf,
        emotion_conf_thresh=args.emotion_conf_thresh, emotion_margin=args.emotion_margin,
        use_tta=not args.no_tta, detector_backend=args.detector_backend,
    ).run()