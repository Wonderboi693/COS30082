"""Runtime settings for the web API, overridable with environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Config

PROJECT_ROOT = Path(Config.PROJECT_ROOT)
CHECKPOINT_DIR = Path(Config.CHECKPOINT_DIR)


def _path(name, default):
    return Path(os.environ.get(name, default)).expanduser()


@dataclass
class Settings:
    # Embedding model key (see src/api/models.py); empty -> best available
    model: str = field(default_factory=lambda: os.environ.get("FACE_MODEL", ""))
    # Empty -> the model's validation EER threshold
    threshold: str = field(default_factory=lambda: os.environ.get("FACE_THRESHOLD", ""))
    device: str = field(default_factory=lambda: os.environ.get("FACE_DEVICE", "auto"))
    checkpoint_dir: Path = field(default_factory=lambda: _path("FACE_CHECKPOINT_DIR", CHECKPOINT_DIR))

    # Optional group modules (used when their checkpoints exist)
    antispoof_checkpoint: Path = field(default_factory=lambda: _path(
        "FACE_ANTISPOOF_CHECKPOINT", Config.ANTISPOOF_CHECKPOINT))
    liveness_threshold: float = field(default_factory=lambda: float(os.environ.get("FACE_LIVENESS_THRESHOLD", "0.15")))
    emotion_checkpoint: Path = field(default_factory=lambda: _path(
        "FACE_EMOTION_CHECKPOINT", Config.EMOTION_CHECKPOINT))

    # Same locations the group's .gitignore already reserves
    gallery_dir: Path = field(default_factory=lambda: _path("FACE_GALLERY_DIR", PROJECT_ROOT / "data" / "gallery"))
    attendance_dir: Path = field(default_factory=lambda: _path(
        "FACE_ATTENDANCE_DIR", PROJECT_ROOT / "outputs" / "attendance_logs"))
    attendance_cooldown_s: float = field(default_factory=lambda: float(os.environ.get("FACE_ATTENDANCE_COOLDOWN", "60")))

    frontend_dir: Path = PROJECT_ROOT / "frontend" / "web"
