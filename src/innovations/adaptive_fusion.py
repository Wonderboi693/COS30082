from collections import deque
from typing import Dict, Any, Optional
import numpy as np


class AdaptiveSecurityFusion:
    """
    Individual Innovation Module (Task 5.1.1 - Minh Huy):

      1. Temporal Liveness & Emotion Smoothing (EMA + sliding-window majority vote + stability gate)
         to eliminate single-frame webcam noise and false spoof lockouts.
      2. Confidence-Gated Template Adaptation: updates an enrolled user's
         256-d centroid only when both liveness and match confidence exceed
         strict high-security thresholds.
    """

    def __init__(
        self,
        window_size: int = 5,
        ema_alpha: float = 0.6,
        adaptation_lr: float = 0.05,
        min_liveness_for_update: float = 0.90,
        min_sim_for_update: float = 0.55,
        emotion_stability_frames: int = 3,
    ):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.adaptation_lr = adaptation_lr
        self.min_liveness_for_update = min_liveness_for_update
        self.min_sim_for_update = min_sim_for_update
        self.emotion_stability_frames = emotion_stability_frames

        self._liveness_history: deque = deque(maxlen=window_size)
        self._emotion_ema: Optional[Dict[str, float]] = None
        self._emotion_history: deque = deque(maxlen=window_size)
        self._pending_emotion: Optional[str] = None
        self._stability_counter: int = 0
        self._confirmed_emotion: Optional[str] = None

    def reset_session(self) -> None:
        """Clears temporal buffers when a subject leaves the camera frame."""
        self._liveness_history.clear()
        self._emotion_ema = None
        self._emotion_history.clear()
        self._pending_emotion = None
        self._stability_counter = 0
        self._confirmed_emotion = None

    def stabilize_liveness(self, raw_score: float, threshold: float = 0.65) -> Dict[str, Any]:
        """Computes temporally smoothed liveness confidence across recent frames."""
        self._liveness_history.append(float(raw_score))
        smoothed_score = float(np.mean(self._liveness_history))
        return {
            "raw_score": round(raw_score, 4),
            "smoothed_score": round(smoothed_score, 4),
            "passed": smoothed_score >= threshold,
            "frames_buffered": len(self._liveness_history),
        }

    def stabilize_emotion(self, distribution: Dict[str, float]) -> Dict[str, Any]:
        """
        Temporal emotion stabilization:
          - EMA over class probabilities (mượt độ xác suất)
          - Sliding-window majority vote over dominant labels (chống nháy)
          - Stability counter: emotion mới chỉ công nhận khi xuất hiện liên tiếp N frames

        Returns backward-compatible keys plus stability metadata.
        """
        # --- EMA over probabilities ---
        if self._emotion_ema is None or set(self._emotion_ema.keys()) != set(distribution.keys()):
            self._emotion_ema = dict(distribution)
        else:
            for cls_name, prob in distribution.items():
                self._emotion_ema[cls_name] = (
                    self.ema_alpha * prob + (1.0 - self.ema_alpha) * self._emotion_ema[cls_name]
                )

        # --- Sliding-window majority vote over dominant labels ---
        raw_dominant = max(distribution, key=distribution.get)
        self._emotion_history.append(raw_dominant)

        vote_counts: Dict[str, int] = {}
        for label in self._emotion_history:
            vote_counts[label] = vote_counts.get(label, 0) + 1
        voted_dominant = max(vote_counts, key=vote_counts.get)

        # --- Stability gate ---
        if voted_dominant == self._pending_emotion:
            self._stability_counter += 1
        else:
            self._pending_emotion = voted_dominant
            self._stability_counter = 1

        if self._stability_counter >= self.emotion_stability_frames:
            stable_dominant = voted_dominant
            self._confirmed_emotion = stable_dominant
        else:
            stable_dominant = self._confirmed_emotion or voted_dominant

        confidence = float(self._emotion_ema.get(stable_dominant, 0.0))
        is_stable = self._stability_counter >= self.emotion_stability_frames

        return {
            "dominant_emotion": stable_dominant,
            "confidence": round(confidence, 4),
            "smoothed_distribution": {k: round(v, 4) for k, v in self._emotion_ema.items()},
            "raw_dominant": raw_dominant,
            "voted_dominant": voted_dominant,
            "stability_frames": self._stability_counter,
            "is_stable": is_stable,
        }

    def compute_adapted_template(
        self,
        stored_embedding: np.ndarray,
        probe_embedding: np.ndarray,
        liveness_score: float,
        similarity: float,
    ) -> Optional[np.ndarray]:
        """
        Updates the enrolled L2-normalized centroid when high-confidence conditions are met.
        Returns the updated unit-norm vector, or None if update gates are not satisfied.
        """
        if liveness_score < self.min_liveness_for_update or similarity < self.min_sim_for_update:
            return None

        updated = (1.0 - self.adaptation_lr) * stored_embedding + self.adaptation_lr * probe_embedding
        norm = np.linalg.norm(updated)
        if norm < 1e-8:
            return None
        return (updated / norm).astype(np.float32)
