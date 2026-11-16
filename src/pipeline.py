import time
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image

from src.config import Config
from src.anti_spoofing.detector import LivenessDetector
from src.verification.extractor import FaceEmbeddingExtractor
from src.emotion.detector import EmotionDetector


class FaceAnalysisPipeline:
    """
    Unified Facial Analysis Pipeline (Story 2.2):
      1. Anti-Spoofing Gatekeeper (MSU-MFSD) -> Blocks presentation attacks immediately.
      2. Biometric Feature Extractor (Triplet Loss) -> Computes 256-d L2-normalized embedding.
      3. 1:N Identity Lookup / 1:1 Verification -> Matches against enrolled gallery.
      4. Emotion Recognition Agent (RAF-DB) -> Predicts expression state (< 30ms).
    """
    def __init__(
        self,
        antispoof_checkpoint: str = "checkpoints/best_antispoof.pt",
        verification_checkpoint: str = "checkpoints/best_metric.pth",
        emotion_checkpoint: str = "checkpoints/best_emotion.pt",
        liveness_threshold: float = 0.65,
        verification_threshold: float = 0.3332,
        device: str = "auto"
    ):
        print("[*] Initializing Unified FaceAnalysisPipeline...")
        self.liveness_gate = LivenessDetector(
            checkpoint_path=antispoof_checkpoint,
            threshold=liveness_threshold,
            device=device
        )
        self.liveness_threshold = liveness_threshold

        self.extractor = FaceEmbeddingExtractor(
            checkpoint_path=verification_checkpoint,
            device=device,
            normalize=True
        )
        self.verification_threshold = verification_threshold

        self.emotion_engine = EmotionDetector(
            checkpoint_path=emotion_checkpoint,
            device=device
        )

        # Pre-compile CUDA/cuDNN kernels to eliminate cold-start latency on frame 1
        self.warmup()

    def warmup(self) -> None:
        """Runs dummy passes through all three models to initialize CUDA/cuDNN kernels."""
        dummy_crop = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        for _ in range(2):
            _ = self.liveness_gate.check_liveness(dummy_crop)
            _ = self.extractor.get_embedding(dummy_crop)
            _ = self.emotion_engine.predict_emotion(dummy_crop)

    def process_frame(
        self,
        face_crop: Union[str, Path, np.ndarray, Image.Image],
        gallery: Optional[Any] = None,
        enrolled_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes the multi-modal pipeline on a cropped face frame.
        Supports 1:N lookup via `gallery` (requires `.identify(embedding, threshold)`)
        or 1:1 matching via `enrolled_embedding`.
        """
        t_start = time.perf_counter()

        # --- Stage 1: Anti-Spoofing Liveness Gatekeeper ---
        t_live_start = time.perf_counter()
        is_live, liveness_score = self.liveness_gate.check_liveness(face_crop)
        liveness_ms = (time.perf_counter() - t_live_start) * 1000.0

        if not is_live:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "access_granted": False,
                "rejection_reason": "SPOOF_ATTACK_DETECTED",
                "liveness": {
                    "passed": False,
                    "score": round(float(liveness_score), 4),
                    "threshold": self.liveness_threshold,
                    "latency_ms": round(liveness_ms, 2)
                },
                "identity": None,
                "emotion": None,
                "total_latency_ms": round(total_ms, 2)
            }

        # --- Stage 2: Biometric Identity Extraction & Matching ---
        t_id_start = time.perf_counter()
        probe_embedding = self.extractor.get_embedding(face_crop)
        identity_result: Dict[str, Any] = {
            "employee_id": None,
            "name": "Unknown",
            "embedding": probe_embedding,
            "verified": None,
            "similarity": None,
            "threshold": self.verification_threshold
        }

        access_granted = True
        rejection_reason = None

        if gallery is not None:
            # 1:N Gallery Search (IdentityGallery interface)
            match = gallery.identify(probe_embedding, threshold=self.verification_threshold)
            identity_result["employee_id"] = match.get("employee_id")
            identity_result["name"] = match.get("name", "Unknown")
            identity_result["similarity"] = match.get("similarity")
            identity_result["verified"] = match.get("verified", False)
            access_granted = bool(identity_result["verified"])
            if not access_granted:
                rejection_reason = "IDENTITY_UNRECOGNIZED"

        elif enrolled_embedding is not None:
            # 1:1 Template Verification
            sim = float(np.dot(probe_embedding, enrolled_embedding))
            is_match = sim >= self.verification_threshold
            identity_result["similarity"] = round(sim, 4)
            identity_result["verified"] = is_match
            access_granted = is_match
            if not is_match:
                rejection_reason = "IDENTITY_MISMATCH"

        id_ms = (time.perf_counter() - t_id_start) * 1000.0
        identity_result["latency_ms"] = round(id_ms, 2)

        # --- Stage 3: Workplace Emotion Recognition ---
        emotion_result = self.emotion_engine.predict_emotion(face_crop)
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "access_granted": access_granted,
            "rejection_reason": rejection_reason,
            "liveness": {
                "passed": True,
                "score": round(float(liveness_score), 4),
                "threshold": self.liveness_threshold,
                "latency_ms": round(liveness_ms, 2)
            },
            "identity": identity_result,
            "emotion": {
                "dominant": emotion_result["dominant_emotion"],
                "confidence": emotion_result["confidence"],
                "distribution": emotion_result["distribution"],
                "latency_ms": emotion_result["latency_ms"]
            },
            "total_latency_ms": round(total_ms, 2)
        }
