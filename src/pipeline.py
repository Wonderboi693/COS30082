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
    Unified Facial Analysis Pipeline:
      1. Anti-Spoofing Gatekeeper (MSU-MFSD)
         -> Blocks physical attacks (print, screen, replay).
      2. Feature Extractor (Metric Learning / Triplet Loss)
         -> Computes unit L2-normalized 256-d biometric identity vector.
      3. Emotion Recognition Agent (RAF-DB)
         -> Predicts employee emotional state with latency < 30ms.
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
        print("[*] Initializing Pipeline Modules...")
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

    def process_frame(
        self,
        face_crop: Union[str, Path, np.ndarray, Image.Image],
        enrolled_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline on a cropped face frame.
        """
        # --- Stage 1: Anti-Spoofing Liveness Gate ---
        is_live, liveness_score = self.liveness_gate.check_liveness(face_crop)

        if not is_live:
            return {
                "access_granted": False,
                "rejection_reason": "SPOOF_ATTACK_DETECTED",
                "liveness": {
                    "passed": False,
                    "score": round(float(liveness_score), 4),
                    "threshold": self.liveness_threshold
                },
                "identity": None,
                "emotion": None
            }

        # --- Stage 2: Biometric Identity Extraction & Matching ---
        probe_embedding = self.extractor.get_embedding(face_crop)
        identity_result = {
            "embedding": probe_embedding,
            "verified": None,
            "similarity": None
        }

        access_granted = True
        rejection_reason = None

        if enrolled_embedding is not None:
            sim = float(np.dot(probe_embedding, enrolled_embedding))
            is_match = sim >= self.verification_threshold
            identity_result["similarity"] = round(sim, 4)
            identity_result["threshold"] = self.verification_threshold
            identity_result["verified"] = is_match
            access_granted = is_match
            if not is_match:
                rejection_reason = "IDENTITY_MISMATCH"

        # --- Stage 3: Emotion Recognition ---
        emotion_result = self.emotion_engine.predict_emotion(face_crop)

        return {
            "access_granted": access_granted,
            "rejection_reason": rejection_reason,
            "liveness": {
                "passed": True,
                "score": round(float(liveness_score), 4),
                "threshold": self.liveness_threshold
            },
            "identity": identity_result,
            "emotion": {
                "dominant": emotion_result["dominant_emotion"],
                "confidence": emotion_result["confidence"],
                "distribution": emotion_result["distribution"],
                "latency_ms": emotion_result["latency_ms"]
            }
        }