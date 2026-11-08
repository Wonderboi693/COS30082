from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from PIL import Image

from src.config import Config
from src.anti_spoofing.detector import LivenessDetector
from src.verification.extractor import FaceEmbeddingExtractor


class FaceAnalysisPipeline:
    """
    Unified Facial Verification Pipeline with Inline Anti-Spoof Gatekeeper.
    
    Workflow:
      1. Presentation Attack Detection (Liveness Gate)
         -> If spoof: Immediately abort with security rejection.
      2. Feature Extraction (Metric-Learning FaceNet Backbone)
         -> If live: Generate unit L2-normalized 256-d face embedding.
      3. Identity Verification / 1:1 Matching
         -> Compare cosine similarity against registered gallery template.
    """
    def __init__(
        self,
        antispoof_checkpoint: str = "checkpoints/best_antispoof.pt",
        verification_checkpoint: str = "checkpoints/best_metric.pth",
        liveness_threshold: float = 0.65,
        verification_threshold: float = 0.3332,
        device: str = "auto"
    ):
        print("[*] Initializing Liveness Gatekeeper...")
        self.liveness_gate = LivenessDetector(
            checkpoint_path=antispoof_checkpoint,
            threshold=liveness_threshold,
            device=device
        )
        self.liveness_threshold = liveness_threshold

        print("[*] Initializing Face Embedding Extractor...")
        self.extractor = FaceEmbeddingExtractor(
            checkpoint_path=verification_checkpoint,
            device=device,
            normalize=True
        )
        self.verification_threshold = verification_threshold

    def process_probe(
        self,
        probe_image: Union[str, Path, np.ndarray, Image.Image],
        enrolled_embedding: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Processes an incoming face probe through the secure verification gate.
        """
        # Step 1: Presentation Attack Gatekeeper
        is_live, liveness_score = self.liveness_gate.check_liveness(probe_image)

        if not is_live:
            return {
                "access_granted": False,
                "rejection_reason": "SPOOF_ATTACK_DETECTED",
                "liveness_passed": False,
                "liveness_score": round(float(liveness_score), 4),
                "liveness_threshold": self.liveness_threshold,
                "similarity": None,
                "verified": False,
                "embedding": None
            }

        # Step 2: Extract Biometric Embedding
        probe_embedding = self.extractor.get_embedding(probe_image)

        result = {
            "access_granted": True,
            "rejection_reason": None,
            "liveness_passed": True,
            "liveness_score": round(float(liveness_score), 4),
            "liveness_threshold": self.liveness_threshold,
            "embedding": probe_embedding,
            "similarity": None,
            "verified": None
        }

        # Step 3: Optional 1:1 Identity Verification Matching
        if enrolled_embedding is not None:
            # Cosine similarity for unit vectors: dot product
            sim = float(np.dot(probe_embedding, enrolled_embedding))
            is_match = sim >= self.verification_threshold

            result["similarity"] = round(sim, 4)
            result["verification_threshold"] = self.verification_threshold
            result["verified"] = is_match
            result["access_granted"] = is_match

            if not is_match:
                result["rejection_reason"] = "IDENTITY_MISMATCH"

        return result