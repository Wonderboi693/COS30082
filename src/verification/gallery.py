import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch


class IdentityGallery:
    """
    1:N Employee Biometric Gallery (Task 4.2.1).
    Persists L2-normalized 256-d embeddings in `embeddings.pt` and employee records
    in `metadata.json`, performing vectorized cosine similarity search.
    """
    def __init__(self, gallery_dir: str = "data/gallery"):
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.gallery_dir / "embeddings.pt"
        self.meta_path = self.gallery_dir / "metadata.json"

        self.records: List[Dict[str, str]] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, 256)
        self.load()

    def load(self) -> None:
        if self.emb_path.exists() and self.meta_path.exists():
            tensor_embs = torch.load(self.emb_path, map_location="cpu", weights_only=True)
            self.embeddings = tensor_embs.numpy().astype(np.float32)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.embeddings = None
            self.records = []

    def save(self) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
        if self.embeddings is not None and len(self.records) > 0:
            torch.save(torch.from_numpy(self.embeddings), self.emb_path)
        elif self.emb_path.exists():
            self.emb_path.unlink()

    def enroll(self, employee_id: str, name: str, embedding: np.ndarray) -> None:
        """Enrolls a new employee or updates an existing employee's template."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            raise ValueError("Cannot enroll zero-norm embedding.")
        unit_emb = (embedding / norm).astype(np.float32)

        # Update existing record if employee_id already exists
        for idx, rec in enumerate(self.records):
            if rec["employee_id"] == employee_id:
                self.records[idx]["name"] = name
                self.embeddings[idx] = unit_emb
                self.save()
                return

        self.records.append({"employee_id": employee_id, "name": name})
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = unit_emb.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, unit_emb.reshape(1, -1)])
        self.save()

    def update_template(self, employee_id: str, new_embedding: np.ndarray) -> None:
        """Updates an enrolled employee's centroid (used by AdaptiveSecurityFusion)."""
        for idx, rec in enumerate(self.records):
            if rec["employee_id"] == employee_id:
                self.embeddings[idx] = new_embedding.astype(np.float32)
                self.save()
                return

    def identify(self, probe_embedding: np.ndarray, threshold: float = 0.3332) -> Dict[str, Any]:
        """Performs 1:N cosine similarity lookup against all enrolled employees."""
        if self.embeddings is None or len(self.records) == 0:
            return {
                "verified": False,
                "employee_id": None,
                "name": "Unknown",
                "similarity": 0.0
            }

        # Vectorized dot product (both gallery and probe are L2-normalized)
        sims = self.embeddings @ probe_embedding.astype(np.float32)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            matched = self.records[best_idx]
            return {
                "verified": True,
                "employee_id": matched["employee_id"],
                "name": matched["name"],
                "similarity": round(best_sim, 4)
            }

        return {
            "verified": False,
            "employee_id": None,
            "name": "Unknown",
            "similarity": round(best_sim, 4)
        }
