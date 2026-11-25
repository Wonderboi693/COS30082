"""
Face Identity Registry module for enterprise face recognition.
Fulfills AGENT.md requirements:
  - Register new employees dynamically without retraining (§1.2, §7).
  - Store reference embeddings as mean of enrolled face crops (§12).
  - Match probe embeddings against enrolled gallery using Cosine/Euclidean metric.
  - Persist gallery embeddings and metadata to disk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.config import Config
from src.verification.extractor import FaceEmbeddingExtractor


class FaceRegistry:
    """
    Dynamic Employee Face Gallery & Matcher.
    Stores unit L2-normalized 256-d embeddings for enrolled personnel.
    """

    def __init__(
        self,
        gallery_dir: Union[str, Path] = "data/gallery",
        checkpoint_path: Optional[str] = None,
        threshold: float = 0.3332,
        metric: str = "cosine",
        device: str = "auto",
        auto_load: bool = True
    ):
        self.gallery_dir = Path(gallery_dir).resolve()
        self.faces_dir = self.gallery_dir / "enrolled_faces"
        self.db_path = self.gallery_dir / "registry.json"

        self.threshold = threshold
        self.metric = metric.lower()
        self.device = device
        self.checkpoint_path = checkpoint_path

        self._extractor: Optional[FaceEmbeddingExtractor] = None
        self._identities: Dict[str, Dict[str, Any]] = {}

        self.faces_dir.mkdir(parents=True, exist_ok=True)

        if auto_load and self.db_path.is_file():
            self.load()

    @property
    def extractor(self) -> FaceEmbeddingExtractor:
        if self._extractor is None:
            self._extractor = FaceEmbeddingExtractor(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                normalize=True
            )
        return self._extractor

    def register(
        self,
        name: str,
        images: Union[str, Path, np.ndarray, Image.Image, List[Union[str, Path, np.ndarray, Image.Image]]],
        metadata: Optional[Dict[str, Any]] = None,
        save_crops: bool = True
    ) -> np.ndarray:
        name = name.strip()
        if not name:
            raise ValueError("Employee name cannot be empty.")

        if not isinstance(images, list):
            image_list = [images]
        else:
            image_list = images

        if len(image_list) == 0:
            raise ValueError("At least one face image is required to register an identity.")

        embeddings = []
        user_crop_dir = self.faces_dir / name
        if save_crops:
            user_crop_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(image_list):
            emb = self.extractor.get_embedding(img)
            embeddings.append(emb)

            if save_crops:
                pil_img = self._to_pil(img)
                timestamp = int(time.time() * 1000)
                crop_path = user_crop_dir / f"face_{timestamp}_{idx}.jpg"
                pil_img.save(crop_path, quality=95)

        mean_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 1e-12:
            mean_embedding = mean_embedding / norm
        else:
            mean_embedding = embeddings[0]

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "embedding": mean_embedding.astype(np.float32),
            "num_samples": len(image_list),
            "created_at": now,
            "metadata": metadata or {}
        }

        self._identities[name] = record
        self.save()
        print(f"[FaceRegistry] Successfully enrolled '{name}' with {len(image_list)} sample(s).")
        return mean_embedding

    def identify(
        self,
        probe: Union[str, Path, np.ndarray, Image.Image],
        threshold: Optional[float] = None,
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        thresh = threshold if threshold is not None else self.threshold
        used_metric = (metric or self.metric).lower()

        if isinstance(probe, np.ndarray) and probe.ndim == 1:
            probe_emb = probe
            if np.linalg.norm(probe_emb) > 0:
                probe_emb = probe_emb / np.linalg.norm(probe_emb)
        else:
            probe_emb = self.extractor.get_embedding(probe)

        if not self._identities:
            return {
                "name": "Unknown",
                "verified": False,
                "confidence": 0.0,
                "threshold": thresh,
                "metric": used_metric,
                "candidates": []
            }

        names = list(self._identities.keys())
        gallery_embs = np.stack([self._identities[n]["embedding"] for n in names], axis=0)

        if used_metric == "cosine":
            sims = np.dot(gallery_embs, probe_emb)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            is_match = best_score >= thresh
        elif used_metric == "euclidean":
            dists = np.linalg.norm(gallery_embs - probe_emb, axis=1)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            best_score = float(-best_dist)
            is_match = best_dist <= thresh
        else:
            raise ValueError(f"Unsupported metric: '{used_metric}'. Choose 'cosine' or 'euclidean'.")

        matched_name = names[best_idx] if is_match else "Unknown"

        candidates = [
            {"name": n, "score": round(float(s), 4)}
            for n, s in zip(names, sims if used_metric == "cosine" else -dists)
        ]
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return {
            "name": matched_name,
            "verified": is_match,
            "confidence": round(best_score, 4),
            "threshold": thresh,
            "metric": used_metric,
            "candidates": candidates[:5]
        }

    def verify(
        self,
        probe: Union[str, Path, np.ndarray, Image.Image],
        claimed_name: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        if claimed_name not in self._identities:
            return False, 0.0

        thresh = threshold if threshold is not None else self.threshold
        probe_emb = self.extractor.get_embedding(probe)
        ref_emb = self._identities[claimed_name]["embedding"]

        sim = float(np.dot(probe_emb, ref_emb))
        return bool(sim >= thresh), round(sim, 4)

    def remove(self, name: str) -> bool:
        if name in self._identities:
            del self._identities[name]
            self.save()
            print(f"[FaceRegistry] Removed identity: '{name}'")
            return True
        return False

    def list_identities(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "num_samples": data["num_samples"],
                "created_at": data.get("created_at", "N/A"),
                "metadata": data.get("metadata", {})
            }
            for name, data in self._identities.items()
        ]

    def clear(self) -> None:
        self._identities.clear()
        self.save()

    def save(self, filepath: Optional[Union[str, Path]] = None) -> None:
        target_path = Path(filepath or self.db_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for name, data in self._identities.items():
            serializable[name] = {
                "embedding": data["embedding"].tolist(),
                "num_samples": data["num_samples"],
                "created_at": data.get("created_at", ""),
                "metadata": data.get("metadata", {})
            }

        with target_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def load(self, filepath: Optional[Union[str, Path]] = None) -> None:
        source_path = Path(filepath or self.db_path)
        if not source_path.is_file():
            return

        with source_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self._identities = {}
        for name, data in raw.items():
            self._identities[name] = {
                "embedding": np.array(data["embedding"], dtype=np.float32),
                "num_samples": data.get("num_samples", 1),
                "created_at": data.get("created_at", ""),
                "metadata": data.get("metadata", {})
            }
        print(f"[FaceRegistry] Loaded {len(self._identities)} registered identity/identities from {source_path.name}")

    @staticmethod
    def _to_pil(img_input: Any) -> Image.Image:
        if isinstance(img_input, (str, Path)):
            with Image.open(img_input) as opened:
                return opened.convert("RGB")
        if isinstance(img_input, np.ndarray):
            if img_input.ndim == 3 and img_input.shape[2] == 3:
                img_input = img_input[..., ::-1]  # BGR to RGB
            return Image.fromarray(img_input.astype(np.uint8)).convert("RGB")
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        raise TypeError(f"Cannot convert {type(img_input)} to PIL Image.")


if __name__ == "__main__":
    registry = FaceRegistry(gallery_dir="data/gallery")
    print(f"Current enrolled count: {len(registry.list_identities())}")
