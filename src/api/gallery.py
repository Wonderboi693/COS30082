"""Enrolled identities and the attendance log, stored as local (git-ignored) files.

Each person keeps their enrolled face crops; embeddings are stored per model, so
switching the embedding model re-embeds the saved crops instead of re-enrolling.
"""
import csv
import json
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def _slug(text):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")
    return s[:40] or "person"


class Gallery:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.faces_dir = self.root / "enrolled_faces"
        self.file = self.root / "gallery.json"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._people = json.loads(self.file.read_text(encoding="utf-8"))["people"] if self.file.is_file() else []
        self._templates = {}   # model key -> (ids, matrix)

    def __len__(self):
        return len(self._people)

    def _save(self):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"people": self._people}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.file)
        self._templates.clear()

    def _template(self, model_key):
        """One template per person: normalised mean of their embeddings for this model."""
        if model_key not in self._templates:
            ids, rows = [], []
            for p in self._people:
                embs = p["embeddings"].get(model_key)
                if embs:
                    m = np.asarray(embs, dtype=np.float32).mean(axis=0)
                    ids.append(p["id"])
                    rows.append(m / (np.linalg.norm(m) + 1e-12))
            self._templates[model_key] = (ids, np.stack(rows) if rows else None)
        return self._templates[model_key]

    def list(self):
        with self._lock:
            return [{"id": p["id"], "name": p["name"], "samples": len(p["photos"]), "created": p["created"],
                     "photo": p["photos"][0] if p["photos"] else None, "models": sorted(p["embeddings"])}
                    for p in self._people]

    def enroll(self, name, crops, model_key, embeddings, person_id=None):
        person_id = _slug(person_id) if person_id else f"{_slug(name)}_{uuid.uuid4().hex[:6]}"
        with self._lock:
            person = next((p for p in self._people if p["id"] == person_id), None)
            if person is None:
                person = {"id": person_id, "name": name.strip(), "photos": [], "embeddings": {},
                          "created": datetime.now().isoformat(timespec="seconds")}
                self._people.append(person)
            # Other models' embeddings no longer cover all photos -> recomputed lazily
            person["embeddings"] = {model_key: person["embeddings"].get(model_key, [])}
            for emb, crop in zip(embeddings, crops):
                fname = f"{person_id}_{uuid.uuid4().hex[:8]}.jpg"
                crop.save(self.faces_dir / fname, quality=92)
                person["photos"].append(fname)
                person["embeddings"][model_key].append([round(float(v), 6) for v in emb])
            self._save()
            return {"id": person_id, "name": person["name"], "samples": len(person["photos"])}

    def ensure_model(self, model_key, extract_fn):
        """Embeds saved crops for people that have no embeddings for this model yet."""
        with self._lock:
            changed = 0
            for p in self._people:
                if len(p["embeddings"].get(model_key, [])) == len(p["photos"]):
                    continue
                crops = []
                for fname in p["photos"]:
                    with Image.open(self.faces_dir / fname) as img:
                        crops.append(img.convert("RGB"))
                p["embeddings"][model_key] = [[round(float(v), 6) for v in e] for e in extract_fn(crops)]
                changed += 1
            if changed:
                self._save()
            return changed

    def delete(self, person_id):
        with self._lock:
            person = next((p for p in self._people if p["id"] == person_id), None)
            if person is None:
                return False
            for fname in person["photos"]:
                (self.faces_dir / fname).unlink(missing_ok=True)
            self._people.remove(person)
            self._save()
            return True

    def photo_path(self, fname):
        path = self.faces_dir / Path(fname).name
        return path if path.is_file() else None

    def identify(self, embedding, model_key, threshold):
        """1:N search: best match, and whether it passes the threshold."""
        with self._lock:
            ids, matrix = self._template(model_key)
            if matrix is None:
                return {"id": None, "name": "Unknown", "closest": None, "similarity": None, "matched": False}
            sims = matrix @ np.asarray(embedding, dtype=np.float32)
            best = int(np.argmax(sims))
            sim = float(sims[best])
            person = next(p for p in self._people if p["id"] == ids[best])
            matched = sim >= threshold
            return {"id": person["id"] if matched else None, "name": person["name"] if matched else "Unknown",
                    "closest": person["name"], "similarity": round(sim, 4), "matched": matched}


class AttendanceLog:
    FIELDS = ["timestamp", "person_id", "name", "model", "similarity", "liveness_score", "emotion"]

    def __init__(self, root: Path, cooldown_s=60.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cooldown_s = cooldown_s
        self._last = {}
        self._lock = threading.Lock()

    def _file(self, day):
        return self.root / f"attendance_{day}.csv"

    def record(self, person_id, name, model, similarity, liveness_score=None, emotion=None):
        """Logs a check-in unless the same person was logged within the cooldown."""
        now = time.time()
        with self._lock:
            if now - self._last.get(person_id, 0) < self.cooldown_s:
                return False
            self._last[person_id] = now
            stamp = datetime.now()
            path = self._file(stamp.strftime("%Y-%m-%d"))
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS)
                if new:
                    w.writeheader()
                w.writerow({"timestamp": stamp.isoformat(timespec="seconds"), "person_id": person_id, "name": name,
                            "model": model, "similarity": similarity, "liveness_score": liveness_score,
                            "emotion": emotion})
            return True

    def read(self, day=None):
        day = day or datetime.now().strftime("%Y-%m-%d")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("day must be YYYY-MM-DD")
        path = self._file(day)
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
