"""API tests for the web backend (real checkpoints, temporary gallery).

Needs checkpoints/best_softmax.pt. Identity tests need face images: set
FACE_TEST_IMAGES to a folder of identity sub-folders (e.g. classification train_data);
defaults to the group's Config.CLASSIFICATION_DIR/train_data.
Run: python -m pytest tests/test_api.py -q
"""
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402

CKPT = ROOT / "checkpoints" / "best_softmax.pt"
IMAGES = Path(os.environ.get("FACE_TEST_IMAGES", Path(Config.CLASSIFICATION_DIR) / "train_data"))

pytestmark = pytest.mark.skipif(not CKPT.is_file(), reason="checkpoints/best_softmax.pt missing")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    os.environ.update({
        "FACE_GALLERY_DIR": str(tmp / "gallery"), "FACE_ATTENDANCE_DIR": str(tmp / "attendance"),
        "FACE_DEVICE": "cpu", "FACE_MODEL": "softmax",
        # Identity flow only: 64x64 dataset crops are rejected by the liveness model.
        "FACE_ANTISPOOF_CHECKPOINT": str(tmp / "none.pt"), "FACE_EMOTION_CHECKPOINT": str(tmp / "none.pt"),
    })
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def jpg(src):
    buf = io.BytesIO()
    img = Image.open(src) if isinstance(src, Path) else Image.fromarray(src)
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def identity_images(n_people=2, per_person=4):
    if not IMAGES.is_dir():
        pytest.skip(f"No test images at {IMAGES}")
    people = sorted(d for d in IMAGES.iterdir() if d.is_dir())[:n_people]
    return [sorted(p.glob("*.jpg"))[:per_person] for p in people]


def test_health_models_and_frontend(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model"] == "softmax" and h["embedding_dim"] == 512
    assert abs(h["threshold"] - 0.2007) < 1e-6
    assert h["liveness"]["available"] is False and h["emotion"]["available"] is False
    models = {m["key"]: m for m in client.get("/api/models").json()}
    assert set(models) == {"triplet", "softmax", "arcface", "classification"} and models["softmax"]["active"]
    assert client.get("/").status_code == 200


def test_arcface_extractor_unit_norm():
    from src.verification.arcface.extractor import ArcFaceExtractor
    e = ArcFaceExtractor(CKPT, device="cpu").extract([Image.new("RGB", (64, 64), "gray")] * 2)
    assert e.shape == (2, 512) and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)


def test_bad_inputs(client):
    assert client.post("/api/recognize", files={"image": ("x.jpg", b"not an image")}).status_code == 400
    blank = np.full((480, 640, 3), 255, np.uint8)   # large image without a face
    assert client.post("/api/recognize", files={"image": ("b.jpg", jpg(blank))}).json()["faces"] == []
    assert client.get("/api/attendance", params={"day": "../../etc"}).status_code == 400
    assert client.delete("/api/people/nobody").status_code == 404
    assert client.put("/api/models/active", json={"key": "nope"}).status_code == 400
    assert client.put("/api/settings", json={"liveness_enabled": True}).status_code == 400


def test_enroll_recognize_verify_switch_delete(client):
    (a1, a2, a3, a4), (b1, *_) = identity_images()
    r = client.post("/api/enroll", data={"name": "Person A", "person_id": "A1"},
                    files=[("images", (p.name, jpg(p))) for p in (a1, a2, a3)])
    assert r.status_code == 200, r.text
    assert r.json()["samples"] == 3

    def sims():
        same = client.post("/api/recognize", files={"image": ("a4.jpg", jpg(a4))}).json()["faces"][0]
        other = client.post("/api/recognize", files={"image": ("b1.jpg", jpg(b1))}).json()["faces"][0]
        return same, other

    same, other = sims()
    print(f"\nsoftmax  same={same['identity']['similarity']} other={other['identity']['similarity']}")
    assert same["identity"]["similarity"] > other["identity"]["similarity"]
    if same["access_granted"]:
        log = client.get("/api/attendance").json()
        assert log and log[0]["person_id"] == "A1" and log[0]["model"] == "softmax"

    v = client.post("/api/verify", files={"image_a": ("a.jpg", jpg(a1)), "image_b": ("b.jpg", jpg(a4))}).json()
    assert -1 <= v["similarity"] <= 1 and v["model"] == "softmax"

    # Switching the model re-embeds the saved photos (no re-enrolment needed).
    models = {m["key"]: m for m in client.get("/api/models").json()}
    for key in [k for k in ("triplet", "arcface") if models[k]["available"]]:
        s = client.put("/api/models/active", json={"key": key}).json()
        assert s["model"] == key and s["reembedded_people"] == 1
        same, other = sims()
        print(f"{key:8} same={same['identity']['similarity']} other={other['identity']['similarity']}")
        assert same["identity"]["similarity"] > other["identity"]["similarity"]
    client.put("/api/models/active", json={"key": "softmax"})

    people = client.get("/api/people").json()
    assert [p["id"] for p in people] == ["A1"]
    assert client.get(f"/api/people/photo/{people[0]['photo']}").status_code == 200
    assert client.delete("/api/people/A1").status_code == 200
    assert client.get("/api/people").json() == []
