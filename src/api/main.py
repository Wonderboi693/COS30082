"""FastAPI backend for the COS30082 face attendance web app.

Run from the repository root:
    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
then open http://localhost:8000 (API docs: /docs).
"""
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import FaceEngine, decode_image
from .settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = FaceEngine(Settings())
    logging.getLogger("face_api").info("Engine ready: %s", app.state.engine.status())
    yield


app = FastAPI(title="COS30082 Face Attendance API", version="2.0.0", lifespan=lifespan)


def engine() -> FaceEngine:
    return app.state.engine


async def read_image(upload: UploadFile):
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"Empty file: {upload.filename}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (> 10 MB): {upload.filename}")
    try:
        return decode_image(data)
    except Exception:
        raise HTTPException(400, f"Not a valid image: {upload.filename}")


class ModelIn(BaseModel):
    key: str


class SettingsIn(BaseModel):
    threshold: Optional[float] = Field(None, ge=-1.0, le=1.0)
    liveness_enabled: Optional[bool] = None
    emotion_enabled: Optional[bool] = None


# --------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"status": "ok", **engine().status()}


@app.get("/api/models")
def list_models():
    return engine().models()


@app.put("/api/models/active")
def set_model(body: ModelIn):
    try:
        reembedded = engine().set_model(body.key)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"model": body.key, "threshold": engine().threshold, "reembedded_people": reembedded}


@app.put("/api/settings")
def update_settings(body: SettingsIn):
    e = engine()
    if body.threshold is not None:
        e.threshold = body.threshold
    if body.liveness_enabled is not None:
        if body.liveness_enabled and e.liveness is None:
            raise HTTPException(400, "Liveness checkpoint not available")
        e.liveness_enabled = body.liveness_enabled
    if body.emotion_enabled is not None:
        if body.emotion_enabled and e.emotion is None:
            raise HTTPException(400, "Emotion checkpoint not available")
        e.emotion_enabled = body.emotion_enabled
    return e.status()


@app.post("/api/recognize")
async def recognize(image: UploadFile = File(...), log_attendance: bool = Form(True), stream: bool = Form(False)):
    return engine().analyze(await read_image(image), log_attendance=log_attendance, stream=stream)


@app.post("/api/enroll")
async def enroll(name: str = Form(...), images: List[UploadFile] = File(...), person_id: Optional[str] = Form(None)):
    if not name.strip():
        raise HTTPException(400, "Name is required")
    person, skipped = engine().enroll(name, [await read_image(f) for f in images], person_id or None)
    if person is None:
        raise HTTPException(422, "No face found in any of the images")
    return {**person, "skipped_images": skipped}


@app.get("/api/people")
def people():
    return engine().gallery.list()


@app.delete("/api/people/{person_id}")
def delete_person(person_id: str):
    if not engine().gallery.delete(person_id):
        raise HTTPException(404, "Person not found")
    return {"deleted": person_id}


@app.get("/api/people/photo/{fname}")
def person_photo(fname: str):
    path = engine().gallery.photo_path(fname)
    if path is None:
        raise HTTPException(404, "Photo not found")
    return FileResponse(path)


@app.post("/api/verify")
async def verify(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    result = engine().verify(await read_image(image_a), await read_image(image_b))
    if result is None:
        raise HTTPException(422, "No face found in one of the images")
    return result


@app.get("/api/attendance")
def attendance(day: Optional[str] = None):
    try:
        return engine().attendance.read(day)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# --------------------------------------------------------------------- frontend
_web = Settings().frontend_dir
if _web.is_dir():
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
