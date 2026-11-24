# Face Attendance Web App

Browser UI (`frontend/web/`) + FastAPI backend (`src/api/`) that runs every model in this repo:

| Part | Module | Checkpoint (in `checkpoints/`) |
|---|---|---|
| Face recognition (choose in the UI) | Triplet `src/verification/metric_learning` | `best_metric.pth` (default, val AUC 0.965) |
| | Softmax / ArcFace `src/verification/arcface` | `best_softmax.pt`, `best_arcface.pt` |
| | Classification `src/verification/classification` | `best.pt` (used if present; threshold not calibrated) |
| Anti-spoofing | `src/anti_spoofing` | `best_antispoof.pt` |
| Emotion | `src/emotion` | `best_emotion.pt` |
| Temporal smoothing (live camera) | `src/innovations/adaptive_fusion.py` | – |

Any missing checkpoint just switches that part off. Face detection uses OpenCV's Haar
cascade because `src/common/face_detector.py` is still empty.

## Run

```bash
pip install -r requirements.txt -r requirements-web.txt
# checkpoints: GitHub releases v0.2.0-checkpoints (group) and v0.3.0-checkpoints (softmax/ArcFace)
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 (the camera needs `localhost` or HTTPS; on WSL use the Windows
browser). API docs: http://localhost:8000/docs.

Tabs: **Live check-in** (webcam, model selector, threshold, liveness/emotion switches),
**Enroll** (3–5 captures or uploads), **Verify 1:1**, **People**, **Attendance**, **Models**.
Enrolled face crops are kept, so switching the model re-embeds them automatically.

Local data (git-ignored): `data/gallery/` (enrolled people), `outputs/attendance_logs/` (daily CSV).

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `FACE_MODEL` | best available | `triplet`, `softmax`, `arcface`, `classification` |
| `FACE_THRESHOLD` | model's validation EER threshold | cosine match threshold |
| `FACE_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `FACE_CHECKPOINT_DIR` | `checkpoints/` | where the checkpoints are |
| `FACE_ANTISPOOF_CHECKPOINT`, `FACE_EMOTION_CHECKPOINT` | `Config` paths | optional modules |
| `FACE_GALLERY_DIR`, `FACE_ATTENDANCE_DIR` | `data/gallery`, `outputs/attendance_logs` | local storage |
| `FACE_ATTENDANCE_COOLDOWN` | `60` | seconds before the same person is logged again |

Thresholds (cosine, EER point on `verification_pairs_val.txt`, 8,805 pairs): triplet 0.3333,
softmax 0.2007, ArcFace 0.2391. Webcam images differ from the 64×64 dataset crops; adjust the
threshold slider if needed.

## API

| Method | Path | Body |
|---|---|---|
| GET | `/api/health` | – |
| GET | `/api/models` | – |
| PUT | `/api/models/active` | `{"key": "softmax"}` |
| PUT | `/api/settings` | `{"threshold": 0.3, "liveness_enabled": true, "emotion_enabled": true}` |
| POST | `/api/recognize` | form: `image`, `log_attendance`, `stream` |
| POST | `/api/enroll` | form: `name`, `person_id?`, `images[]` |
| GET / DELETE | `/api/people`, `/api/people/{id}` | – |
| POST | `/api/verify` | form: `image_a`, `image_b` |
| GET | `/api/attendance?day=YYYY-MM-DD` | – |

## Test

```bash
python -m pytest tests/test_api.py -q      # needs checkpoints/best_softmax.pt (+ train_data for identity tests)
```
