import os
import sys
import time
from pathlib import Path

# Automatically route OpenCV's bundled Qt GUI backend through XWayland on Wayland compositors
if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2

# Ensure project root is on sys.path when running `python frontend/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.face_detector import FaceDetector
from src.verification.gallery import IdentityGallery
from src.pipeline import FaceAnalysisPipeline
from src.innovations.adaptive_fusion import AdaptiveSecurityFusion


EMOTION_COLORS = {
    "Happy": (80, 220, 100),
    "Neutral": (220, 220, 220),
    "Surprise": (255, 200, 50),
    "Sad": (230, 140, 60),
    "Fear": (200, 90, 200),
    "Disgust": (60, 180, 140),
    "Angry": (60, 60, 240),
}


def draw_hud_box(
    frame,
    box,
    color,
    header_text: str,
    sub_text: str = ""
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Top label banner
    (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    banner_y1 = max(0, y1 - th - 12)
    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 12, y1), color, -1)
    cv2.putText(
        frame,
        header_text,
        (x1 + 6, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (15, 15, 15),
        2,
        cv2.LINE_AA
    )

    # Bottom status banner
    if sub_text:
        (sw, sh), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y2), (x1 + sw + 10, y2 + sh + 10), (25, 25, 25), -1)
        cv2.putText(
            frame,
            sub_text,
            (x1 + 5, y2 + sh + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA
        )


def run_live_app(camera_index: int = 0) -> None:
    detector = FaceDetector()
    gallery = IdentityGallery(gallery_dir="data/gallery")
    pipeline = FaceAnalysisPipeline()
    fusion = AdaptiveSecurityFusion(window_size=5)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    print("\n========================================================")
    print("  Smart Biometric Attendance & Emotion HUD (Live)")
    print(f"  Enrolled Employees in Gallery: {len(gallery.records)}")
    print("  Controls: [E] Enroll current face  |  [R] Reset buffers  |  [Q] Quit")
    print("========================================================\n")

    latest_live_crop = None
    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to grab webcam frame.")
                break

            # Mirror frame for natural webcam interaction
            frame = cv2.flip(frame, 1)
            faces = detector.detect_faces(frame)

            if not faces:
                fusion.reset_session()
                latest_live_crop = None
                cv2.putText(
                    frame,
                    "No face detected — step into view",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA
                )
            else:
                # Process primary (largest) detected face; pass full_frame for scene-level liveness
                x1, y1, x2, y2, face_crop = faces[0]
                result = pipeline.process_frame(face_crop, gallery=gallery, full_frame=frame)

                raw_live_score = result["liveness"]["score"]
                smooth_live = fusion.stabilize_liveness(
                    raw_live_score, threshold=pipeline.liveness_threshold
                )

                # Guard against pipeline short-circuit (identity/emotion are None when liveness fails)
                if not result["liveness"]["passed"] or not smooth_live["passed"] or result["emotion"] is None:
                    # RED: Spoof Attack Blocked
                    latest_live_crop = None
                    draw_hud_box(
                        frame,
                        (x1, y1, x2, y2),
                        color=(40, 40, 240),
                        header_text=f"SPOOF BLOCKED ({smooth_live['smoothed_score']:.2f})",
                        sub_text=f"Raw: {raw_live_score:.2f} | Pipeline: {result['total_latency_ms']:.1f} ms"
                    )
                else:
                    latest_live_crop = face_crop
                    identity = result["identity"]
                    emotion = result["emotion"]
                    smooth_emo = fusion.stabilize_emotion(emotion["distribution"])
                    emo_label = smooth_emo["dominant_emotion"]
                    emo_conf = smooth_emo["confidence"]

                    if identity["verified"]:
                        # GREEN: Verified Employee
                        emp_id = identity["employee_id"]
                        emp_name = identity["name"]
                        sim = identity["similarity"]

                        # Adaptive template maintenance (Task 5.1.1)
                        emp_idx = next(
                            i for i, r in enumerate(gallery.records) if r["employee_id"] == emp_id
                        )
                        adapted = fusion.compute_adapted_template(
                            stored_embedding=gallery.embeddings[emp_idx],
                            probe_embedding=identity["embedding"],
                            liveness_score=smooth_live["smoothed_score"],
                            similarity=sim
                        )
                        if adapted is not None:
                            gallery.update_template(emp_id, adapted)

                        draw_hud_box(
                            frame,
                            (x1, y1, x2, y2),
                            color=(60, 220, 80),
                            header_text=f"{emp_name} ({emp_id}) | sim={sim:.2f}",
                            sub_text=f"Emotion: {emo_label} ({emo_conf:.0%}) | Live: {smooth_live['smoothed_score']:.2f} | {result['total_latency_ms']:.1f}ms"
                        )
                    else:
                        # YELLOW: Live Face, Unknown Identity
                        sim = identity["similarity"] or 0.0
                        draw_hud_box(
                            frame,
                            (x1, y1, x2, y2),
                            color=(40, 215, 245),
                            header_text=f"UNKNOWN (best sim={sim:.2f}) — Press [E] to Enroll",
                            sub_text=f"Emotion: {emo_label} ({emo_conf:.0%}) | Live: {smooth_live['smoothed_score']:.2f} | {result['total_latency_ms']:.1f}ms"
                        )

            # Top status bar (FPS & Gallery count)
            now = time.perf_counter()
            fps = 1.0 / max(1e-5, now - prev_time)
            prev_time = now
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Enrolled: {len(gallery.records)} | [E] Enroll  [R] Reset  [Q] Quit",
                (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (235, 235, 235),
                1,
                cv2.LINE_AA
            )

            cv2.imshow("COS30082 — Biometric Attendance & Emotion HUD", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                fusion.reset_session()
                print("[*] Temporal fusion buffers reset.")
            elif key == ord("e"):
                if latest_live_crop is None:
                    print("[!] Enrollment blocked: No verified live face in frame.")
                else:
                    emp_id = input("\n[Enroll] Enter Employee ID (e.g., EMP001): ").strip()
                    emp_name = input("[Enroll] Enter Employee Name (e.g., Minh Huy): ").strip()
                    if emp_id and emp_name:
                        emb = pipeline.extractor.get_embedding(latest_live_crop)
                        gallery.enroll(emp_id, emp_name, emb)
                        print(f"[✓] Enrolled {emp_name} ({emp_id}) into 1:N gallery!\n")
                    # Drain stale frames buffered during blocking input() and reset temporal state
                    for _ in range(5):
                        cap.grab()
                    fusion.reset_session()
                    prev_time = time.perf_counter()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_app(camera_index=0)
