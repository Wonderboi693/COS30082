import time
from pathlib import Path
import torch
from src.config import Config
from src.emotion.detector import EmotionDetector

def test_emotion_engine():
    print(f"[*] Initializing EmotionDetector on device: {Config.DEVICE}")
    detector = EmotionDetector(
        checkpoint_path="checkpoints/best_emotion.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    val_dir = Path("data/raw/emotion_data/val")
    sample_files = list(val_dir.rglob("*.jpg"))[:10]
    if not sample_files:
        sample_files = list(val_dir.rglob("*.png"))[:10]

    # Warmup GPU
    for _ in range(5):
        _ = detector.predict_emotion(sample_files[0])

    print("\n" + "=" * 55)
    print("  EMOTION RECOGNITION VALIDATION & LATENCY BENCHMARK")
    print("=" * 55)

    latencies = []
    for file_path in sample_files:
        ground_truth = file_path.parent.name
        res = detector.predict_emotion(file_path)
        latencies.append(res["latency_ms"])

        print(f"File: {file_path.name[:20]:20} | True: {ground_truth:8} -> "
              f"Pred: {res['dominant_emotion']:8} ({res['confidence'] * 100:5.1f}%) | "
              f"Latency: {res['latency_ms']:.2f} ms")

    avg_latency = sum(latencies) / len(latencies)
    print("=" * 55)
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print(f"Acceptance Target (< 30ms): {'[PASSED]' if avg_latency < 30.0 else '[FAILED]'}")
    print("=" * 55)

if __name__ == "__main__":
    test_emotion_engine()