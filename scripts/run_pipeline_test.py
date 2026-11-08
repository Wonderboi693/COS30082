import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import FaceAnalysisPipeline

pipeline = FaceAnalysisPipeline()

val_live = next(Path("data/raw/anti-spoofing_data/val/live").glob("*.*"))
val_spoof = next(Path("data/raw/anti-spoofing_data/val/spoof").glob("*.*"))

print("\n--- TEST: Physical Attack Sample ---")
res_spoof = pipeline.process_frame(val_spoof)
print(f"Granted: {res_spoof['access_granted']} | Reason: {res_spoof['rejection_reason']}")

print("\n--- TEST: Live User Sample ---")
res_live = pipeline.process_frame(val_live)
print(f"Granted:     {res_live['access_granted']}")
print(f"Liveness:    {res_live['liveness']['score']}")
print(f"Emotion:     {res_live['emotion']['dominant']} ({res_live['emotion']['confidence'] * 100:.1f}%)")
print(f"Latencies:   Emotion inference took {res_live['emotion']['latency_ms']} ms")