from pathlib import Path
from src.pipeline import FaceAnalysisPipeline

pipeline = FaceAnalysisPipeline(
    antispoof_checkpoint="checkpoints/best_antispoof.pt",
    verification_checkpoint="checkpoints/best_metric.pth",
    liveness_threshold=0.65,
    verification_threshold=0.3332
)

# Grab sample images
val_live = next(Path("data/raw/anti-spoofing_data/val/live").glob("*.*"))
val_spoof = next(Path("data/raw/anti-spoofing_data/val/spoof").glob("*.*"))

print("\n" + "=" * 50)
print("TEST 1: Presentation Attack Probe")
print("=" * 50)
res_spoof = pipeline.process_probe(val_spoof)
print(f"Access Granted:    {res_spoof['access_granted']}")
print(f"Rejection Reason:  {res_spoof['rejection_reason']}")
print(f"Liveness Score:    {res_spoof['liveness_score']} (Threshold: {res_spoof['liveness_threshold']})")

print("\n" + "=" * 50)
print("TEST 2: Bona Fide (Live) Probe Enrollment & Match")
print("=" * 50)
res_live = pipeline.process_probe(val_live)
print(f"Access Granted:    {res_live['access_granted']}")
print(f"Liveness Score:    {res_live['liveness_score']}")
print(f"Extracted Vector:  Shape {res_live['embedding'].shape}, Norm {round(float(sum(res_live['embedding']**2)), 2)}")

# 1:1 Matching against itself
res_match = pipeline.process_probe(val_live, enrolled_embedding=res_live["embedding"])
print(f"Self-Match Score:  {res_match['similarity']} (Verified: {res_match['verified']})")