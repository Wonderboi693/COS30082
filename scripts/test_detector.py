from pathlib import Path
from src.anti_spoofing.detector import LivenessDetector

detector = LivenessDetector("checkpoints/best_antispoof.pt", threshold=0.65)

# Test first 3 validation samples from each category
val_dir = Path("data/raw/anti-spoofing_data/val")
live_samples = sorted(list((val_dir / "live").glob("*.*")))[:3]
spoof_samples = sorted(list((val_dir / "spoof").glob("*.*")))[:3]

print("--- BONA FIDE (LIVE) SAMPLES ---")
for p in live_samples:
    is_live, prob = detector.check_liveness(p)
    print(f"[{'PASS' if is_live else 'FAIL'}] {p.name:25} | P(Live): {prob:.4f}")

print("\n--- ATTACK (SPOOF) SAMPLES ---")
for p in spoof_samples:
    is_live, prob = detector.check_liveness(p)
    print(f"[{'BLOCKED' if not is_live else 'BREACH'}] {p.name:25} | P(Live): {prob:.4f}")