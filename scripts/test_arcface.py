import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.verification.arcface.extractor import ArcFaceExtractor


def test_arcface_models(num_pairs=400):
    pairs_file = Path(Config.DATA_ROOT) / "verification_pairs_val.txt"
    lines = [l.split() for l in pairs_file.read_text().splitlines() if l.strip()][:num_pairs]

    for name, ckpt in [("Softmax", Config.SOFTMAX_CHECKPOINT), ("ArcFace", Config.ARCFACE_CHECKPOINT)]:
        if not Path(ckpt).is_file():
            print(f"[!] {name}: checkpoint not found at {ckpt}")
            continue
        print(f"[*] Loading {name} embedding model from {Path(ckpt).name}")
        extractor = ArcFaceExtractor(ckpt)

        # Warmup GPU
        for _ in range(5):
            _ = extractor.get_embedding(Path(Config.DATA_ROOT) / lines[0][0])

        start = time.perf_counter()
        emb_a = extractor.extract([Path(Config.DATA_ROOT) / a for a, _, _ in lines])
        emb_b = extractor.extract([Path(Config.DATA_ROOT) / b for _, b, _ in lines])
        per_image_ms = (time.perf_counter() - start) * 1000.0 / (2 * len(lines))

        labels = np.array([int(l) for _, _, l in lines])
        sims = (emb_a * emb_b).sum(axis=1)
        preds = sims >= extractor.threshold

        print("=" * 55)
        print(f"  {name.upper()} VERIFICATION CHECK ({len(lines)} val pairs)")
        print("=" * 55)
        print(f"Cosine AUC          : {roc_auc_score(labels, sims):.4f}")
        print(f"Accuracy @ {extractor.threshold:.4f} : {(preds == labels).mean() * 100:.2f}%")
        print(f"Same / diff mean cos: {sims[labels == 1].mean():.3f} / {sims[labels == 0].mean():.3f}")
        print(f"Embedding latency   : {per_image_ms:.2f} ms per image (batched)")
        print("=" * 55)


if __name__ == "__main__":
    test_arcface_models()
