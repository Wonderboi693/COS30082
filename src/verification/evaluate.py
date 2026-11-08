import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import torch

from src.config import Config
from src.verification.extractor import FaceEmbeddingExtractor


def compute_metrics(similarities: np.ndarray, labels: np.ndarray):
    """Compute ROC-AUC, Equal Error Rate (EER), optimal threshold, and accuracy."""
    auc = roc_auc_score(labels, similarities)
    fpr, tpr, thresholds = roc_curve(labels, similarities)

    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    optimal_threshold = float(thresholds[eer_idx])

    # Evaluate accuracy across thresholds to find the maximum
    preds = similarities[:, None] >= thresholds[None, :]
    accuracies = (preds == labels[:, None]).mean(axis=0)
    best_acc = float(accuracies.max())

    return {
        "roc_auc": float(auc),
        "eer": eer,
        "optimal_threshold": optimal_threshold,
        "best_accuracy": best_acc
    }


def parse_pairs_file(file_path: Path, data_root: Path):
    """
    Parses space-delimited or comma-delimited pair files.
    Resolves image paths against data_root / verification_data if needed.
    """
    pairs = []
    text = file_path.read_text(encoding="utf-8").strip().splitlines()

    # Detect delimiter
    first_line = text[0].strip()
    delimiter = "," if "," in first_line else None

    # Check if header exists
    start_idx = 1 if ("img" in first_line.lower() or "image" in first_line.lower()) else 0

    for line in text[start_idx:]:
        parts = line.strip().split(delimiter) if delimiter else line.strip().split()
        if len(parts) < 2:
            continue

        # Format: img1, img2, label OR id, img1, img2, label
        if len(parts) == 3:
            p1_raw, p2_raw, lbl = parts[0], parts[1], parts[2]
        elif len(parts) == 4:
            p1_raw, p2_raw, lbl = parts[1], parts[2], parts[3]
        else:
            p1_raw, p2_raw = parts[0], parts[1]
            lbl = None

        # Resolve paths
        def resolve_img(raw_str):
            p = Path(raw_str)
            if p.is_file():
                return p.resolve()
            candidate1 = data_root / raw_str
            if candidate1.is_file():
                return candidate1.resolve()
            candidate2 = data_root / "verification_data" / p.name
            if candidate2.is_file():
                return candidate2.resolve()
            raise FileNotFoundError(f"Cannot locate evaluation image: {raw_str}")

        pairs.append((resolve_img(p1_raw), resolve_img(p2_raw), int(lbl) if lbl is not None else None))

    return pairs


def run_benchmark(checkpoint_path: str, pairs_file: str, metric: str = "cosine", batch_size: int = 64):
    data_root = Path(Config.DATA_ROOT).resolve()
    pairs_path = Path(pairs_file).resolve()

    print(f"[*] Loading pairs from: {pairs_path.name}")
    pairs = parse_pairs_file(pairs_path, data_root)
    print(f"[*] Total pairs to evaluate: {len(pairs)}")

    # Extract unique images only once to minimize inference time
    all_unique_paths = sorted(list({p[0] for p in pairs} | {p[1] for p in pairs}))
    print(f"[*] Extracting embeddings for {len(all_unique_paths)} unique images...")

    extractor = FaceEmbeddingExtractor(checkpoint_path=checkpoint_path, normalize=True)
    embeddings = extractor.extract(all_unique_paths, batch_size=batch_size)
    path_to_idx = {p: i for i, p in enumerate(all_unique_paths)}

    similarities = []
    labels = []

    for img1, img2, label in tqdm(pairs, desc="Computing pair similarities"):
        idx1, idx2 = path_to_idx[img1], path_to_idx[img2]
        v1, v2 = embeddings[idx1], embeddings[idx2]

        if metric == "cosine":
            # For unit-normalized vectors: cosine similarity == dot product
            score = float(np.dot(v1, v2))
        elif metric == "euclidean":
            # Negative Euclidean distance so higher values indicate greater similarity
            score = float(-np.linalg.norm(v1 - v2))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        similarities.append(score)
        if label is not None:
            labels.append(label)

    sims_arr = np.array(similarities)

    if labels and len(labels) == len(pairs):
        labels_arr = np.array(labels)
        metrics = compute_metrics(sims_arr, labels_arr)
        print("\n" + "=" * 45)
        print(f"  VERIFICATION BENCHMARK ({metric.upper()})")
        print("=" * 45)
        print(f"  ROC-AUC Score:      {metrics['roc_auc'] * 100:.2f}%")
        print(f"  Equal Error Rate:   {metrics['eer'] * 100:.2f}%")
        print(f"  Best Accuracy:      {metrics['best_accuracy'] * 100:.2f}%")
        print(f"  Optimal Threshold:  {metrics['optimal_threshold']:.4f}")
        print("=" * 45)
        return metrics
    else:
        # Unlabeled test submission
        output_csv = pairs_path.parent / "verification_predictions.csv"
        with output_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "similarity"])
            for idx, s in enumerate(sims_arr):
                writer.writerow([idx, s])
        print(f"[✓] Saved unlabeled prediction scores to: {output_csv}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate face verification performance.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to checkpoint")
    parser.add_argument("--pairs-file", default="data/raw/verification_pairs_val.txt", help="Path to pairs text/csv")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_benchmark(args.checkpoint, args.pairs_file, metric=args.metric, batch_size=args.batch_size)
