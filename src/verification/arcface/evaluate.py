"""Verification of softmax / ArcFace embeddings on a pairs file (cosine vs Euclidean, ROC/AUC/EER).

    python -m src.verification.arcface.evaluate --checkpoint checkpoints/best_softmax.pt --name softmax
    python -m src.verification.arcface.evaluate --checkpoint checkpoints/best_arcface.pt --name arcface

Pairs file default: Config.DATA_ROOT/verification_pairs_val.txt (lines "<path1> <path2> <label>",
paths relative to Config.DATA_ROOT). Writes outputs/evaluation_results/<name>.json, <name>_roc.png
and <name>_scores.npz.

Scores are "higher = more likely same person":
- cosine: cos(e1, e2) on raw embeddings.
- euclidean: -||e1 - e2|| on RAW (unnormalised) embeddings. On L2-normalised embeddings the
  Euclidean distance is a monotonic function of cosine (d^2 = 2 - 2cos) and would give an
  identical ROC, so the raw embedding is used to make the comparison meaningful.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.config import Config

from .dataset import ImageList
from .model import load_embedder

DATASET_DIR = Path(Config.DATA_ROOT)
PAIRS_FILE = DATASET_DIR / "verification_pairs_val.txt"
RESULTS_DIR = Path(Config.PROJECT_ROOT) / "outputs" / "evaluation_results"

METRICS = ("cosine", "euclidean")


def pair_scores(e1: torch.Tensor, e2: torch.Tensor, metric: str) -> np.ndarray:
    if metric == "cosine":
        s = F.cosine_similarity(e1, e2, dim=1)
    elif metric == "euclidean":
        s = -torch.linalg.norm(e1 - e2, dim=1)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return s.cpu().numpy()


def verification_report(labels, scores, metric: str) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    threshold = float(thr[i])
    # Report the threshold in the metric's natural unit (distance for Euclidean).
    natural = threshold if metric == "cosine" else -threshold
    rule = "accept if cosine >= threshold" if metric == "cosine" else "accept if distance <= threshold"
    return {
        "metric": metric,
        "auc": float(roc_auc_score(labels, scores)),
        "eer": float((fpr[i] + fnr[i]) / 2),
        "threshold_at_eer": natural,
        "decision_rule": rule,
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int((labels == 0).sum()),
        "roc": {"fpr": fpr, "tpr": tpr},
    }


def read_pairs(pairs_file, root=None):
    """Lines: '<path1> <path2> <label>'. Paths are relative to root (default: file's folder)."""
    pairs_file = Path(pairs_file)
    root = Path(root) if root else pairs_file.parent
    a, b, y = [], [], []
    for line in pairs_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ValueError(f"bad pairs line: {line!r}")
        a.append(root / parts[0])
        b.append(root / parts[1])
        y.append(int(parts[2]))
    return a, b, np.array(y)


@torch.no_grad()
def embed_paths(model, paths, device, batch_size=256, workers=2):
    loader = DataLoader(ImageList(paths), batch_size=batch_size, num_workers=workers)
    out = torch.empty(len(paths), model.emb_dim)
    for x, idx in loader:
        out[idx] = model(x.to(device)).float().cpu()
    return out


def evaluate(model, pairs_file, root, device, batch_size=256, workers=2):
    a, b, labels = read_pairs(pairs_file, root)
    unique = sorted(set(a) | set(b))
    index = {p: i for i, p in enumerate(unique)}
    emb = embed_paths(model, unique, device, batch_size, workers)
    ea = emb[[index[p] for p in a]]
    eb = emb[[index[p] for p in b]]
    reports, scores = {}, {}
    for metric in METRICS:
        scores[metric] = pair_scores(ea, eb, metric)
        reports[metric] = verification_report(labels, scores[metric], metric)
    return reports, scores, labels


def plot_roc(reports, title, path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    for metric, r in reports.items():
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{metric} (AUC={r['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def save_results(name, checkpoint, pairs, reports, scores, labels, out_dir=RESULTS_DIR):
    """Write <name>.json, <name>_roc.png, <name>_scores.npz; print a summary; return the ROC figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}_scores.npz", labels=labels, **scores)
    fig = plot_roc(reports, f"ROC - {name}", out / f"{name}_roc.png")
    summary = {"name": name, "checkpoint": str(checkpoint), "pairs": str(pairs),
               "metrics": {m: {k: v for k, v in r.items() if k != "roc"} for m, r in reports.items()}}
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2))
    for m, r in summary["metrics"].items():
        print(f"{name} {m:9s} AUC={r['auc']:.4f} EER={r['eer']:.4f} threshold={r['threshold_at_eer']:.4f}")
    return fig


def main():
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True, help="model name used for output files")
    ap.add_argument("--pairs", default=str(PAIRS_FILE))
    ap.add_argument("--root", default=str(DATASET_DIR), help="folder the pair paths are relative to")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if not Path(args.pairs).exists():
        raise SystemExit(f"Pairs file not found: {args.pairs}. Download verification_pairs_val.txt first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_embedder(args.checkpoint, device)
    reports, scores, labels = evaluate(model, args.pairs, args.root, device, args.batch_size, args.workers)

    plt.close(save_results(args.name, args.checkpoint, args.pairs, reports, scores, labels, args.out_dir))


if __name__ == "__main__":
    main()
