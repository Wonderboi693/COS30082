import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import Config
from src.anti_spoofing.dataset import get_antispoof_loaders
from src.anti_spoofing.model import AntiSpoofNet

torch.backends.cudnn.benchmark = True


def compute_iso_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    preds = (probabilities >= threshold).astype(np.float32)
    
    # APCER: Attack presentations classified as bona fide (label 0, pred 1)
    attack_mask = (labels == 0.0)
    num_attacks = np.sum(attack_mask)
    apcer = np.sum(preds[attack_mask] == 1.0) / num_attacks if num_attacks > 0 else 0.0

    # BPCER: Bona fide presentations classified as attacks (label 1, pred 0)
    bona_fide_mask = (labels == 1.0)
    num_bona_fide = np.sum(bona_fide_mask)
    bpcer = np.sum(preds[bona_fide_mask] == 0.0) / num_bona_fide if num_bona_fide > 0 else 0.0

    acer = (apcer + bpcer) / 2.0
    accuracy = (preds == labels).mean()
    auc = roc_auc_score(labels, probabilities) if len(np.unique(labels)) > 1 else 0.5

    return {
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "accuracy": accuracy,
        "roc_auc": auc
    }


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training Liveness", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_loss += loss.item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total_count += len(labels)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * len(labels)
        total_count += len(labels)

    metrics = compute_iso_metrics(np.array(all_probs), np.array(all_labels), threshold=threshold)
    metrics["loss"] = total_loss / total_count
    return metrics


def run():
    parser = argparse.ArgumentParser(description="Train Anti-Spoofing binary liveness classifier.")
    parser.add_argument("--epochs", type=int, default=Config.ANTISPOOF_EPOCHS)
    parser.add_argument("--lr", type=float, default=Config.ANTISPOOF_LR)
    parser.add_argument("--batch-size", type=int, default=Config.ANTISPOOF_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=Config.LIVENESS_THRESHOLD)
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"[*] Initiating Anti-Spoofing Training on {device}")

    train_loader, val_loader = get_antispoof_loaders(batch_size=args.batch_size)
    model = AntiSpoofNet(pretrained_backbone=False).to(device)

    # Positive weight can be tuned if datasets have heavy class imbalance
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_acer = float("inf")
    save_path = Path(Config.ANTISPOOF_CHECKPOINT)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=args.threshold)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Tr Loss: {tr_loss:.4f} (Acc: {tr_acc * 100:.1f}%) | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"ACER: {val_metrics['acer'] * 100:.2f}% | "
              f"APCER: {val_metrics['apcer'] * 100:.2f}% | "
              f"BPCER: {val_metrics['bpcer'] * 100:.2f}% | "
              f"AUC: {val_metrics['roc_auc'] * 100:.2f}%")

        if val_metrics["acer"] < best_acer:
            best_acer = val_metrics["acer"]
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acer": best_acer,
                "val_auc": val_metrics["roc_auc"],
                "threshold": args.threshold,
                "image_size": Config.IMAGE_SIZE[0]
            }
            torch.save(checkpoint, save_path)
            print(f"    --> Saved best anti-spoof checkpoint (ACER: {best_acer * 100:.2f}%) to {save_path.name}")


if __name__ == "__main__":
    run()
