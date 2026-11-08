import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from src.config import Config
from src.emotion.dataset import get_emotion_loaders
from src.emotion.model import EmotionNet

torch.backends.cudnn.benchmark = True


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training EmotionNet", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total_count += len(labels)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

        total_loss += loss.item() * len(labels)
        total_count += len(labels)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = (all_preds == all_targets).mean()
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "loss": total_loss / total_count,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "targets": all_targets
    }


def run():
    parser = argparse.ArgumentParser(description="Train EmotionNet multi-class expression classifier.")
    parser.add_argument("--epochs", type=int, default=Config.EMOTION_EPOCHS)
    parser.add_argument("--lr", type=float, default=Config.EMOTION_LR)
    parser.add_argument("--batch-size", type=int, default=Config.EMOTION_BATCH_SIZE)
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"[*] Training Emotion Recognition Classifier on {device}")

    train_loader, val_loader = get_emotion_loaders(batch_size=args.batch_size)
    model = EmotionNet(num_classes=Config.NUM_EMOTIONS, pretrained_backbone=False).to(device)

    # Label smoothing regularizes overlapping facial expression distributions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = 0.0
    save_path = Path(Config.EMOTION_CHECKPOINT)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_res = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Tr Loss: {tr_loss:.4f} (Acc: {tr_acc * 100:.1f}%) | "
              f"Val Loss: {val_res['loss']:.4f} | "
              f"Val Acc: {val_res['accuracy'] * 100:.2f}% | "
              f"Macro F1: {val_res['macro_f1'] * 100:.2f}%")

        if val_res["macro_f1"] > best_f1:
            best_f1 = val_res["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_accuracy": val_res["accuracy"],
                "val_macro_f1": best_f1,
                "classes": Config.EMOTION_CLASSES,
                "image_size": Config.IMAGE_SIZE[0]
            }, save_path)
            print(f"    --> Saved best emotion checkpoint (Macro F1: {best_f1 * 100:.2f}%) to {save_path.name}")


if __name__ == "__main__":
    run()