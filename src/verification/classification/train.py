import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import Config
from src.verification.classification.dataset import get_classification_loaders
from src.verification.classification.model import ClassificationFaceNet

# Enable cuDNN autotuner for fixed input sizes (64x64)
torch.backends.cudnn.benchmark = True

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}] Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # FP16 Automatic Mixed Precision
        with torch.amp.autocast('cuda'):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "acc": f"{(correct/total)*100:.2f}%"
        })

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def run():
    print(f"[*] Running on device: {Config.DEVICE} with Mixed Precision (AMP)")
    train_loader, val_loader, test_loader, num_classes = get_classification_loaders()
    print(f"[*] Total identities in training set: {num_classes}")

    model = ClassificationFaceNet(num_classes=num_classes).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, Config.DEVICE, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, Config.DEVICE)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "embedding_dim": Config.EMBEDDING_DIM
            }, Config.SUPERVISED_CHECKPOINT)
            print(f"    --> Saved best checkpoint: Val Acc = {val_acc*100:.2f}%")

    # Evaluate on held-out test data
    best_weights = torch.load(Config.SUPERVISED_CHECKPOINT, map_location=Config.DEVICE)
    model.load_state_dict(best_weights["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, Config.DEVICE)
    print(f"\n[Test Set Final Result] Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    run()
