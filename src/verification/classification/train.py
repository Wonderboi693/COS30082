import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.config import Config
from .dataset import build_loaders, get_preprocessing_config
from .model import ARCHITECTURES, ClassificationFaceNet, resolve_device


@dataclass
class TrainConfig:
    data_root: str = Config.CLASSIFICATION_DIR
    output_dir: str = Config.CHECKPOINT_DIR
    manifest: str | None = None
    architecture: str = getattr(Config, "ARCHITECTURE", "resnet18")
    embedding_dim: int = Config.EMBEDDING_DIM
    image_size: int = Config.IMAGE_SIZE[0]
    epochs: int = Config.EPOCHS
    batch_size: int = Config.BATCH_SIZE
    learning_rate: float = Config.LEARNING_RATE
    weight_decay: float = Config.WEIGHT_DECAY
    patience: int = 7
    num_workers: int = Config.NUM_WORKERS
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    resume: str | None = None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct_tensor = torch.tensor(0, device=device)
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type):
                logits, _ = model(images)
                loss = nn.functional.cross_entropy(logits, labels)

            b_size = labels.size(0)
            total_loss += loss.item() * b_size
            correct_tensor += (logits.argmax(dim=1) == labels).sum()
            total_samples += b_size

    acc = (correct_tensor.item() / total_samples) if total_samples > 0 else 0.0
    return {"loss": total_loss / max(total_samples, 1), "accuracy": acc, "samples": total_samples}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def train(config):
    seed_everything(config.seed)
    device = resolve_device(config.device)
    loaders, classes = build_loaders(
        config.data_root, config.batch_size, config.image_size, config.num_workers,
        config.seed, config.manifest, check_duplicates=False, pin_memory=(device.type == "cuda"),
    )
    preprocessing = get_preprocessing_config(config.image_size)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = ClassificationFaceNet(
        num_classes=len(classes),
        embedding_dim=config.embedding_dim,
        architecture=config.architecture
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = config.amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_accuracy, best_loss, bad_epochs = 0, -1.0, float("inf"), 0
    history = []

    if config.resume and Path(config.resume).is_file():
        resume_data = torch.load(config.resume, map_location=device)
        model.load_state_dict(resume_data["model_state"])
        optimizer.load_state_dict(resume_data["optimizer_state"])
        scheduler.load_state_dict(resume_data["scheduler_state"])
        scaler.load_state_dict(resume_data["scaler_state"])
        start_epoch = resume_data["epoch"] + 1
        best_accuracy = resume_data.get("best_accuracy", -1.0)
        best_loss = resume_data.get("best_loss", float("inf"))
        bad_epochs = resume_data.get("bad_epochs", 0)
        history = resume_data.get("history", [])
        print(f"[*] Resumed from {config.resume} at Epoch {start_epoch + 1}")

    print(f"[*] Training on {device} | {len(classes)} classes | AMP: {use_amp}")

    for epoch in range(start_epoch, config.epochs):
        if bad_epochs >= config.patience:
            print(f"[*] Early stopping triggered after {config.patience} epochs without improvement.")
            break

        model.train()
        total_loss, total_count = 0.0, 0
        correct_tensor = torch.tensor(0, device=device)

        pbar = tqdm(loaders["train"], desc=f"Epoch [{epoch + 1:02d}/{config.epochs:02d}] Train", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            b_size = labels.size(0)
            total_loss += loss.item() * b_size
            correct_tensor += (logits.argmax(dim=1) == labels).sum()
            total_count += b_size

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_acc = correct_tensor.item() / max(total_count, 1)
        train_loss = total_loss / max(total_count, 1)

        val_metrics = evaluate(model, loaders["val"], device)
        val_acc, val_loss = val_metrics["accuracy"], val_metrics["loss"]

        improved = val_acc > best_accuracy
        if improved:
            best_accuracy, best_loss, bad_epochs = val_acc, val_loss, 0
        else:
            bad_epochs += 1

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })

        print(f"Epoch [{epoch + 1:02d}/{config.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model.config,
            "class_to_idx": classes,
            "image_size": config.image_size,
            "preprocessing": preprocessing,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "best_loss": best_loss,
            "bad_epochs": bad_epochs,
            "history": history
        }

        save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            print(f"    --> Saved best checkpoint: Val Acc = {val_acc * 100:.2f}%")

    # Evaluate best weights on test split
    best_ckpt = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate(model, loaders["test"], device)
    print(f"\n[Test Set Final Result] Loss: {test_metrics['loss']:.4f} | Accuracy: {test_metrics['accuracy'] * 100:.2f}%")

    # Export inference-only embeddings checkpoint
    save_checkpoint(output / "embeddings.pt", {
        "kind": "embeddings",
        "model_state": model.without_classifier(normalize=True).state_dict(),
        "model_config": model.config,
        "image_size": config.image_size,
        "preprocessing": preprocessing,
        "class_to_idx": classes,
        "training_steps": 1000
    })
    return test_metrics


def run():
    parser = argparse.ArgumentParser(description="Train face classifier.")
    parser.add_argument("--data-root", default=Config.CLASSIFICATION_DIR)
    parser.add_argument("--output-dir", default=Config.CHECKPOINT_DIR)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    train(TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        resume=args.resume,
        amp=not args.no_amp
    ))


if __name__ == "__main__":
    run()