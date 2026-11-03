import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import Config
from .dataset import build_loaders, dataset_fingerprint, get_preprocessing_config
from .model import ARCHITECTURES, ClassificationFaceNet, resolve_device


@dataclass
class TrainConfig:
    data_root: str = Config.CLASSIFICATION_DIR
    output_dir: str = Config.SUPERVISED_RUN_DIR
    manifest: str | None = None
    architecture: str = Config.ARCHITECTURE
    embedding_dim: int = Config.EMBEDDING_DIM
    image_size: int = Config.IMAGE_SIZE[0]
    epochs: int = Config.EPOCHS
    batch_size: int = Config.BATCH_SIZE
    learning_rate: float = Config.LEARNING_RATE
    backbone_lr: float = Config.BACKBONE_LEARNING_RATE
    weight_decay: float = Config.WEIGHT_DECAY
    label_smoothing: float = 0.1
    warmup_epochs: int = 2
    patience: int = 10
    num_workers: int = Config.NUM_WORKERS
    seed: int = 42
    device: str = "auto"
    pretrained: bool = True
    amp: bool = True
    check_duplicates: bool = True
    resume: str | None = None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model, loader, device, num_classes):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    # Store observed pairs rather than a large identity-by-identity matrix.
    confusion = Counter()
    predictions = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            probabilities = logits.softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            count += labels.size(0)
            for actual, guess, score in zip(labels.cpu().tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()):
                confusion[(actual, guess)] += 1
                predictions.append({"actual": actual, "predicted": guess, "confidence": score})
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count,
            "confusion_matrix": {"format": "sparse_coo", "shape": [num_classes, num_classes],
                                 "entries": [[actual, guess, count]
                                             for (actual, guess), count in sorted(confusion.items())]}}, predictions


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path, checkpoint):
    # Replace only after a complete write, so an interrupted save preserves the old file.
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(config):
    if config.epochs < 1 or config.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if not 0 <= config.warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be >= 0 and < epochs so the backbone is fine-tuned")
    if config.learning_rate <= 0 or config.backbone_lr <= 0 or config.weight_decay < 0:
        raise ValueError("Learning rates must be positive and weight_decay nonnegative")
    if not 0 <= config.label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    loaders, classes = build_loaders(
        config.data_root, config.batch_size, config.image_size, config.num_workers,
        config.seed, config.manifest, config.check_duplicates, device.type == "cuda",
    )
    fingerprint = dataset_fingerprint(loaders, config.data_root)
    preprocessing = get_preprocessing_config(config.image_size)
    output = Path(config.output_dir)
    # A run directory represents one experiment; do not mix histories/checkpoints.
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise ValueError(f"Run directory is not empty: {output}. Choose another or use --resume.")
    resume = torch.load(config.resume, map_location="cpu", weights_only=True) if config.resume else None
    model = ClassificationFaceNet(len(classes), config.embedding_dim, config.architecture,
                           pretrained=config.pretrained and resume is None).to(device)
    if resume:
        if resume["model_config"] != model.config or resume["class_to_idx"] != classes:
            raise ValueError("Resume checkpoint model/identity mapping differs from this run")
        if resume["data_fingerprint"] != fingerprint or resume["image_size"] != config.image_size:
            raise ValueError("Resume dataset/splits/preprocessing differ from the saved run")
        # Scheduling and augmentation settings must match to preserve experiment meaning.
        for name in ("epochs", "warmup_epochs", "learning_rate", "backbone_lr", "weight_decay", "label_smoothing", "seed", "batch_size"):
            if resume["train_config"][name] != getattr(config, name):
                raise ValueError(f"Resume setting changed: {name}. Start a new experiment instead.")
        if Path(config.resume).resolve().parent != output.resolve():
            raise ValueError("Resume in the original run directory (copy its whole directory between PCs)")
        if (not (output / "best.pt").is_file() and resume["best_accuracy"] >= 0
                and resume["bad_epochs"] > 0):
            raise ValueError("Resume needs the run's earlier best.pt as well as last.pt")
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": config.backbone_lr},
        {"params": list(model.embedding.parameters()) + list(model.classifier.parameters()), "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    start, steps, best_accuracy, best_loss, bad_epochs = 0, 0, -1.0, float("inf"), 0
    history = []
    attempted_steps = 0
    if resume:
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        start, steps = resume["epoch"] + 1, resume["training_steps"]
        attempted_steps = resume.get("attempted_steps", steps)
        best_accuracy, best_loss = resume["best_accuracy"], resume["best_loss"]
        bad_epochs, history = resume["bad_epochs"], resume["history"]
        torch.set_rng_state(resume["torch_rng_state"])
        random.setstate(resume["python_rng_state"])
        loaders["train"].generator.set_state(resume["loader_rng_state"])
        if device.type == "cuda" and resume.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
    output.mkdir(parents=True, exist_ok=True)
    if resume and not (output / "best.pt").is_file() and resume["best_accuracy"] >= 0:
        # Recover a crash between saving an improved last.pt and saving best.pt.
        save_checkpoint(output / "best.pt", resume)
    write_json(output / "config.json", asdict(config))
    write_json(output / "class_to_idx.json", classes)
    print(f"Device: {device}; classes: {len(classes)}; split sizes: "
          f"{ {key: len(loader.dataset) for key, loader in loaders.items()} }", flush=True)
    for epoch in range(start, config.epochs):
        if bad_epochs >= config.patience:
            break
        model.train()
        frozen = epoch < config.warmup_epochs
        model.features.requires_grad_(not frozen)
        if frozen:
            model.features.eval()  # Keep pretrained batch-norm statistics during head warm-up.
        total_loss, correct, count = 0.0, 0, 0
        epoch_updates, skipped_updates = 0, 0
        epoch_started = time.perf_counter()
        for batch_index, (images, labels) in enumerate(loaders["train"], start=1):
            images, labels = images.to(device), labels.to(device)
            # A final singleton batch can reach a 1x1 feature map at small image
            # sizes. Use existing BN statistics for that batch instead of dropping it.
            singleton_bn = []
            if labels.size(0) == 1:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d) and module.training:
                        singleton_bn.append(module)
                        module.eval()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            if not math.isfinite(loss.item()):
                raise ValueError("Non-finite training loss; lower learning rates and inspect the data")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            attempted_steps += 1
            # AMP may skip an update on non-finite gradients and reduce its scale.
            if not use_amp or scaler.get_scale() >= previous_scale:
                steps += 1
                epoch_updates += 1
            else:
                skipped_updates += 1
            for module in singleton_bn:
                module.train()
            count += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            if batch_index == 1 or batch_index % 100 == 0:
                elapsed = time.perf_counter() - epoch_started
                print(f"Epoch {epoch + 1}/{config.epochs}, batch {batch_index}/{len(loaders['train'])}: "
                      f"{count / max(elapsed, 1e-6):.1f} images/s, "
                      f"updates={epoch_updates}, AMP skipped={skipped_updates}", flush=True)
        if epoch_updates == 0:
            raise ValueError("No optimiser updates succeeded. Retry with --no-amp or inspect gradient stability.")
        validation, _ = evaluate(model, loaders["val"], device, len(classes))
        if not math.isfinite(validation["loss"]):
            raise ValueError("Non-finite validation loss")
        history.append({"epoch": epoch + 1, "train_loss": total_loss / count,
                        "train_accuracy": correct / count, "val_loss": validation["loss"],
                        "val_accuracy": validation["accuracy"], "backbone_frozen": frozen,
                        "optimizer_updates": epoch_updates, "amp_skipped_batches": skipped_updates})
        # Do not select a head-only warm-up checkpoint as the final fine-tuned model.
        improved = not frozen and (validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss, bad_epochs = validation["accuracy"], validation["loss"], 0
        elif not frozen:
            bad_epochs += 1
        scheduler.step()
        checkpoint = {
            "model_state": model.state_dict(), "model_config": model.config,
            "class_to_idx": classes, "image_size": config.image_size,
            "preprocessing": preprocessing,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "epoch": epoch, "training_steps": steps,
            "best_accuracy": best_accuracy, "best_loss": best_loss, "bad_epochs": bad_epochs,
            "attempted_steps": attempted_steps, "history": history, "train_config": asdict(config), "data_fingerprint": fingerprint,
            "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
            "loader_rng_state": loaders["train"].generator.get_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        # best_loss is infinity until the warm-up ends; it is not written to JSON.
        save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch + 1}/{config.epochs}: train={correct / count:.4f}, "
              f"val={validation['accuracy']:.4f}, val_loss={validation['loss']:.4f}", flush=True)
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    # The held-out test set is evaluated only after checkpoint selection is finished.
    test_metrics, predictions = evaluate(model, loaders["test"], device, len(classes))
    metrics = {"best_epoch": best["epoch"] + 1, "validation_accuracy": best["best_accuracy"],
               "validation_loss": best["best_loss"], "test": test_metrics,
               "class_to_idx": classes, "data_fingerprint": fingerprint,
               "training_steps": steps, "attempted_steps": attempted_steps, "device": str(device)}
    write_json(output / "metrics.json", metrics)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "actual", "predicted", "confidence"])
        writer.writeheader()
        for sample, prediction in zip(loaders["test"].dataset.samples, predictions):
            writer.writerow({"image_path": sample.path.relative_to(Path(config.data_root).resolve()).as_posix(), **prediction})
    save_checkpoint(output / "embeddings.pt", {
        "kind": "embeddings", "model_state": model.without_classifier().state_dict(),
        "model_config": model.config, "image_size": config.image_size,
        "preprocessing": preprocessing, "class_to_idx": classes,
        "training_steps": best["training_steps"], "data_fingerprint": fingerprint,
    })
    print(f"Best epoch {metrics['best_epoch']}; test accuracy={test_metrics['accuracy']:.4f}", flush=True)
    return metrics


def run():
    parser = argparse.ArgumentParser(description="Train the supervised face classifier and export embeddings.")
    parser.add_argument("--data-root", default=Config.CLASSIFICATION_DIR)
    parser.add_argument("--output-dir", default=Config.SUPERVISED_RUN_DIR)
    parser.add_argument("--manifest")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=Config.ARCHITECTURE)
    for name, default in [("embedding-dim", Config.EMBEDDING_DIM), ("image-size", Config.IMAGE_SIZE[0]), ("epochs", Config.EPOCHS),
                          ("batch-size", Config.BATCH_SIZE), ("warmup-epochs", 2), ("patience", 10),
                          ("num-workers", Config.NUM_WORKERS), ("seed", 42)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in [("learning-rate", Config.LEARNING_RATE), ("backbone-lr", Config.BACKBONE_LEARNING_RATE),
                          ("weight-decay", Config.WEIGHT_DECAY), ("label-smoothing", 0.1)]:
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-duplicate-check", action="store_true")
    parser.add_argument("--resume")
    options = vars(parser.parse_args())
    options["pretrained"] = not options.pop("no_pretrained")
    options["amp"] = not options.pop("no_amp")
    options["check_duplicates"] = not options.pop("skip_duplicate_check")
    try:
        train(TrainConfig(**options))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    run()
