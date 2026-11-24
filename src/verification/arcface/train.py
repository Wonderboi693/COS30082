"""Train classification-based face embeddings (softmax baseline or ArcFace).

    python -m src.verification.arcface.train --loss softmax           # -> checkpoints/best_softmax.pt
    python -m src.verification.arcface.train --loss arcface           # -> checkpoints/best_arcface.pt
    python -m src.verification.arcface.train --loss arcface --resume  # continue an interrupted run

Each run folder Config.ARCFACE_RUN_DIR/<run_name>/ holds
config.json, history.csv, last.pt (end of every completed epoch, used for resume) and
best.pt (highest val_data accuracy). An interrupted epoch is replayed in full on resume.
Early stopping: training stops when val accuracy has not improved by at least min_delta
for `patience` consecutive epochs (the counter is saved in last.pt, so it survives resume).
After the final epoch the best model predicts the unlabelled test_data and writes
test_predictions.csv (format of the sample submission), and the best weights are exported
to checkpoints/best_<loss>.pt in the group's format ("model_state", "model_config", ...).
"""
import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import Config

from .dataset import IMG_SIZE, MEAN, STD, TEST_DIR, TEST_LIST, TRAIN_DIR, VAL_DIR, ImageList, build_loaders
from .model import EMB_DIM, FaceEmbedder, build_head

EXPORT_PATHS = {"softmax": Config.SOFTMAX_CHECKPOINT, "arcface": Config.ARCFACE_CHECKPOINT}
THRESHOLDS = {"softmax": Config.SOFTMAX_THRESHOLD, "arcface": Config.ARCFACE_THRESHOLD}
DEFAULT_RUNS = {"softmax": "softmax_v2", "arcface": "arcface_v4"}
DEFAULT_EPOCHS = {"softmax": Config.SOFTMAX_EPOCHS, "arcface": Config.ARCFACE_EPOCHS}


@dataclass
class TrainConfig:
    loss: str                        # "softmax" or "arcface"
    run_name: str
    epochs: int = 40                 # maximum; early stopping usually ends sooner
    patience: int = 5                # early stop after this many epochs without improvement (0 = off)
    min_delta: float = 0.002         # val accuracy gain that counts as an improvement
    batch_size: int = Config.ARCFACE_BATCH_SIZE
    lr: float = Config.ARCFACE_LR    # peak SGD lr for neck + head
    backbone_lr_mult: float = 0.1    # pretrained backbone uses lr * mult
    weight_decay: float = 5e-4
    warmup_epochs: float = 1.0
    freeze: str = "stem"             # none | stem | layer1 | layer2 | layer3
    arc_s: float = Config.ARCFACE_SCALE
    arc_m: float = Config.ARCFACE_MARGIN
    margin_warmup_epochs: float = 2.0  # ArcFace margin grows linearly from 0 to arc_m
    workers: int = 4
    seed: int = 82
    resume: bool = False
    smoke: bool = False              # 1 epoch, 2 train + 2 val batches (pipeline check)
    pretrained: bool = True          # False only for tests
    train_dir: str = str(TRAIN_DIR)
    val_dir: str = str(VAL_DIR)
    test_dir: str = str(TEST_DIR)
    test_list: str = str(TEST_LIST)
    runs_dir: str = str(Config.ARCFACE_RUN_DIR)
    min_free_ram_mb: int = 700       # stop safely (keep finished epochs) if available RAM drops below this
    log_every: int = 200             # steps between lines in <run>/progress.log
    export_path: str = ""             # export best weights here in the group format ("" = no export)


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def val_accuracy(model, head, loader, device, max_batches=None):
    model.eval(); head.eval()
    correct = total = 0
    n = min(len(loader), max_batches) if max_batches else len(loader)
    for i, (x, y) in enumerate(tqdm(loader, total=n, desc="  validating", leave=False)):
        if i >= n:
            break
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = head(model(x.to(device, non_blocking=True)))
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total += len(y)
    return correct / max(1, total)


@torch.no_grad()
def predict_test(model, head, classes, device, test_dir, test_list, out_path, workers=2, max_images=None):
    names = [n.strip() for n in Path(test_list).read_text().splitlines() if n.strip()]
    names = names[:max_images] if max_images else names
    loader = DataLoader(ImageList([Path(test_dir) / n for n in names]), batch_size=256, num_workers=workers)
    model.eval(); head.eval()
    preds = [None] * len(names)
    for x, idx in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = head(model(x.to(device))).argmax(1).cpu()
        for i, c in zip(idx.tolist(), p.tolist()):
            preds[i] = classes[c]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(zip(names, preds))


def train(cfg: TrainConfig) -> Path:
    """Train one model; returns the run folder."""
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    if (run_dir / "summary.json").exists() and not cfg.smoke:
        print(f"{cfg.run_name} already finished ({run_dir}); skipping. Delete the folder to retrain.")
        return run_dir
    if (run_dir / "last.pt").exists() and not cfg.resume and not cfg.smoke:
        raise RuntimeError(f"{run_dir} has a checkpoint. Use --resume, or a new --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(cfg.batch_size, cfg.workers, cfg.train_dir, cfg.val_dir)
    classes = train_loader.dataset.classes
    model = FaceEmbedder(EMB_DIM, pretrained=cfg.pretrained, freeze=cfg.freeze).to(device)
    head = build_head(cfg.loss, EMB_DIM, len(classes), s=cfg.arc_s, m=cfg.arc_m).to(device)

    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("neck")]
    new = list(model.neck.parameters()) + list(head.parameters())
    opt = torch.optim.SGD([{"params": backbone, "lr": cfg.lr * cfg.backbone_lr_mult},
                           {"params": new, "lr": cfg.lr}],
                          momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = min(2, len(train_loader)) if cfg.smoke else len(train_loader)
    epochs = 1 if cfg.smoke else cfg.epochs
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    config = {**asdict(cfg), "emb_dim": EMB_DIM, "img_size": IMG_SIZE, "num_classes": len(classes),
              "backbone": "resnet18", "init": "imagenet" if cfg.pretrained else "random",
              "optimizer": "SGD nesterov momentum 0.9", "schedule": "linear warmup + cosine (per step)",
              "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset)}

    start_epoch, best_acc, no_improve = 0, -1.0, 0
    last_path, best_path, hist_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "history.csv"
    if cfg.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"]); head.load_state_dict(ckpt["head"])
        opt.load_state_dict(ckpt["optimizer"]); scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt["best_val_acc"]
        no_improve = ckpt.get("epochs_no_improve", 0)
        torch.set_rng_state(ckpt["rng"].cpu())
        print(f"Resumed from completed epoch {ckpt['epoch'] + 1}; best val acc {best_acc:.4f}; "
              f"{no_improve} epoch(s) without improvement")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    if not hist_path.exists() or start_epoch == 0:
        hist_path.write_text("epoch,train_loss,train_acc,val_acc,lr,margin,seconds\n")

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Run {cfg.run_name} | loss {cfg.loss} | device {gpu} | {len(classes)} classes | "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images | "
          f"{steps_per_epoch} batches x {cfg.batch_size} per epoch | epochs {start_epoch + 1}..{epochs} | "
          f"early stop: {'patience ' + str(cfg.patience) if cfg.patience else 'off'}")

    stopped_early, epochs_trained = False, start_epoch

    for epoch in range(start_epoch, epochs):
        model.train(); head.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        bar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch",
                   mininterval=1.0)
        for i, (x, y) in enumerate(bar):
            if i >= steps_per_epoch:
                break
            step = epoch * steps_per_epoch + i
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_factor(step, total_steps, warmup_steps)
            if cfg.loss == "arcface":
                ramp = min(1.0, step / max(1, cfg.margin_warmup_epochs * steps_per_epoch))
                head.set_margin(cfg.arc_m * ramp)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = head(model(x), y)
                loss = criterion(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            scaler.step(opt); scaler.update()
            if not math.isfinite(loss.item()):
                raise RuntimeError("Loss became non-finite; lower the learning rate.")
            loss_sum += loss.item() * len(y); seen += len(y)
            # ArcFace logits include the margin penalty on the true class; accuracy uses margin-free cosines.
            scores = head.last_cos if cfg.loss == "arcface" else logits
            correct += (scores.argmax(1) == y).sum().item()
            if (i + 1) % cfg.log_every == 0 or i + 1 == steps_per_epoch:
                ram = psutil.virtual_memory().available // 2**20
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (steps_per_epoch - i - 1)
                vram = torch.cuda.max_memory_allocated() // 2**20 if device.type == "cuda" else 0
                with open(run_dir / "progress.log", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} epoch {epoch + 1}/{epochs} "
                            f"step {i + 1}/{steps_per_epoch} ({100 * (i + 1) / steps_per_epoch:.0f}%) "
                            f"loss {loss.item():.3f} acc {correct / seen:.3f} "
                            f"{(i + 1) / elapsed:.1f} batch/s epoch_eta {eta / 60:.1f} min "
                            f"ram_free {ram} MB vram_peak {vram} MB\n")
                if ram < cfg.min_free_ram_mb:
                    raise RuntimeError(f"Available RAM {ram} MB < {cfg.min_free_ram_mb} MB: stopped to protect WSL. "
                                       "Finished epochs are saved; resume later with fewer workers.")
            if i % 20 == 0:
                bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / seen:.3f}",
                                lr=f"{opt.param_groups[1]['lr']:.4f}",
                                **({"m": f"{head.m:.2f}"} if cfg.loss == "arcface" else {}))
        bar.close()
        epochs_trained = epoch + 1

        val_acc = val_accuracy(model, head, val_loader, device, max_batches=2 if cfg.smoke else None)
        row = [epoch + 1, loss_sum / seen, correct / seen, val_acc, opt.param_groups[1]["lr"],
               getattr(head, "m", 0.0), time.time() - t0]
        with open(hist_path, "a") as f:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")

        # Checkpoints only at the end of a fully completed epoch.
        is_best = val_acc > best_acc
        improved = val_acc > best_acc + cfg.min_delta
        no_improve = 0 if improved else no_improve + 1
        state = {"model": model.state_dict(), "head": head.state_dict(), "config": config,
                 "classes": classes, "epoch": epoch, "val_acc": val_acc}
        if is_best:
            best_acc = val_acc
            torch.save(state, best_path)
        torch.save({**state, "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best_val_acc": best_acc, "epochs_no_improve": no_improve, "rng": torch.get_rng_state()}, last_path)
        with open(run_dir / "progress.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} EPOCH {epoch + 1}/{epochs} DONE train_loss {row[1]:.4f} "
                    f"train_acc {row[2]:.4f} val_acc {val_acc:.4f} best {max(best_acc, val_acc):.4f} "
                    f"no_improve {no_improve} ({row[-1]:.0f}s)\n")
        print(f"Epoch {epoch + 1}/{epochs} done: train_loss {row[1]:.4f} train_acc {row[2]:.4f} "
              f"val_acc {val_acc:.4f} ({row[-1]:.0f}s) | saved last.pt{' + best.pt (new best)' if is_best else ''}"
              f"{f' | no improvement {no_improve}/{cfg.patience}' if cfg.patience and no_improve else ''}")
        if cfg.patience and no_improve >= cfg.patience:
            stopped_early = True
            print(f"Early stopping: val accuracy did not improve by {cfg.min_delta} for {cfg.patience} epochs.")
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"]); head.load_state_dict(best["head"])
    predict_test(model, head, classes, device, cfg.test_dir, cfg.test_list, run_dir / "test_predictions.csv",
                 cfg.workers, max_images=16 if cfg.smoke else None)
    (run_dir / "summary.json").write_text(json.dumps(
        {"best_val_acc": best_acc, "best_epoch": best["epoch"] + 1, "max_epochs": epochs,
         "epochs_trained": epochs_trained, "stopped_early": stopped_early}, indent=2))
    if cfg.export_path and not cfg.smoke:
        export_checkpoint(run_dir, cfg.export_path)
    print(f"Done. Best val acc {best_acc:.4f} at epoch {best['epoch'] + 1}. Run: {run_dir}")
    return run_dir


def export_checkpoint(run_dir, out_path):
    """Write the best weights of a run in the group's checkpoint format (inference only).

    The stored threshold is the validation EER threshold from src/config.py; re-evaluate
    (python -m src.verification.arcface.evaluate) and update Config after retraining.
    """
    best = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    loss = best["config"]["loss"]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "embeddings",
        "epoch": best["epoch"] + 1,
        "model_state": best["model"],
        "model_config": {"embedding_dim": best["config"].get("emb_dim", EMB_DIM), "architecture": "resnet18",
                         "loss": loss},
        "val_accuracy": best["val_acc"],
        "threshold": THRESHOLDS.get(loss),
        "image_size": IMG_SIZE,
        "preprocessing": {"image_size": IMG_SIZE, "mean": list(MEAN), "std": list(STD)},
        "run_name": best["config"].get("run_name"),
    }, out)
    print(f"    --> Exported {loss} embedding checkpoint to {out}")
    return out


def run():
    parser = argparse.ArgumentParser(description="Train softmax / ArcFace face embeddings.")
    parser.add_argument("--loss", choices=["softmax", "arcface"], required=True)
    parser.add_argument("--run-name", default=None, help="default: softmax_v2 / arcface_v4")
    parser.add_argument("--epochs", type=int, default=None, help="default: Config.SOFTMAX_EPOCHS / ARCFACE_EPOCHS")
    parser.add_argument("--batch-size", type=int, default=Config.ARCFACE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.ARCFACE_LR)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="continue from the last finished epoch")
    parser.add_argument("--smoke", action="store_true", help="2-batch pipeline check, nothing exported")
    parser.add_argument("--export-only", action="store_true", help="only export an existing run's best.pt")
    args = parser.parse_args()

    run_name = args.run_name or DEFAULT_RUNS[args.loss]
    if args.export_only:
        export_checkpoint(Path(Config.ARCFACE_RUN_DIR) / run_name, EXPORT_PATHS[args.loss])
        return
    train(TrainConfig(
        loss=args.loss, run_name=f"smoke_{args.loss}" if args.smoke else run_name,
        epochs=args.epochs or DEFAULT_EPOCHS[args.loss], batch_size=args.batch_size, lr=args.lr,
        workers=args.workers, resume=args.resume, smoke=args.smoke, export_path=EXPORT_PATHS[args.loss]))


if __name__ == "__main__":
    run()
