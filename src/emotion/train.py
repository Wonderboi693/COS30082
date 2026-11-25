import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from src.config import Config
from src.emotion.dataset import get_emotion_loaders
from src.emotion.model import EmotionNet

torch.backends.cudnn.benchmark = True
AMP_ENABLED = True

PRETRAINED_STAGES = (3, 4, 5, 6)
FRESH_STEM = (0, 1)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stage_modules(model, indices):
    return [model.features[i] for i in indices]


def freeze_pretrained_stages(model):
    for m in stage_modules(model, PRETRAINED_STAGES):
        for p in m.parameters():
            p.requires_grad = False


def train_epoch(model, loader, criterion, optimizer, scaler, device, frozen_eval=(),
                max_batches=None):
    model.train()
    for m in frozen_eval:
        m.eval()

    total_loss, total_correct, total_count = 0.0, 0, 0
    use_amp = AMP_ENABLED and device.type == "cuda"

    for i, (images, labels) in enumerate(tqdm(loader, desc="Training EmotionNet", leave=False)):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += len(labels)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_preds, all_targets = [], []
    use_amp = AMP_ENABLED and device.type == "cuda"

    for i, (images, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        total_loss += loss.item() * len(labels)
        total_count += len(labels)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return {
        "loss": total_loss / total_count,
        "accuracy": float((all_preds == all_targets).mean()),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
        "predictions": all_preds,
        "targets": all_targets,
    }


def run():
    global AMP_ENABLED
    parser = argparse.ArgumentParser(description="Train EmotionNet.")
    parser.add_argument("--lr", type=float, default=Config.EMOTION_LR, help="Base learning rate")
    parser.add_argument("--batch-size", type=int, default=Config.EMOTION_BATCH_SIZE)
    parser.add_argument("--head-epochs", type=int, default=5,
                        help="Stage 1 warm-up epochs (pretrained stages frozen). 0 to skip.")
    parser.add_argument("--full-epochs", type=int, default=25,
                        help="Stage 2 epochs (all layers trainable)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (epochs without val macro-F1 improvement).")
    parser.add_argument("--pretrained-lr-mult", type=float, default=0.5,
                        help="LR multiplier for pretrained backbone during stage 2.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="Initialise backbone from ImageNet (use --no-pretrained for scratch baseline)")
    parser.add_argument("--tag", default=None, help="Run name used for log file")
    parser.add_argument("--save-path", default=None,
                        help="Checkpoint path (default: Config.EMOTION_CHECKPOINT)")
    parser.add_argument("--results-dir", default=None,
                        help="Where to write the JSON run log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (use if validation loss is NaN)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Smoke test only: limit batches per epoch / evaluation")
    args = parser.parse_args()
    AMP_ENABLED = not args.no_amp

    set_seed(args.seed)
    device = Config.DEVICE
    tag = args.tag or ("pretrained" if args.pretrained else "scratch")
    print(f"[*] Training Emotion Recognition Classifier on {device} (run: {tag})")

    train_loader, val_loader, test_loader = get_emotion_loaders(batch_size=args.batch_size)

    model = EmotionNet(
        num_classes=Config.NUM_EMOTIONS,
        pretrained_backbone=args.pretrained,
        freeze_backbone=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    save_path = Path(args.save_path or Config.EMOTION_CHECKPOINT)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir) if args.results_dir else Path(Config.PROJECT_ROOT) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"emotion_{tag}.json"

    def dump_log(extra=None):
        data = {
            "tag": tag,
            "hyperparameters": vars(args),
            "image_size": Config.IMAGE_SIZE,
            "classes": Config.EMOTION_CLASSES,
            "best_val_macro_f1": best_f1,
            "history": history,
        }
        data.update(extra or {"status": "running"})
        log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    use_amp = AMP_ENABLED and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = 0.0
    patience_counter = 0
    history = []

    def save_best(epoch, stage, val_res):
        torch.save({
            "epoch": epoch,
            "stage": stage,
            "model_state": model.state_dict(),
            "classes": Config.EMOTION_CLASSES,
            "image_size": Config.IMAGE_SIZE[0],
            "val_macro_f1": val_res["macro_f1"],
            "val_accuracy": val_res["accuracy"],
            "train_args": vars(args),
        }, save_path)

    def run_stage(name, epochs, optimizer, scheduler, frozen_eval=()):
        nonlocal best_f1, patience_counter
        if epochs <= 0:
            return
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer,
                                          scaler, device, frozen_eval,
                                          max_batches=args.max_batches)
            val_res = evaluate(model, val_loader, criterion, device,
                               max_batches=args.max_batches)
            scheduler.step()
            history.append({
                "stage": name, "epoch": epoch,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": val_res["loss"], "val_acc": val_res["accuracy"],
                "val_macro_f1": val_res["macro_f1"],
            })
            dump_log()
            print(f"  [{name}] Epoch {epoch:02d}/{epochs:02d} "
                  f"Tr Loss {tr_loss:.4f} (Acc {tr_acc*100:.1f}%) | "
                  f"Val Loss {val_res['loss']:.4f} | Val Acc {val_res['accuracy']*100:.2f}% | "
                  f"Macro F1 {val_res['macro_f1']*100:.2f}%")
            if val_res["macro_f1"] > best_f1:
                best_f1 = val_res["macro_f1"]
                patience_counter = 0
                save_best(epoch, name, val_res)
                print(f"    --> Saved best checkpoint ({name}) Macro F1 {best_f1*100:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"    --> Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break

    fresh_params = (
        [p for i in FRESH_STEM for p in model.features[i].parameters()]
        + list(model.classifier.parameters())
    )
    pretrained_params = [p for i in PRETRAINED_STAGES for p in model.features[i].parameters()]
    frozen_bn = stage_modules(model, PRETRAINED_STAGES)

    t0 = time.time()
    if args.pretrained:
        if args.head_epochs > 0:
            print(f"\n[*] Stage 1: warm-up {args.head_epochs} epochs (ResNet stages frozen)")
            freeze_pretrained_stages(model)
            opt1 = optim.AdamW(fresh_params, lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
            sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.head_epochs)
            run_stage("warmup", args.head_epochs, opt1, sched1, frozen_eval=frozen_bn)

        print(f"\n[*] Stage 2: full fine-tuning {args.full_epochs} epochs (all layers trainable)")
        model.unfreeze_backbone()
        opt2 = optim.AdamW(
            [{"params": pretrained_params, "lr": args.lr * args.pretrained_lr_mult},
             {"params": fresh_params, "lr": args.lr}],
            weight_decay=Config.WEIGHT_DECAY,
        )
        sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.full_epochs)
        run_stage("full_finetune", args.full_epochs, opt2, sched2)
    else:
        total = args.head_epochs + args.full_epochs
        print(f"\n[*] Training from scratch for {total} epochs (single stage)")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        run_stage("scratch", total, opt, sched)

    train_minutes = (time.time() - t0) / 60.0
    print(f"\n[✓] Training done in {train_minutes:.1f} min. Best val Macro F1: {best_f1*100:.2f}%")

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if test_loader is not None:
        split, final_loader = "test", test_loader
    else:
        split, final_loader = "val", val_loader
        print("[!] No test split available: the final metrics below are on the VALIDATION split.")
    final_res = evaluate(model, final_loader, criterion, device, max_batches=args.max_batches)
    labels = list(range(Config.NUM_EMOTIONS))
    report = classification_report(
        final_res["targets"], final_res["predictions"], labels=labels,
        target_names=Config.EMOTION_CLASSES, zero_division=0, output_dict=True)
    cm = confusion_matrix(final_res["targets"], final_res["predictions"], labels=labels)

    print(f"[{split.upper()}] Accuracy {final_res['accuracy']*100:.2f}% | Macro F1 {final_res['macro_f1']*100:.2f}%")
    print(classification_report(final_res["targets"], final_res["predictions"], labels=labels,
                                target_names=Config.EMOTION_CLASSES, zero_division=0))

    log = {
        "status": "finished",        "status": "finished",
        "tag": tag,
        "hyperparameters": vars(args),
        "base_lr": args.lr,
        "pretrained_lr": args.lr * args.pretrained_lr_mult,
        "image_size": Config.IMAGE_SIZE,
        "classes": Config.EMOTION_CLASSES,
        "best_val_macro_f1": best_f1,
        "final_eval_split": split,
        f"{split}_accuracy": final_res["accuracy"],
        f"{split}_macro_f1": final_res["macro_f1"],
        f"{split}_classification_report": report,
        f"{split}_confusion_matrix": cm.tolist(),
        "train_minutes": train_minutes,
        "history": history,
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[✓] Logged run to {log_path}")
    print(f"[✓] Best checkpoint: {save_path}")


if __name__ == "__main__":
    run()
