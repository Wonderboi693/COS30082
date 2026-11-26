import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.config import Config
from src.emotion.dataset import get_emotion_loaders
from src.emotion.model import EmotionNet


def run():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation: ResNet34 (teacher) -> ResNet18 (student)"
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="checkpoints/best_emotion_resnet34.pt",
        help="Path to trained ResNet34 checkpoint",
    )
    parser.add_argument(
        "--student-backbone",
        default="resnet18",
        choices=["resnet18"],
        help="Student backbone (fixed to resnet18)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Distillation epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Student learning rate")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.EMOTION_BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Distillation temperature",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for distillation loss",
    )
    parser.add_argument("--tag", default="distill_r34_to_r18")
    parser.add_argument(
        "--save-path",
        default="checkpoints/best_emotion_distill_student.pt",
        help="Where to save best student checkpoint",
    )
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"[*] Device: {device}")
    print(f"[*] Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"[*] Student backbone: {args.student_backbone}")
    print(f"[*] Temperature: {args.temperature} | Alpha: {args.alpha}")

    train_loader, val_loader, test_loader = get_emotion_loaders(
        batch_size=args.batch_size
    )

    teacher = EmotionNet(
        num_classes=Config.NUM_EMOTIONS,
        pretrained_backbone=False,
        freeze_backbone=False,
        backbone="resnet34",
    ).to(device)

    ckpt = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = EmotionNet(
        num_classes=Config.NUM_EMOTIONS,
        pretrained_backbone=True,
        freeze_backbone=False,
        backbone=args.student_backbone,
    ).to(device)

    optimizer = optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for images, labels in tqdm(
            train_loader, desc=f"Distill Epoch {epoch}/{args.epochs}", leave=False
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            loss_hard = ce_criterion(student_logits, labels)

            soft_student = F.log_softmax(
                student_logits / args.temperature, dim=-1
            )
            soft_teacher = F.softmax(
                teacher_logits / args.temperature, dim=-1
            )
            loss_soft = (
                F.kl_div(soft_student, soft_teacher, reduction="batchmean")
                * (args.temperature ** 2)
            )

            loss = (1.0 - args.alpha) * loss_hard + args.alpha * loss_soft

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total_count += len(labels)

        scheduler.step()

        student.eval()
        val_correct = 0
        val_count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = student(images)
                preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_count += len(labels)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_acc = val_correct / val_count
        val_f1 = f1_score(
            all_targets, all_preds, average="macro", zero_division=0
        )

        avg_train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        print(
            f"  [Distill] Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss {avg_train_loss:.4f} (Acc {train_acc*100:.1f}%) | "
            f"Val Acc {val_acc*100:.2f}% | Val Macro F1 {val_f1*100:.2f}%"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "tag": args.tag,
                    "backbone": args.student_backbone,
                    "model_state": student.state_dict(),
                    "val_macro_f1": best_f1,
                    "val_accuracy": val_acc,
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                },
                save_path,
            )
            print(f"    --> Saved best student checkpoint ({save_path})")

    train_minutes = (time.time() - t0) / 60.0
    print(f"\n[✓] Distillation done in {train_minutes:.1f} min. Best val Macro F1: {best_f1*100:.2f}%")

    results_dir = Path(Config.PROJECT_ROOT) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"emotion_{args.tag}.json"
    log = {
        "tag": args.tag,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_backbone": args.student_backbone,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_macro_f1": best_f1,
        "train_minutes": train_minutes,
        "history": history,
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[✓] Log saved to {log_path}")
    print(f"[✓] Best student checkpoint: {save_path}")


if __name__ == "__main__":
    run()
