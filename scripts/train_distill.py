"""
Knowledge Distillation training for Emotion Recognition.
Teacher: ResNet-34 (da train xong, dong bang)
Student: ResNet-18 (duoc duong du tu teacher)

Loss = alpha * CE(student, hard_label) + (1-alpha) * T^2 * KL(soft_teacher, soft_student)
"""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import Config
from src.emotion.dataset import get_emotion_loaders
from src.emotion.model import EmotionNet

torch.backends.cudnn.benchmark = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distillation_loss(student_logits, teacher_logits, labels, alpha=0.7, T=4.0):
    hard_loss = F.cross_entropy(student_logits, labels)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def train():
    parser = argparse.ArgumentParser(description="Knowledge Distillation: ResNet-34 -> ResNet-18")
    parser.add_argument("--teacher-ckpt", required=True, help="Checkpoint teacher (ResNet-34)")
    parser.add_argument("--student-backbone", default="resnet18", choices=["resnet18"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7, help="Trong so hard label")
    parser.add_argument("--T", type=float, default=4.0, help="Temperature distillation")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--tag", default="distill_r34_to_r18")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"

    print(f"[*] Device: {device} | AMP: {use_amp}")
    print(f"[*] Teacher: {args.teacher_ckpt}")
    print(f"[*] Student: {args.student_backbone}")
    print(f"[*] Alpha: {args.alpha} | Temperature: {args.T}")

    # ==================== LOAD TEACHER ====================
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher_classes = teacher_ckpt.get("classes", Config.EMOTION_CLASSES)
    teacher_backbone = teacher_ckpt.get("backbone", "resnet34")

    teacher = EmotionNet(num_classes=len(teacher_classes), pretrained_backbone=False, backbone=teacher_backbone)
    teacher.load_state_dict(teacher_ckpt["model_state"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[✓] Teacher loaded: {teacher_backbone}, classes={teacher_classes}")

    train_labels = [lbl for _, lbl in train_loader.dataset.samples]
    counts = np.bincount(train_labels, minlength=Config.NUM_EMOTIONS)
    weights = 1.0 / (counts.astype(np.float64) + 1e-6)
    weights = weights / weights.sum() * Config.NUM_EMOTIONS
    class_weight = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[*] Class weights: " + ", ".join(f"{c}={w:.3f}" for c, w in zip(Config.EMOTION_CLASSES, weights)))

    # ==================== STUDENT ====================
    student = EmotionNet(num_classes=Config.NUM_EMOTIONS, pretrained_backbone=True, backbone=args.student_backbone)
    student.to(device).train()

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_path = Path(Config.CHECKPOINT_DIR) / f"emotion_distill_{args.tag}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        student.train()
        teacher.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Distill Epoch {epoch:02d}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                with torch.no_grad():
                    teacher_logits = teacher(images)
                student_logits = student(images)
                loss = distillation_loss(student_logits, teacher_logits, labels, alpha=args.alpha, T=args.T)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            total_correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total_count += len(labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        # ==================== VALIDATION ====================
        student.eval()
        val_loss = 0.0
        val_correct = 0
        val_count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = student(images)
                    loss = F.cross_entropy(logits, labels, weight=class_weight)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_count += len(labels)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / val_count
        val_acc = val_correct / val_count
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

        print(
            f"  Epoch {epoch:02d} | Train Loss {train_loss:.4f} (Acc {train_acc*100:.2f}%) | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.2f}% | Val Macro F1 {val_f1*100:.2f}%"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "backbone": args.student_backbone,
                "teacher": args.teacher_ckpt,
                "model_state": student.state_dict(),
                "classes": Config.EMOTION_CLASSES,
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "distill_alpha": args.alpha,
                "distill_T": args.T,
            }, save_path)
            print(f"    --> Saved best checkpoint (Macro F1 {best_val_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    --> Early stopping at epoch {epoch}")
                break

    print(f"\n[✓] Distillation done. Best val Macro F1: {best_val_f1*100:.2f}%")
    print(f"[✓] Best checkpoint: {save_path}")


if __name__ == "__main__":
    train()