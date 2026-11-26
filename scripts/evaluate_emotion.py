import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import Config
from src.emotion.dataset import EmotionDataset, EXTENSIONS
from src.emotion.model import EmotionNet
from src.emotion.detector import EmotionDetector


def evaluate_with_dataloader(checkpoint_path, test_dir, device, batch_size=64):
    """Dùng DataLoader + transform eval (nhanh, dùng transform giống training)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt.get("classes", Config.EMOTION_CLASSES)

    model = EmotionNet(num_classes=len(classes), pretrained_backbone=False, backbone="resnet18")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = EmotionDataset(str(test_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=Config.NUM_WORKERS, pin_memory=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    return np.array(all_targets), np.array(all_preds), dataset.classes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate emotion model on test set.")
    parser.add_argument("--checkpoint", default=str(Config.EMOTION_CHECKPOINT),
                        help="Đường dẫn checkpoint (.pt)")
    parser.add_argument("--test-dir", default=None,
                        help="Thư mục test (mặc định: data/data_emotion/test)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu",
                        help="cpu hoặc cuda")
    parser.add_argument("--output-json", default=None,
                        help="Lưu kết quả ra file JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    test_dir = Path(args.test_dir) if args.test_dir else \
        Path(Config.EMOTION_DIR) / "test"

    if not test_dir.is_dir():
        print(f"[!] Không tìm thấy thư mục test: {test_dir}")
        sys.exit(1)

    if not Path(args.checkpoint).is_file():
        print(f"[!] Không tìm thấy checkpoint: {args.checkpoint}")
        sys.exit(1)

    print(f"[*] Checkpoint: {args.checkpoint}")
    print(f"[*] Test dir: {test_dir}")
    print(f"[*] Device: {device}")
    print()

    targets, preds, class_names = evaluate_with_dataloader(
        args.checkpoint, test_dir, device, args.batch_size)

    acc = float((targets == preds).mean())
    macro_f1 = float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
        targets, preds, average="macro", zero_division=0))

    print("=" * 60)
    print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ({len(targets)} ảnh)")
    print("=" * 60)
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Macro F1 : {macro_f1*100:.2f}%")
    print()

    report = classification_report(
        targets, preds,
        labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0, output_dict=True)

    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 52)
    for cls in class_names:
        if cls in report:
            r = report[cls]
            print(f"{cls:<12} {r['precision']:<10.3f} {r['recall']:<10.3f} "
                  f"{r['f1-score']:<10.3f} {r['support']:<10}")

    print()
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    print("      " + " ".join(f"{c[:4]:>4}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<6} " + " ".join(f"{v:>4}" for v in row))

    if args.output_json:
        out = {
            "checkpoint": args.checkpoint,
            "test_dir": str(test_dir),
            "device": str(device),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "num_samples": int(len(targets)),
            "classes": class_names,
            "predictions": preds.tolist(),
            "targets": targets.tolist(),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[✓] Đã lưu kết quả JSON vào: {args.output_json}")


if __name__ == "__main__":
    main()