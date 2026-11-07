import argparse
import os
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

from src.config import Config
from src.verification.loss import TripletMarginLoss
from src.verification.metric_learning.dataset import get_triplet_loaders
from src.verification.metric_learning.model import MetricFaceNet

torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss, total_pos_d, total_neg_d, count = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Triplet Train", leave=False)
    for a, p, n in pbar:
        a = a.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        n = n.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            loss = criterion(emb_a, emb_p, emb_n)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        b_size = a.size(0)
        with torch.no_grad():
            d_pos = torch.norm(emb_a - emb_p, p=2, dim=1).mean().item()
            d_neg = torch.norm(emb_a - emb_n, p=2, dim=1).mean().item()

        total_loss += loss.item() * b_size
        total_pos_d += d_pos * b_size
        total_neg_d += d_neg * b_size
        count += b_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "d_pos": f"{d_pos:.3f}",
            "d_neg": f"{d_neg:.3f}",
            "gap": f"{(d_neg - d_pos):.3f}"
        })

    return total_loss / count, total_pos_d / count, total_neg_d / count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_pos_d, total_neg_d, count = 0.0, 0.0, 0.0, 0

    for a, p, n in loader:
        a = a.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        n = n.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            loss = criterion(emb_a, emb_p, emb_n)

        b_size = a.size(0)
        d_pos = torch.norm(emb_a - emb_p, p=2, dim=1).mean().item()
        d_neg = torch.norm(emb_a - emb_n, p=2, dim=1).mean().item()

        total_loss += loss.item() * b_size
        total_pos_d += d_pos * b_size
        total_neg_d += d_neg * b_size
        count += b_size

    return total_loss / count, total_pos_d / count, total_neg_d / count


def save_checkpoint(path: Path, checkpoint: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def run():
    parser = argparse.ArgumentParser(description="Train metric learning pipeline using Triplet Loss.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--pretrained-ckpt", default=Config.SUPERVISED_CHECKPOINT)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"[*] Starting Triplet Metric Learning on: {device}")

    train_loader, val_loader = get_triplet_loaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS)
    model = MetricFaceNet(embedding_dim=Config.EMBEDDING_DIM).to(device)

    # Load backbone weights from US 1.1
    if Path(args.pretrained_ckpt).is_file() and not args.resume:
        model.load_pretrained_backbone(args.pretrained_ckpt)
    elif not args.resume:
        print("[!] Pretrained checkpoint not found. Starting from scratch.")

    criterion = TripletMarginLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    best_val_loss = float("inf")
    output_dir = Path(Config.CHECKPOINT_DIR)

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[*] Resumed metric training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_pos, tr_neg = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs
        )
        val_loss, val_pos, val_neg = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        tr_gap = tr_neg - tr_pos
        val_gap = val_neg - val_pos
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} (Gap: {tr_gap:.3f}) | "
              f"Val Loss: {val_loss:.4f} (Gap: {val_gap:.3f} | d_pos: {val_pos:.3f}, d_neg: {val_neg:.3f})")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": {"embedding_dim": Config.EMBEDDING_DIM, "architecture": "resnet18"},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "image_size": Config.IMAGE_SIZE[0],
            "kind": "embeddings"
        }

        save_checkpoint(output_dir / "last_metric.pt", checkpoint)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint["best_val_loss"] = best_val_loss
            save_checkpoint(Path(Config.METRIC_CHECKPOINT), checkpoint)
            print(f"    --> Saved best metric checkpoint to {Config.METRIC_CHECKPOINT}")


if __name__ == "__main__":
    run()