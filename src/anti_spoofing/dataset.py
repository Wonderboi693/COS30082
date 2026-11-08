import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import Config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AntiSpoofDataset(Dataset):
    """
    Expects directory structure:
        dataset_dir/
            live/   (or real/, bona_fide/)  -> label 1.0
            spoof/  (or attack/, fake/)     -> label 0.0
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.samples = []

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Antispoof directory not found: {self.root_dir}")

        live_aliases = {"live", "real", "bona_fide"}
        spoof_aliases = {"spoof", "attack", "fake"}

        live_count, spoof_count = 0, 0

        for folder in self.root_dir.iterdir():
            if not folder.is_dir():
                continue

            folder_lower = folder.name.lower()
            if folder_lower in live_aliases:
                label = 1.0
            elif folder_lower in spoof_aliases:
                label = 0.0
            else:
                continue

            for file_path in folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in EXTENSIONS:
                    self.samples.append((file_path, label))
                    if label == 1.0:
                        live_count += 1
                    else:
                        spoof_count += 1

        print(f"[*] Loaded Anti-Spoof dataset from {self.root_dir.name}: "
              f"{live_count} Live, {spoof_count} Spoof ({len(self.samples)} total)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with Image.open(file_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def get_antispoof_loaders(
    data_dir: str = Config.ANTISPOOF_DIR,
    batch_size: int = Config.ANTISPOOF_BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS
):
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    # Perturbations simulating print resolution loss & screen re-capture
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = AntiSpoofDataset(str(train_dir), transform=train_transform)
    val_dataset = AntiSpoofDataset(str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader
