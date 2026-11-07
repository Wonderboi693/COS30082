import os
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import Config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TripletFaceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform

        if not self.root_dir.is_dir():
            raise ValueError(f"Directory not found: {self.root_dir}")

        self.identity_to_files = defaultdict(list)
        self.all_identities = []
        self.samples = []  # List of (file_path, identity_name)

        identities = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for identity_dir in identities:
            files = [
                f for f in identity_dir.iterdir()
                if f.is_file() and f.suffix.lower() in EXTENSIONS
            ]
            # Must have at least 2 photos to form (Anchor, Positive) pairs
            if len(files) >= 2:
                name = identity_dir.name
                self.identity_to_files[name] = files
                self.all_identities.append(name)
                for f in files:
                    self.samples.append((f, name))

        print(f"[*] Triplet Dataset indexed: {len(self.samples)} samples across {len(self.all_identities)} identities.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 1. Anchor
        anchor_path, anchor_id = self.samples[index]

        # 2. Positive candidate (same identity, different file)
        pos_candidates = self.identity_to_files[anchor_id]
        positive_path = random.choice(pos_candidates)
        while positive_path == anchor_path and len(pos_candidates) > 1:
            positive_path = random.choice(pos_candidates)

        # 3. Negative candidate (different identity)
        neg_id = random.choice(self.all_identities)
        while neg_id == anchor_id:
            neg_id = random.choice(self.all_identities)
        negative_path = random.choice(self.identity_to_files[neg_id])

        # Load images
        try:
            with Image.open(anchor_path) as a_img, \
                 Image.open(positive_path) as p_img, \
                 Image.open(negative_path) as n_img:
                a_img = a_img.convert("RGB")
                p_img = p_img.convert("RGB")
                n_img = n_img.convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError(f"Error decoding triplet image: {error}") from error

        if self.transform:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        return a_img, p_img, n_img


def get_triplet_loaders(batch_size: int = Config.BATCH_SIZE, num_workers: int = Config.NUM_WORKERS):
    train_dir = os.path.join(Config.CLASSIFICATION_DIR, "train_data")
    val_dir = os.path.join(Config.CLASSIFICATION_DIR, "val_data")

    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = TripletFaceDataset(train_dir, transform=train_transform)
    val_dataset = TripletFaceDataset(val_dir, transform=val_transform)

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