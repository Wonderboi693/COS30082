from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import Config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EmotionDataset(Dataset):
    """
    Expects directory structure:
        dataset_dir/
            train/ (or val/)
                Angry/
                Disgust/
                Fear/
                Happy/
                Neutral/
                Sad/
                Surprise/
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.classes = Config.EMOTION_CLASSES
        self.class_to_idx = {cls_name.lower(): idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Emotion dataset directory not found: {self.root_dir}")

        for folder in self.root_dir.iterdir():
            if not folder.is_dir():
                continue

            folder_key = folder.name.lower()
            if folder_key not in self.class_to_idx:
                continue

            label = self.class_to_idx[folder_key]
            for file_path in folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in EXTENSIONS:
                    self.samples.append((file_path, label))

        print(f"[*] Loaded {len(self.samples)} images from {self.root_dir.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with Image.open(file_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def get_emotion_loaders(
    data_dir: str = None,
    batch_size: int = Config.EMOTION_BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS
):
    if data_dir is None:
        data_dir = Config.EMOTION_DIR

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = EmotionDataset(str(train_dir), transform=train_transform)
    val_dataset = EmotionDataset(str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader