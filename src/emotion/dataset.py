from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.config import Config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EmotionDataset(Dataset):
    """
    Expects directory structure:
        split_dir/
            Angry/  Disgust/  Fear/  Happy/  Neutral/  Sad/  Surprise/
    Folder names are matched case-insensitively against Config.EMOTION_CLASSES.
    Raises an error if any expected class folder is missing, so a mis-named folder
    can never silently drop a class from training.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.classes = Config.EMOTION_CLASSES
        self.class_to_idx = {c.lower(): i for i, c in enumerate(self.classes)}
        self.samples = []

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Emotion dataset directory not found: {self.root_dir}")

        found = {f.name.lower(): f for f in self.root_dir.iterdir() if f.is_dir()}
        missing = [c for c in self.classes if c.lower() not in found]
        if missing:
            raise ValueError(
                f"Missing class folders in {self.root_dir}: {missing}. "
                f"Found folders: {sorted(found)}. Rename them or edit Config.EMOTION_CLASSES."
            )
        extra = sorted(set(found) - set(self.class_to_idx))
        if extra:
            print(f"[!] Ignoring unexpected folders in {self.root_dir.name}: {extra}")

        for key, folder in found.items():
            if key not in self.class_to_idx:
                continue
            label = self.class_to_idx[key]
            for file_path in folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in EXTENSIONS:
                    self.samples.append((file_path, label))

        counts = np.bincount([lbl for _, lbl in self.samples], minlength=len(self.classes))
        print(f"[*] Loaded {len(self.samples)} images from {self.root_dir.name} "
              f"| per class: {dict(zip(self.classes, counts.tolist()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with Image.open(file_path) as img:
            img = img.convert("RGB")  # grayscale -> 3 channels for the ResNet stem

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def _make_loader(dataset, batch_size, num_workers, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def get_emotion_loaders(
    data_dir: str = None,
    batch_size: int = Config.EMOTION_BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    val_fraction: float = 0.15,
    seed: int = 42,
):
    """
    Returns (train_loader, val_loader, test_loader).

    - If data_dir/val exists it is used as the validation set.
    - Otherwise a stratified `val_fraction` of data_dir/train is held out for validation
      (the Balanced RAF-DB download only has train/ and test/).
    - data_dir/test is the held-out test set: use it once, after training.
      If it does not exist, test_loader is None.
    """
    if data_dir is None:
        data_dir = Config.EMOTION_DIR
    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if val_dir.is_dir():
        train_dataset = EmotionDataset(str(train_dir), transform=train_transform)
        val_dataset = EmotionDataset(str(val_dir), transform=eval_transform)
    else:
        print(f"[*] No val/ folder: holding out {val_fraction:.0%} of train/ for validation")
        full_train = EmotionDataset(str(train_dir), transform=train_transform)  # augmented
        full_val = EmotionDataset(str(train_dir), transform=eval_transform)     # same images, no augmentation
        labels = [lbl for _, lbl in full_train.samples]
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)), test_size=val_fraction, stratify=labels, random_state=seed
        )
        train_dataset = Subset(full_train, train_idx)
        val_dataset = Subset(full_val, val_idx)
        print("[!] Note: if the 'balanced' dataset contains augmented copies of the same source image, "
              "copies can land in both train and val and inflate validation scores. "
              "Check file names (e.g. an '_aug' suffix) and group copies together if so.")

    train_loader = _make_loader(train_dataset, batch_size, num_workers, shuffle=True, drop_last=True)
    val_loader = _make_loader(val_dataset, batch_size, num_workers, shuffle=False, drop_last=False)

    if test_dir.is_dir():
        test_dataset = EmotionDataset(str(test_dir), transform=eval_transform)
        test_loader = _make_loader(test_dataset, batch_size, num_workers, shuffle=False, drop_last=False)
    else:
        print(f"[!] No test/ folder in {data_dir}: there is no held-out test split. "
              "train.py will report the final metrics on the validation split instead.")
        test_loader = None

    return train_loader, val_loader, test_loader