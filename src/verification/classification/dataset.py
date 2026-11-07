import csv
import random
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train_data", "val": "val_data", "test": "test_data"}
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class Sample:
    path: Path
    person_id: str
    label: int
    split: str


def get_preprocessing_config(image_size=64):
    return {"image_size": image_size, "mean": list(MEAN), "std": list(STD)}


def build_transform(image_size=64, training=False, mean=MEAN, std=STD):
    operations = [transforms.Resize((image_size, image_size), antialias=True)]
    if training:
        operations += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    operations += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(operations)


def load_samples(data_root, manifest=None, check_duplicates=False):
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    records = []

    if manifest:
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append((root / row["image_path"], row["person_id"], row["split"], row.get("label")))
    else:
        for split, directory in SPLITS.items():
            split_root = root / directory
            if not split_root.is_dir():
                raise ValueError(f"Missing split folder: {split_root}")
            for person in sorted(split_root.iterdir()):
                if person.is_dir():
                    records.extend(
                        (path, person.name, split, None)
                        for path in sorted(person.iterdir())
                        if path.is_file() and path.suffix.lower() in EXTENSIONS
                    )

    class_names = sorted({identity for _, identity, split, _ in records if split == "train"})
    if len(class_names) < 2:
        raise ValueError("Training requires at least two nonempty identity classes")
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    samples = {split: [] for split in SPLITS}

    for path, identity, split, _ in records:
        if identity in class_to_idx:
            samples[split].append(Sample(path, identity, class_to_idx[identity], split))

    for split, items in samples.items():
        if not items:
            raise ValueError(f"The {split} split has no images")
    return samples, class_to_idx


class IdentityDataset(Dataset):
    def __init__(self, samples, transform, class_to_idx=None):
        self.class_to_idx = dict(class_to_idx or {})
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode face image {sample.path}: {error}") from error
        return tensor, sample.label


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(data_root, batch_size=Config.BATCH_SIZE, image_size=Config.IMAGE_SIZE[0],
                  num_workers=Config.NUM_WORKERS, seed=42, manifest=None,
                  check_duplicates=False, pin_memory=True):
    samples, mapping = load_samples(data_root, manifest, check_duplicates)
    loaders = {}
    for split, items in samples.items():
        dataset = IdentityDataset(items, build_transform(image_size, split == "train"), mapping)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )
    return loaders, mapping


def get_classification_loaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS,
                               data_root=Config.CLASSIFICATION_DIR, image_size=Config.IMAGE_SIZE[0],
                               seed=42, manifest=None):
    loaders, mapping = build_loaders(
        data_root, batch_size, image_size, num_workers, seed, manifest,
        pin_memory=torch.cuda.is_available(),
    )
    return loaders["train"], loaders["val"], loaders["test"], len(mapping)