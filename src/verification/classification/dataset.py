import csv
import hashlib
import json
import random
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train_data", "val": "val_data", "test": "test_data"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    person_id: str
    label: int
    split: str
    content_sha256: str = ""


def get_preprocessing_config(image_size=224):
    """Settings saved with a model so inference uses its training preprocessing."""
    return {"image_size": image_size, "mean": list(MEAN), "std": list(STD)}


def build_transform(image_size=224, training=False, mean=MEAN, std=STD):
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("RGB normalization requires three means and three positive standard deviations")
    operations = [transforms.Resize((image_size, image_size), antialias=True)]
    if training:
        operations += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        ]
    operations += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(operations)


def load_samples(data_root, manifest=None, check_duplicates=True):
    """Read the brief's split folders, or a shared relative-path CSV manifest.

    Training determines the label map. Evaluation can contain a subset of known
    classes, but an unknown identity is an error for closed-set classification.
    """
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    records = []
    if manifest:
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"image_path", "person_id", "split"} <= set(reader.fieldnames or []):
                raise ValueError("Manifest needs image_path, person_id, and split columns")
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
                        for path in sorted(person.rglob("*"))
                        if path.is_file() and path.suffix.lower() in EXTENSIONS
                    )
    class_names = sorted({identity for _, identity, split, _ in records if split == "train"})
    if len(class_names) < 2:
        raise ValueError("Training requires at least two nonempty identity classes")
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    samples = {split: [] for split in SPLITS}
    seen_paths, seen_contents = {}, {}

    def inspect(record):
        path, identity, split, supplied_label = record
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes the dataset root: {path}")
        if split not in samples:
            raise ValueError(f"Unknown split {split!r}; use train, val, or test")
        if identity not in class_to_idx:
            raise ValueError(f"Unknown evaluation identity {identity!r}; not present in training")
        label = class_to_idx[identity]
        if supplied_label not in (None, "") and int(supplied_label) != label:
            raise ValueError(f"Inconsistent label for {identity}: expected {label}, got {supplied_label}")
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Missing or unsupported image: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return Sample(path, identity, label, split, digest)

    # Small-file reads benefit from parallel I/O, particularly on Windows.
    # Bound each batch of futures instead of enqueueing half a million at once.
    with ThreadPoolExecutor(max_workers=8) as executor:
        for start in range(0, len(records), 4096):
            inspected = executor.map(inspect, records[start:start + 4096])
            for sample in inspected:
                path, identity, split = sample.path, sample.person_id, sample.split
                if path in seen_paths:
                    raise ValueError(f"Image assigned more than once: {path}")
                seen_paths[path] = split
                if check_duplicates:
                    previous = seen_contents.get(sample.content_sha256)
                    if previous and previous[:2] != (split, identity):
                        raise ValueError(f"Duplicate photo crosses splits or identities: {previous[2]} and {path}")
                    seen_contents[sample.content_sha256] = (split, identity, path)
                samples[split].append(sample)
            if len(records) >= 4096:
                print(f"Checked identity images: {min(start + 4096, len(records))}/{len(records)}", flush=True)
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
            with Image.open(sample.path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                tensor = self.transform(image)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode face image {sample.path}: {error}") from error
        return tensor, sample.label


def seed_worker(worker_id):
    # Torch seeds each worker; keep Python/NumPy augmentation reproducible too.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(data_root, batch_size=32, image_size=224, num_workers=0,
                  seed=42, manifest=None, check_duplicates=True, pin_memory=False):
    if batch_size < 2 or num_workers < 0:
        raise ValueError("batch_size must be >= 2 and num_workers must be >= 0")
    samples, mapping = load_samples(data_root, manifest, check_duplicates)
    loaders = {}
    for split, items in samples.items():
        dataset = IdentityDataset(items, build_transform(image_size, split == "train"), mapping)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=split == "train",
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
        )
    return loaders, mapping


def dataset_fingerprint(loaders, data_root):
    """Identify split assignments AND image contents without absolute host paths."""
    root = Path(data_root).resolve()
    digest = hashlib.sha256()
    for split in SPLITS:
        for sample in sorted(loaders[split].dataset.samples,
                             key=lambda item: item.path.relative_to(root).as_posix()):
            entry = [split, sample.person_id, sample.label, sample.path.relative_to(root).as_posix(),
                     sample.content_sha256 or hashlib.sha256(sample.path.read_bytes()).hexdigest()]
            digest.update(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def get_classification_loaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS,
                               data_root=Config.CLASSIFICATION_DIR, image_size=Config.IMAGE_SIZE[0],
                               seed=42, manifest=None):
    """Return train, validation and test loaders with one shared identity mapping."""
    loaders, mapping = build_loaders(
        data_root, batch_size, image_size, num_workers, seed, manifest,
        pin_memory=torch.cuda.is_available(),
    )
    return loaders["train"], loaders["val"], loaders["test"], len(mapping)
