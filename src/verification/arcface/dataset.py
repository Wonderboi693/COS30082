"""Identity-folder datasets, transforms and loaders for the softmax / ArcFace embedding models.

Layout: Config.CLASSIFICATION_DIR/{train_data,val_data}/<identity>/<image>.
"""
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.config import Config

IMG_SIZE = Config.ARCFACE_IMAGE_SIZE
MEAN, STD = Config.ARCFACE_MEAN, Config.ARCFACE_STD
TRAIN_DIR = Path(Config.CLASSIFICATION_DIR) / "train_data"
VAL_DIR = Path(Config.CLASSIFICATION_DIR) / "val_data"
TEST_DIR = Path(Config.CLASSIFICATION_DIR) / "test_data"
TEST_LIST = Path(Config.CLASSIFICATION_DIR) / "classification_test.txt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_transform(train: bool):
    """Train: flip + small crop + colour jitter. Eval: deterministic resize only."""
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_rgb(path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def class_mapping(root: Path) -> dict:
    """Folder name -> index, sorted numerically when names are integers."""
    names = [p.name for p in Path(root).iterdir() if p.is_dir()]
    names.sort(key=lambda n: (0, int(n)) if n.isdigit() else (1, n))
    return {name: i for i, name in enumerate(names)}


class IdentityFolder(Dataset):
    """root/<identity>/<image>. Uses the given class_to_idx so splits share labels."""

    def __init__(self, root, class_to_idx=None, transform=None, max_per_class=None, seed=0):
        self.root = Path(root)
        self.class_to_idx = class_to_idx or class_mapping(self.root)
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
        self.transform = transform
        rng = random.Random(seed)
        self.samples = []
        for name, idx in self.class_to_idx.items():
            folder = self.root / name
            if not folder.is_dir():
                continue
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS)
            if max_per_class:
                rng.shuffle(files)
                files = sorted(files[:max_per_class])
            self.samples += [(f, idx) for f in files]
        unknown = {p.name for p in self.root.iterdir() if p.is_dir()} - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"{len(unknown)} folders in {root} are not in the class mapping")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_rgb(path)
        return (self.transform(img) if self.transform else img), label


class ImageList(Dataset):
    """Flat list of image paths (verification images, unlabelled test images)."""

    def __init__(self, paths, transform=None):
        self.paths = [Path(p) for p in paths]
        self.transform = transform or build_transform(train=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_rgb(self.paths[i])), i


def build_datasets(train_dir=TRAIN_DIR, val_dir=VAL_DIR, max_per_class=None):
    train = IdentityFolder(train_dir, transform=build_transform(True), max_per_class=max_per_class)
    val = IdentityFolder(val_dir, class_to_idx=train.class_to_idx, transform=build_transform(False))
    return train, val


def build_loaders(batch_size=128, workers=4, train_dir=TRAIN_DIR, val_dir=VAL_DIR, max_per_class=None):
    train, val = build_datasets(train_dir, val_dir, max_per_class)
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader
