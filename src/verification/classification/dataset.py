import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import Config

def get_classification_loaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    train_dir = os.path.join(Config.CLASSIFICATION_DIR, "train_data")
    val_dir   = os.path.join(Config.CLASSIFICATION_DIR, "val_data")
    test_dir  = os.path.join(Config.CLASSIFICATION_DIR, "test_data")

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Deterministic transform for validation/testing
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set   = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_set  = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    num_classes = len(train_set.classes)
    return train_loader, val_loader, test_loader, num_classes
