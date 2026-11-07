import os
import torch


class Config:
    # Project paths
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)

    # Dataset paths
    DATA_ROOT = os.path.expanduser(os.environ.get(
        "COS30082_DATA_ROOT", os.path.join(PROJECT_ROOT, "data", "raw")))
    CLASSIFICATION_DIR = os.path.expanduser(os.environ.get(
        "COS30082_CLASSIFICATION_DIR", os.path.join(DATA_ROOT, "classification_data")))
    VERIFICATION_DIR = os.path.expanduser(os.environ.get(
        "COS30082_VERIFICATION_DIR", os.path.join(DATA_ROOT, "verification_data")))

    # Checkpoint management
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    SUPERVISED_RUN_DIR = os.path.expanduser(os.environ.get(
        "COS30082_SUPERVISED_RUN_DIR", os.path.join(CHECKPOINT_DIR, "supervised")))
    SUPERVISED_CHECKPOINT = os.path.expanduser(os.environ.get(
        "COS30082_SUPERVISED_CHECKPOINT", os.path.join(CHECKPOINT_DIR, "best.pt")))
    METRIC_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_metric.pth")

    # Native dataset dimensions (CMU 11-785)
    IMAGE_SIZE = (64, 64)
    IN_CHANNELS = 3

    # Architecture & embedding specifications
    ARCHITECTURE = "resnet18"
    EMBEDDING_DIM = 256

    # Training hyperparameters
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    BACKBONE_LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Compute device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")