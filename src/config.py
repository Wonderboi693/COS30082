import os
import torch

class Config:
    # Resolves to /home/.../COS30082 (two levels up from src/config.py)
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)

    # Dataset paths matching your tree:
    # COS30082/data/raw/classification_data/
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
    CLASSIFICATION_DIR = os.path.join(DATA_ROOT, "classification_data")
    VERIFICATION_DIR = os.path.join(DATA_ROOT, "verification_data")

    # Checkpoint path at root: COS30082/checkpoints/
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    SUPERVISED_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_supervised.pth")
    METRIC_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_metric.pth")

    # Image specs (Kaggle 11-785 dataset)
    IMAGE_SIZE = (64, 64)
    IN_CHANNELS = 3

    # Embedding bottleneck dimension
    EMBEDDING_DIM = 256

    # Training parameters
    BATCH_SIZE = 128
    NUM_WORKERS = 8
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
