import os
import torch

class Config:
    # Project paths
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)

    # Each machine can override paths through environment variables.
    DATA_ROOT = os.path.expanduser(os.environ.get(
        "COS30082_DATA_ROOT", os.path.join(PROJECT_ROOT, "data", "raw")))
    CLASSIFICATION_DIR = os.path.expanduser(os.environ.get(
        "COS30082_CLASSIFICATION_DIR", os.path.join(DATA_ROOT, "classification_data")))
    VERIFICATION_DIR = os.path.expanduser(os.environ.get(
        "COS30082_VERIFICATION_DIR", os.path.join(DATA_ROOT, "verification_data")))

    # Model checkpoints
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    SUPERVISED_RUN_DIR = os.path.expanduser(os.environ.get(
        "COS30082_SUPERVISED_RUN_DIR", os.path.join(CHECKPOINT_DIR, "supervised")))
    SUPERVISED_CHECKPOINT = os.path.expanduser(os.environ.get(
        "COS30082_SUPERVISED_CHECKPOINT", os.path.join(SUPERVISED_RUN_DIR, "best.pt")))
    METRIC_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_metric.pth")

    # Supervised baseline settings
    IMAGE_SIZE = (224, 224)
    IN_CHANNELS = 3

    # Embedding bottleneck dimension
    EMBEDDING_DIM = 512
    ARCHITECTURE = "resnet50"

    # Training parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    BACKBONE_LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
