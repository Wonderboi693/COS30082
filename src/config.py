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

    # ----------------------------------------------------
    # Anti-Spoofing / Liveness Detection (User Story 2.1)
    # ----------------------------------------------------
    ANTISPOOF_DIR = os.path.expanduser(
        os.environ.get(
            "COS30082_ANTISPOOF_DIR",
            os.path.join(PROJECT_ROOT, "data", "raw", "anti-spoofing_data")
        )
    )
    ANTISPOOF_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_antispoof.pt")
    
    LIVENESS_THRESHOLD = 0.65
    ANTISPOOF_BATCH_SIZE = 32
    ANTISPOOF_EPOCHS = 20
    ANTISPOOF_LR = 1e-4

    # ----------------------------------------------------
    # Emotion Recognition Module
    # ----------------------------------------------------
    EMOTION_DIR = os.path.expanduser(
        os.environ.get(
            "COS30082_EMOTION_DIR",
            os.path.join(PROJECT_ROOT, "data", "data_emotion")
        )
    )
    EMOTION_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_emotion.pt")
    
    EMOTION_CLASSES = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise"
    ]
    NUM_EMOTIONS = len(EMOTION_CLASSES)
    EMOTION_BATCH_SIZE = 64
    EMOTION_EPOCHS = 25
    EMOTION_LR = 3e-4

    # ----------------------------------------------------
    # Classification Embeddings: Softmax baseline + ArcFace 
    # ----------------------------------------------------
    ARCFACE_IMAGE_SIZE = 112          # 64x64 crops upsampled for the ImageNet-initialised ResNet18
    ARCFACE_EMBEDDING_DIM = 512
    ARCFACE_MEAN = (0.485, 0.456, 0.406)
    ARCFACE_STD = (0.229, 0.224, 0.225)
    ARCFACE_RUN_DIR = os.path.join(CHECKPOINT_DIR, "arcface_runs")   # per-run history + resume state
    SOFTMAX_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_softmax.pt")
    ARCFACE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_arcface.pt")

    ARCFACE_BATCH_SIZE = 128
    ARCFACE_LR = 0.1                  # neck + head; backbone uses lr * 0.1
    SOFTMAX_EPOCHS = 40
    ARCFACE_EPOCHS = 60
    ARCFACE_SCALE = 32.0
    ARCFACE_MARGIN = 0.35

    # Cosine thresholds at the EER point on verification_pairs_val.txt
    SOFTMAX_THRESHOLD = 0.2007        # val AUC 0.8994
    ARCFACE_THRESHOLD = 0.2391        # val AUC 0.8729
