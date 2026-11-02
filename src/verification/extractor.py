import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from src.config import Config
from src.verification.classification.model import ClassificationFaceNet

class FaceEmbeddingExtractor:
    def __init__(self, checkpoint_path: str = Config.SUPERVISED_CHECKPOINT, device: torch.device = Config.DEVICE):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        num_classes = checkpoint["num_classes"]
        embedding_dim = checkpoint.get("embedding_dim", Config.EMBEDDING_DIM)
        
        self.model = ClassificationFaceNet(num_classes=num_classes, embedding_dim=embedding_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def get_embedding(self, image_input) -> np.ndarray:
        """
        Accepts a file path (str), PIL Image, or numpy array (H, W, C).
        Returns a 1D numpy array representing the L2-normalized embedding.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model.extract_embedding(tensor)
        return embedding.squeeze(0).cpu().numpy()
