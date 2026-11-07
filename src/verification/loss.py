import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    """
    Custom Triplet Margin Loss:
    L(a, p, n) = max(0, ||a - p||_2^2 - ||a - n||_2^2 + margin)
    """
    def __init__(self, margin: float = 0.2, p: int = 2, swap: bool = False):
        super().__init__()
        self.margin = margin
        self.p = p
        self.swap = swap

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Compute pair distances
        dist_pos = F.pairwise_distance(anchor, positive, p=self.p)
        dist_neg = F.pairwise_distance(anchor, negative, p=self.p)

        if self.swap:
            dist_pos_neg = F.pairwise_distance(positive, negative, p=self.p)
            dist_neg = torch.min(dist_neg, dist_pos_neg)

        # L = relu(d_pos - d_neg + alpha)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()