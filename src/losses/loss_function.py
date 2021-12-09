import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Loss function for triplet anchor, positive and negative samples"""

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def euclidean_distance(x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Triplet loss (average)
        L = max(dp - dn + m, 0)
        """
        dp = self.euclidean_distance(anchor, positive)
        dn = self.euclidean_distance(anchor, negative)
        losses = F.relu(dp - dn + self.margin)

        return losses.mean()
