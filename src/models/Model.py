import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CompatibilityModel(nn.Module):
    """Custom model to measure compatibility between fashion products"""

    def __init__(self, hidden_dim=256, emb_dim=128, batch_norm_features=128):
        super(CompatibilityModel, self).__init__()
        # use resnet34 as base model
        self.base_model = models.resnet34(pretrained=True)

        # add 2 layers on top of base model
        self.embedding_layers = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm2d(num_features=batch_norm_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(-1, 512)
        x = self.embedding_layers(x)
        return x
