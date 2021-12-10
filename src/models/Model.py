import torch.nn as nn
from torchvision import models


class CompatibilityModel(nn.Module):
    """Custom model to measure compatibility between fashion products"""

    def __init__(self, hidden_dim=256, emb_dim=128, dropout=0.4):
        super(CompatibilityModel, self).__init__()
        # use resnet34 as base model
        self.base_model = models.resnet18(pretrained=True)

        # add 2 layers on top of base model
        self.embedding_layers = nn.Sequential(
            nn.Linear(1000, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(-1, 1000)
        x = self.embedding_layers(x)
        return x
