import torch.nn as nn
from src.config import config as cfg
from torchvision import models


class CompatibilityModel(nn.Module):
    """Custom model to measure compatibility between fashion products"""

    def __init__(self, hidden_dim=cfg.HIDDEN_DIM, emb_dim=cfg.EMBEDDING_DIM, dropout=cfg.DROPOUT):
        super(CompatibilityModel, self).__init__()
        # use resnet34 as base model
        self.create_base_model()
        # add 2 layers on top of base model
        self.embedding_layers = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def create_base_model(self):
        """Create base model as resnet 18 removing the last fc layer"""
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        # pass input through base and embedding layers
        x = self.base_model(x)
        x = self.embedding_layers(x)
        return x
