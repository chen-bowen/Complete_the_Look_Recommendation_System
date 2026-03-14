"""Compatibility model for fashion product embeddings.

Uses ResNet-18 as backbone with a learned embedding head. Supports loading
from checkpoint via from_pretrained().
"""

import pathlib
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import config as cfg


class CompatibilityModel(nn.Module):
    """Model to measure visual compatibility between fashion products.

    Backbone: ResNet-18 (512-dim) + linear head -> embedding_dim.
    """

    def __init__(
        self,
        hidden_dim: int = cfg.HIDDEN_DIM,
        emb_dim: int = cfg.EMBEDDING_DIM,
        dropout: float = cfg.DROPOUT,
    ):
        super(CompatibilityModel, self).__init__()
        self._create_base_model()
        self.embedding_layers = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def _create_base_model(self) -> None:
        """Create base model as ResNet-18, removing the final classification layer."""
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone -> embedding head."""
        x = self.base_model(x)
        x = self.embedding_layers(x)
        return x

    @classmethod
    def from_pretrained(
        cls,
        path: str | pathlib.Path,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "CompatibilityModel":
        """Load model from saved checkpoint.

        Parameters
        ----------
        path : str | Path
            Path to .pth file or directory containing checkpoint.
        device : torch.device | None
            Device to load on. Defaults to cfg.device.
        **kwargs
            Passed to CompatibilityModel.__init__ if building from scratch.

        Returns
        -------
        CompatibilityModel
        """
        path = pathlib.Path(path)
        if path.is_dir():
            checkpoints = sorted(path.glob("trained_compatibility_model_epoch*.pth"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {path}")
            path = checkpoints[-1]
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = cls(**kwargs)
        model.load_state_dict(state_dict, strict=True)
        if device is not None:
            model = model.to(device)
        return model
