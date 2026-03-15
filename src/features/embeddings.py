"""Feature extractors for similar and compatible product embeddings.

SimilarProductEmbedder: ResNet-18 features for similarity (same-category).
CompatibleProductEmbedder: CompatibilityModel features for complementarity.
"""

import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import tqdm
from torchvision.models import ResNet18_Weights, resnet18

from src.config import config as cfg
from src.config import get_simple_logger
from src.dataloader.data_loaders import (FashionCompleteTheLookDataloader,
                                         FashionProductSTLDataloader)
from src.models.compatibility_model import CompatibilityModel

logger = get_simple_logger(__name__)

class SimilarProductEmbedder:
    """Extract ResNet-18 features for similar-product (same-category) retrieval."""

    def __init__(self, device: torch.device | None = None):
        """Initialize embedder.

        Args:
            device: Device for inference. Defaults to cfg.device.
        """
        self.device = device or cfg.device

    def extract(
        self,
        data_loader,
        task_name: str = "similar_product",
        save_path: Path | None = None,
    ) -> torch.Tensor:
        """Extract features for all images in the data loader.

        Args:
            data_loader: PyTorch DataLoader yielding image batches.
            task_name: Name for cached pickle file.
            save_path: Override save path. Default: CACHED_EMBEDDINGS_DIR/{task_name}_embedding.pickle.

        Returns:
            Tensor of shape (N, 512) with features on CPU.
        """
        logger.info(f"You are using device: {self.device}")
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(self.device)
        resnet.fc = nn.Identity()
        resnet.eval()
        transform = torchvision.transforms.Resize((cfg.HEIGHT, cfg.WIDTH))
        all_features = []

        for batch in tqdm.tqdm(data_loader, desc=f"Extract {task_name}"):
            X = transform(batch)
            X = X.float().to(self.device)
            with torch.no_grad():
                batch_features = resnet(X)
                all_features.append(batch_features)

        all_features = torch.cat(all_features).to("cpu")
        out_path = save_path or (
            cfg.PACKAGE_ROOT
            / "features/cached_embeddings"
            / f"{task_name}_embedding.pickle"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(all_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        return all_features


class CompatibleProductEmbedder:
    """Extract CompatibilityModel features for compatible-product (cross-category) retrieval."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: torch.device | None = None,
    ):
        """Initialize embedder.

        Args:
            model_path: Path to checkpoint or directory. Default: latest in TRAINED_MODEL_DIR.
            device: Device for inference. Defaults to cfg.device.
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = device or cfg.device

    def extract(
        self,
        data_loader,
        task_name: str,
        save_path: Path | None = None,
    ) -> torch.Tensor:
        """Extract compatibility features for all images.

        Args:
            data_loader: PyTorch DataLoader yielding image batches.
            task_name: Name for cached pickle file.
            save_path: Override save path.

        Returns:
            Tensor of shape (N, embedding_dim) with features on CPU.
        """
        logger.info(f"You are using device: {self.device}")
        model = CompatibilityModel.from_pretrained(
            self.model_path or cfg.TRAINED_MODEL_DIR,
            device=self.device,
        )
        model.eval()
        transform = torchvision.transforms.Resize((cfg.HEIGHT, cfg.WIDTH))
        all_features = []

        for batch in tqdm.tqdm(data_loader, desc=f"Extract {task_name}"):
            X = transform(batch)
            X = X.float().to(self.device)
            with torch.no_grad():
                batch_features = model(X)
                all_features.append(batch_features)

        all_features = torch.cat(all_features).to("cpu")
        out_path = save_path or (
            cfg.PACKAGE_ROOT
            / "features/cached_embeddings"
            / f"{task_name}_embedding.pickle"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(all_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        return all_features


# --- Backward compatibility ---
class StyleEmbedding:
    """Legacy facade: delegates to SimilarProductEmbedder and CompatibleProductEmbedder."""

    def __init__(self):
        self._similar = SimilarProductEmbedder()
        self._compatible = CompatibleProductEmbedder()

    def similar_product_embedding(
        self, data_loader, task_name: str = "similar_product"
    ):
        """Extract similar-product features (ResNet-18)."""
        return self._similar.extract(data_loader, task_name=task_name)

    def compatible_product_embedding(self, data_loader, task_name: str):
        """Extract compatible-product features (CompatibilityModel)."""
        return self._compatible.extract(data_loader, task_name=task_name)


if __name__ == "__main__":
    similar = SimilarProductEmbedder()
    similar.extract(
        data_loader=FashionProductSTLDataloader().data_loader(),
        task_name="similar_product",
    )
    similar.extract(
        data_loader=FashionCompleteTheLookDataloader().single_data_loader(),
        task_name="similar_prod_CTL",
    )
    compatible = CompatibleProductEmbedder()
    compatible.extract(
        data_loader=FashionCompleteTheLookDataloader(
            image_type="test"
        ).single_data_loader(),
        task_name="compatible_product_test",
    )
    compatible.extract(
        data_loader=FashionCompleteTheLookDataloader().single_data_loader(),
        task_name="compatible_product",
    )
