"""Recommenders for similar and compatible fashion products.

SimilarProductRecommender: Same-category similarity (ResNet embeddings).
CompatibleProductRecommender: Cross-category compatibility (CompatibilityModel).
"""

import pickle
from pathlib import Path

import pandas as pd

from src.config import config as cfg
from src.dataset.Dataloader import (
    FashionCompleteTheLookDataloader,
    FashionProductSTLDataloader,
)
from src.utils.similarity import calculate_similarity


class SimilarProductRecommender:
    """Recommend similar products (same category) by ResNet embedding similarity."""

    def __init__(
        self,
        embedding_path: Path | str | None = None,
        task_name: str = "similar_product",
    ):
        """Initialize recommender.

        Args:
            embedding_path: Path to pickle file. Default: PACKAGE_ROOT/features/cached_embeddings/{task_name}_embedding.pickle.
            task_name: Task name for default path.
        """
        self.embedding_path = Path(
            embedding_path
            or cfg.PACKAGE_ROOT
            / "features/cached_embeddings"
            / f"{task_name}_embedding.pickle"
        )
        self._features: object | None = None
        self._metadata: pd.DataFrame | None = None

    def _load(self) -> None:
        """Lazy-load embeddings and metadata."""
        if self._features is not None:
            return
        with open(self.embedding_path, "rb") as f:
            self._features = pickle.load(f)
        data_loader = FashionProductSTLDataloader().data_loader()
        self._metadata = data_loader.dataset.metadata

    def recommend(self, product_id: int, top_n: int = 5) -> dict:
        """Return top-n similar products for the given product ID.

        Args:
            product_id: The product ID to query.
            top_n: Number of recommendations to return.

        Returns:
            Dict with keys "input_product" and "recommended_products".
        """
        self._load()
        product_metadata = self._metadata[
            self._metadata["product_id"] == product_id
        ].to_dict(orient="records")[0]
        product_feature_vec = self._features[product_id, :]
        similarity_score = calculate_similarity(
            product_feature_vec, self._features, "cosine"
        )
        product_category = product_metadata["product_type"]
        self._metadata = self._metadata.copy()
        self._metadata["similarity_score"] = similarity_score.cpu()

        recommended = (
            self._metadata[
                (self._metadata["product_type"] == product_category)
                & (self._metadata["similarity_score"] != 1)
            ]
            .sort_values(by="similarity_score", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
        )
        return {
            "input_product": product_metadata,
            "recommended_products": recommended,
        }


class CompatibleProductRecommender:
    """Recommend compatible products (cross-category) by CompatibilityModel embedding similarity."""

    def __init__(
        self,
        embedding_path: Path | str | None = None,
        task_name: str = "compatible_product",
    ):
        """Initialize recommender.

        Args:
            embedding_path: Path to pickle file.
            task_name: Task name for default path.
        """
        self.embedding_path = Path(
            embedding_path
            or cfg.PACKAGE_ROOT
            / "features/cached_embeddings"
            / f"{task_name}_embedding.pickle"
        )
        self._features: object | None = None
        self._metadata: pd.DataFrame | None = None

    def _load(self) -> None:
        """Lazy-load embeddings and metadata."""
        if self._features is not None:
            return
        with open(self.embedding_path, "rb") as f:
            self._features = pickle.load(f)
        data_loader = FashionCompleteTheLookDataloader().single_data_loader()
        self._metadata = data_loader.dataset.metadata[
            ["product_id", "image_single_signature", "product_type", "image_path"]
        ]

    def recommend(self, product_id: int, top_n: int = 5) -> dict:
        """Return top-n compatible products for the given product ID.

        Args:
            product_id: The product ID to query.
            top_n: Number of recommendations to return.

        Returns:
            Dict with keys "input_product" and "recommended_compatible_products".
        """
        self._load()
        product_metadata = self._metadata[
            self._metadata["product_id"] == product_id
        ].to_dict(orient="records")[0]
        product_feature_vec = self._features[product_id, :]
        compatibility_score = calculate_similarity(
            product_feature_vec, self._features, "cosine"
        )
        input_category = product_metadata["product_type"]
        self._metadata = self._metadata.copy()
        self._metadata["compatibility_score"] = compatibility_score.cpu()

        recommended = (
            self._metadata[self._metadata["product_type"] != input_category]
            .sort_values(by="compatibility_score", ascending=False)
            .groupby("product_type")
            .head(1)
            .sort_values(by="compatibility_score", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
        )
        return {
            "input_product": product_metadata,
            "recommended_compatible_products": recommended,
        }


def load_similar_recommender(
    embedding_path: Path | str | None = None,
    task_name: str = "similar_product",
) -> SimilarProductRecommender:
    """Factory: load SimilarProductRecommender.

    Args:
        embedding_path: Override default path.
        task_name: Task name for default path.

    Returns:
        SimilarProductRecommender instance.
    """
    return SimilarProductRecommender(embedding_path=embedding_path, task_name=task_name)


def load_compatible_recommender(
    embedding_path: Path | str | None = None,
    task_name: str = "compatible_product",
) -> CompatibleProductRecommender:
    """Factory: load CompatibleProductRecommender.

    Args:
        embedding_path: Override default path.
        task_name: Task name for default path.

    Returns:
        CompatibleProductRecommender instance.
    """
    return CompatibleProductRecommender(
        embedding_path=embedding_path, task_name=task_name
    )
