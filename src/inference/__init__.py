"""Inference: recommenders for similar and compatible products."""

from src.inference.recommenders import (
    CompatibleProductRecommender,
    SimilarProductRecommender,
    load_compatible_recommender,
    load_similar_recommender,
)

__all__ = [
    "SimilarProductRecommender",
    "CompatibleProductRecommender",
    "load_similar_recommender",
    "load_compatible_recommender",
]
