"""Recommendation entry points: similar and compatible products.

Backward-compatible functions that delegate to SimilarProductRecommender
and CompatibleProductRecommender.
"""

import random

from src.inference.recommenders import (
    CompatibleProductRecommender,
    SimilarProductRecommender,
)
from src.utils import display_recommended_products


def recommend_similar_products(
    product_id: int,
    task_name: str = "similar_product",
    top_n: int = 5,
) -> dict:
    """Return top-n similar products for the given product ID.

    Args:
        product_id: The product ID to query.
        task_name: Embedding task name (default: similar_product).
        top_n: Number of recommendations.

    Returns:
        Dict with "input_product" and "recommended_products".
    """
    recommender = SimilarProductRecommender(task_name=task_name)
    return recommender.recommend(product_id=product_id, top_n=top_n)


def recommend_complementary_products(
    product_id: int,
    task_name: str = "compatible_product",
    top_n: int = 5,
) -> dict:
    """Return top-n compatible products for the given product ID.

    Args:
        product_id: The product ID to query.
        task_name: Embedding task name (default: compatible_product).
        top_n: Number of recommendations.

    Returns:
        Dict with "input_product" and "recommended_compatible_products".
    """
    recommender = CompatibleProductRecommender(task_name=task_name)
    return recommender.recommend(product_id=product_id, top_n=top_n)


if __name__ == "__main__":
    similar = recommend_similar_products(product_id=random.randint(1, 38000))
    print(similar)
    display_recommended_products(
        similar["input_product"]["image_path"],
        *[rec["image_path"] for rec in similar["recommended_products"]],
        [round(rec["similarity_score"], 3) for rec in similar["recommended_products"]],
    )

    compatible = recommend_complementary_products(product_id=random.randint(1, 454000))
    print(compatible)
    display_recommended_products(
        compatible["input_product"]["image_path"],
        *[rec["image_path"] for rec in compatible["recommended_compatible_products"]],
        [
            round(rec["compatibility_score"], 3)
            for rec in compatible["recommended_compatible_products"]
        ],
        save_image=True,
    )
