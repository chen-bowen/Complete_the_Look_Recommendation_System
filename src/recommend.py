import pickle

from src.config import config as cfg
from src.dataset.Dataloader import FashionProductSTLDataloader
from src.utils.similarity import calculate_similarity


def recommend_similar_products(product_id, task_name="similar_product", top_n=5):
    """takes in the product id and returns the top"""
    # get extracted features
    with (
        open(f"{cfg.PACKAGE_ROOT}/features/cached_embeddings/{task_name}_embedding.pickle", "rb")
    ) as file:
        all_products_features = pickle.load(file)

    # get dataset metadata dataframe
    data_loader = FashionProductSTLDataloader().data_loader()
    metadata = data_loader.dataset.metadata

    # get query feature from product id
    product_metadata = metadata[metadata["product_id"] == product_id].to_dict(orient="records")[0]
    product_feature_vec = all_products_features[product_id, :]

    # calculate similarities and get all of the 5 products metadata
    simlarity_score = calculate_similarity(product_feature_vec, all_products_features, "cosine")

    # get query product category and filter the prpduct catelog for the same category
    product_category = product_metadata["product_type"]
    metadata["similarity_score"] = simlarity_score.cpu()

    # get top 5 products metadata
    recommended_products_metadata = (
        metadata[
            (metadata["product_type"] == product_category) & (metadata["similarity_score"] != 1)
        ]
        .sort_values(by="similarity_score", ascending=False)
        .head(top_n)
        .to_dict(orient="records")
    )
    return {
        "input_product": product_metadata,
        "recommended_products": recommended_products_metadata,
    }


def recommend_complementary_products(product_id, task_name="compatible_product", top_n=5):
    pass


if __name__ == "__main__":
    import random

    recommendations = recommend_similar_products(product_id=random.randint(1, 38000))

    from utils.image_utils import display_recommended_products

    print(recommendations)
    display_recommended_products(
        recommendations["input_product"]["image_path"],
        *[rec["image_path"] for rec in recommendations["recommended_products"]],
        [round(rec["similarity_score"], 3) for rec in recommendations["recommended_products"]],
    )
