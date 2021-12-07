import pickle

import torch
from config import config as cfg
from src.dataset.create_dataloader import dataloader
from src.utils.similarity import calculate_similarity


def recommend_similar_products(product_id, top_n=5):
    """takes in the product id and returns the top"""
    # get extracted features
    with (open(f"{cfg.PACKAGE_ROOT}/features/product_features.pickle", "rb")) as file:
        all_products_features = pickle.load(file)

    # get dataset metadata dataframe
    data_loader = dataloader()
    metadata = data_loader.dataset.metadata

    # get query feature from product id
    product_metadata = metadata[metadata["product_id"] == product_id].to_dict(orient="records")[0]
    product_feature_vec = all_products_features[product_id, :]

    # calculate similarities and get all of the 5 products metadata
    simlarity_score = calculate_similarity(product_feature_vec, all_products_features, "cosine")

    # get query product category and filter the prpduct catelog for the same category
    product_category = product_metadata["product_type"]
    metadata["similarity_score"] = simlarity_score

    # get top 5 products metadata
    recommended_products_metadata = (
        metadata[(metadata["similarity_score"] != 1)]
        .sort_values(by="similarity_score", ascending=False)
        .head(top_n)
        .to_dict(orient="records")
    )
    return {
        "input_product": product_metadata,
        "recommended_products": recommended_products_metadata,
    }


if __name__ == "__main__":
    recommendations = recommend_similar_products(product_id=140)

    from utils.show_images import print_image

    print(recommendations)
    print_image(
        recommendations["input_product"]["image_path"],
        *[rec["image_path"] for rec in recommendations["recommended_products"]],
    )
