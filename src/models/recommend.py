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
    product_metadata = metadata[metadata["product_id"] == product_id].to_dict(orient="records")
    product_feature_vec = all_products_features[product_id, :]

    # calculate similarities and get all of the 5 products metadata
    simlarity_score = calculate_similarity(product_feature_vec, all_products_features, "cosine")

    top_n_image_similarity, top_n_images_ids = torch.topk(simlarity_score, top_n + 1)

    # get top 5 products metadata
    recommended_products_metadata = metadata[
        metadata["product_id"].isin(top_n_images_ids[1:].numpy().tolist())
    ].to_dict(orient="records")

    return {
        "input_product": product_metadata[0],
        "recommended_products": recommended_products_metadata,
    }

if __name__ == "__main__":
    recommendations = recommend_similar_products(product_id=200)

    from utils.show_images import print_image

    print(recommendations)
    print_image(
        recommendations["input_product"]["image_path"],
        *[rec["image_path"] for rec in recommendations["recommended_products"]],
    )
