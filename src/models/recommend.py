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
        "input_product": product_metadata,
        "recommended_products": recommended_products_metadata,
    }

if __name__ == "__main__":
    recommendations = recommend_similar_products(product_id=7)
    print(recommendations)
# input1 = torch.randn(1, 128)
# input2 = torch.randn(100, 128)
# input1 = torch.Tensor([9, 61, 52, 17, 95, 10, 71, 21, 58])
# input2 = torch.Tensor([
# [33,60,18,61,68,22,69,72,82],
# [50,15,28,25,80,84,29,61,41],
# [32,38,34,34,65,79,20,94,54],
# [57,82,43,94,15,4,26,92,90],
# [16,13,6,95,68,64,8,37,2],
# [33,60,18,61,68,22,69,72,82],
# [50,15,28,25,80,84,29,61,41],
# [32,38,34,34,65,79,20,94,54],
# [57,82,43,94,15,4,26,92,90],
# [16,13,6,95,68,64,8,37,2]])

# plot_similarity_distribution(simlarity_score)

# print(input1)
# print(input2)
# print(simlarity_score)
# print(topFiveImageSimilarity)
# print(topFiveImagesIds)
