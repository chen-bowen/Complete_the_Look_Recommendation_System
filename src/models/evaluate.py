import os
import pickle

import config as cfg
import pandas as pd
import torch
from features.Embedding import StyleEmbedding
from src.dataset.Dataloader import FashionCompleteTheLookDataloader
from utils.similarity import calculate_similarity


def evaluation():
    """
    Method: Randomly pick a product (anchor), find a product within the same image of a different category (positive),
    find a different product of the same category but from a different image (negative), if the score for anchor-positive
    is higher than score for anchor-negative, then it is deemed to be correct.
    """

    # read in the metadata file
    metadata = pd.read_csv("dataset/metadata/dataset_metadata_ctl_single.csv")
    metadata_test = metadata[metadata["image_type"] == "test"]
    metadata_test.loc[:, "product_id"] = metadata_test.reset_index().index.to_list()
    data_loader = FashionCompleteTheLookDataloader(image_type="test").single_data_loader()

    # load in the embedding
    embedding = StyleEmbedding()
    if not os.path.exists("features/cached_embeddings/compatible_product_test_embedding.pickle"):
        test_features = embedding.compatible_product_embedding(
            data_loader=data_loader, task_name="compatible_product_test"
        )
    else:
        with (
            open("features/cached_embeddings/compatible_product_test_embedding.pickle", "rb")
        ) as file:
            test_features = pickle.load(file)

    # get original image signature
    metadata_test["original_image_signature"] = metadata_test["image_single_signature"].apply(
        lambda row: row.split("_")[0]
    )

    # generate a set of triplets that could be used for evaluation
    triplets = []
    for signature in metadata_test["original_image_signature"].unique():
        # sample anchor image
        image_src = metadata_test[metadata_test["original_image_signature"] == signature]

        anchor = image_src.sample(1).to_dict(orient="records")[0]

        # sample positive image
        positive = (
            image_src[image_src["product_type"] != anchor["product_type"]]
            .sample(1)
            .to_dict(orient="records")[0]
        )

        # sample negative image
        negative_image_src = metadata_test[
            (metadata_test["product_type"] == positive["product_type"])
            & (metadata_test["original_image_signature"] != positive["original_image_signature"])
        ]
        negative = negative_image_src.sample(1).to_dict(orient="records")[0]
        triplets.append((anchor, positive, negative))

    # if the embedding for postive is closer to anchor than negative, add 1 to correct
    total_score = 0
    for anchor, positive, negative in triplets:
        anchor_feat = test_features[anchor["product_id"], :].unsqueeze(0)
        positive_feat = test_features[positive["product_id"], :].unsqueeze(0)
        negative_feat = test_features[negative["product_id"], :].unsqueeze(0)

        sim_ap = calculate_similarity(anchor_feat, positive_feat, sim_function="cosine")
        sim_an = calculate_similarity(anchor_feat, negative_feat, sim_function="cosine")
        if sim_ap > sim_an:
            total_score += 1

    return total_score / len(triplets)


if __name__ == "__main__":
    correct_pcnt = evaluation()
    print(f"The correct percentage of the compatibility test is {correct_pcnt}")
