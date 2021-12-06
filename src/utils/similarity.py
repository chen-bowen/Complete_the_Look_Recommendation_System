import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def calculate_similarity(source_vector, destination_vector, sim_function, top_n=5):
    """
    Inputs:
    1. source_vector - the scene image too which we need to map the products - 1D tensor
    2. destination_vector - the destination product images as tensors - 2D tensor
    3. sim_function - similarity function to be used
    ​
    Outputs:
    top_5_image_similarity - col1: image index, col2: cosine similarity
    top_five_images_ids - tensor with top n image ids in the descending order of similarity
    ​"""
    if sim_function == "cosine":
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        simlarity_rank = cos(source_vector, destination_vector)
    elif sim_function == "euclidean":
        pdist = nn.PairwiseDistance(p=2)
        simlarity_rank = pdist(source_vector, destination_vector)
    else:
        return torch.zeros(top_n)

    ## calculate top 5 images
    top_n_image_similarity, top_n_images_ids = torch.topk(simlarity_rank, top_n)

    return simlarity_rank, top_n_image_similarity, top_n_images_ids


def plot_similarity_distribution(simlarity_rank):
    plt.hist(
        simlarity_rank.detach().numpy(), density=False, bins=10
    )  # density=False would make counts
    plt.ylabel("Count")
    plt.xlabel("Similarity")
    plt.show()
    return
