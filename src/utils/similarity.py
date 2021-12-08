import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def calculate_similarity(source_vector, destination_vector, sim_function):
    """
    Inputs:
    1. source_vector - the scene image too which we need to map the products - 1D tensor
    2. destination_vector - the destination product images as tensors - 2D tensor
    3. sim_function - similarity function to be used

    Outputs:
    top_5_image_similarity - col1: image index, col2: cosine similarity
    top_five_images_ids - tensor with top n image ids in the descending order of similarity
    """
    if sim_function == "cosine":
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        simlarity_score = cos(source_vector, destination_vector)

    elif sim_function == "euclidean":
        pdist = nn.PairwiseDistance(p=2)
        simlarity_score = -pdist(source_vector, destination_vector)
    else:
        return torch.zeros(destination_vector.size(0))

    return simlarity_score
