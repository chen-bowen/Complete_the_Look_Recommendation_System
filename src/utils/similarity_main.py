import numpy as np
import torch
import torch.nn as nn

from similarity import calculate_similarity

# input1 = torch.randn(1, 128)
# input2 = torch.randn(100, 128)
input1 = torch.Tensor([9, 61, 52, 17, 95, 10, 71, 21, 58])
input2 = torch.Tensor(
    [
        [9, 61, 52, 17, 95, 10, 71, 21, 58],
        [33, 60, 18, 61, 68, 22, 69, 72, 82],
        [50, 15, 28, 25, 80, 84, 29, 61, 41],
        [32, 38, 34, 34, 65, 79, 20, 94, 54],
        [57, 82, 43, 94, 15, 4, 26, 92, 90],
        [16, 13, 6, 95, 68, 64, 8, 37, 2],
        [33, 60, 18, 61, 68, 22, 69, 72, 82],
        [50, 15, 28, 25, 80, 84, 29, 61, 41],
        [32, 38, 34, 34, 65, 79, 20, 94, 54],
        [57, 82, 43, 94, 15, 4, 26, 92, 90],
        [16, 13, 6, 95, 68, 64, 8, 37, 2],
    ]
)

simlarity_score = calculate_similarity(
    input1, input2, "cosine"
)
# plot_similarity_distribution(simlarity_score)

print(input1)
print(input2)
print(simlarity_score)
# print(topFiveImageSimilarity)
# print(topFiveImagesIds)
