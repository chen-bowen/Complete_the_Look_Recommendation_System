import numpy as np

import torch
import torch.nn as nn

from similarity import calculateSimilarity, plotSimilarityDistribution

#input1 = torch.randn(1, 128)
#input2 = torch.randn(100, 128)
input1 = torch.Tensor([9, 61, 52, 17, 95, 10, 71, 21, 58])
input2 = torch.Tensor([
[33,60,18,61,68,22,69,72,82],
[50,15,28,25,80,84,29,61,41],
[32,38,34,34,65,79,20,94,54],
[57,82,43,94,15,4,26,92,90],
[16,13,6,95,68,64,8,37,2],
[33,60,18,61,68,22,69,72,82],
[50,15,28,25,80,84,29,61,41],
[32,38,34,34,65,79,20,94,54],
[57,82,43,94,15,4,26,92,90],
[16,13,6,95,68,64,8,37,2]])

simlarityVector, topFiveImageSimilarity, topFiveImagesIds = calculateSimilarity(input1, input2,"euclidean")
plotSimilarityDistribution(simlarityVector)

print(input1)
print(input2)
print(simlarityVector)
print(topFiveImageSimilarity)
print(topFiveImagesIds)