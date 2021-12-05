import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
'''
Inputs: 
1. sourceVector - the scene image too which we need to map the products - 1D tensor
2. destinationVector - the destination product images as tensors - 2D tensor
3. function - similarity function to be used
​
Outputs:
simlarityVector - col1: image index, col2: cosine similarity
topFiveImagesIds - tensor with top 5 image ids in the descending order of similarity
​
'''
def calculateSimilarity(sourceVector, destinationVector, function):
    if function == "cosine":
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        simlarityVector = cos(sourceVector, destinationVector)
    elif function == "euclidean":
        pdist = nn.PairwiseDistance(p=2)
        simlarityVector = pdist(sourceVector, destinationVector)
    else:
        return torch.zeros(5)
    ## calculate top 5 images
    topFiveImageSimilarity, topFiveImagesIds = torch.topk(simlarityVector, 5)
    
    return simlarityVector, topFiveImageSimilarity, topFiveImagesIds

def plotSimilarityDistribution(simlarityVector):
    plt.hist(simlarityVector.detach().numpy(), density=False, bins=10)  # density=False would make counts
    plt.ylabel('Count')
    plt.xlabel('Similarity')
    plt.show()
    return