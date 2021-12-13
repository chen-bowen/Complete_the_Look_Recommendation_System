import pickle
import pandas as pd
from utils.similarity import calculate_similarity

def evaluation():
    with open("features/compatible_product_embedding.pickle", "rb") as f:
        features = pickle.load(f)
    metadata = pd.read_csv("dataset/metadata/dataset_metadata_ctl_single.csv")
    sim = calculate_similarity(features[0],features[1], "cosine")
    breakpoint()
    

if __name__ == "__main__":
    evaluation()