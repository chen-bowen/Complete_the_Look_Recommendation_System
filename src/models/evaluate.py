import pickle
import pandas as pd
import os
import config as cfg
from utils.similarity import calculate_similarity
from src.dataset.Dataloader import FashionCompleteTheLookDataloader
from features.Embedding import StyleEmbedding

def evaluation():
    # with open("features/compatible_product_embedding.pickle", "rb") as f:
    #     features = pickle.load(f)
    if not os.path.exists("dataset/metadata/dataset_metadata_ctl_test_single.csv"):
        metadata = pd.read_csv("dataset/metadata/dataset_metadata_ctl_single.csv")
        metadata_test = metadata[metadata["image_type"]=="test"]
        metadata_test_csv = metadata_test.to_csv("dataset/metadata/dataset_metadata_ctl_test_single.csv")
    metadata = pd.read_csv("dataset/metadata/dataset_metadata_ctl_single.csv")
    metadata_test = metadata[metadata["image_type"]=="test"]
    data_loader=FashionCompleteTheLookDataloader(image_type="test").single_data_loader()
    # breakpoint()
    embedding = StyleEmbedding()
    test_features = embedding.compatible_product_embedding(data_loader=data_loader,task_name="compatible_product_test")
    
    # sim = calculate_similarity(features[0],features[8], "cosine")
    
if __name__ == "__main__":
    evaluation()