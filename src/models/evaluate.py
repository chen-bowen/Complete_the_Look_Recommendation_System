import pickle
from utils.similarity import calculate_similarity

def evaluation():
    with open("features/compatible_product_embedding.pickle", "rb") as f:
        features = pickle.load(f)
    breakpoint()
    calculate_similarity()
    pass

if __name__ == "__main__":
    evaluation()