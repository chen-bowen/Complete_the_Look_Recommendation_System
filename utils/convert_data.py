from PIL import Image
import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset
import sys
import config as cfg

# from utils.feature_extractor import DATA_LIMIT
BATCHES = cfg.batches
BATCH_SIZE = cfg.batch_size
HEIGHT = 56
WIDTH = 56
FOLDER = "product" # or scene

DATA_LIMIT = BATCHES * BATCH_SIZE
x = 0

def process_data(image_type):
    pickle_list = []
    count = 0
    print("Processing images...")
    for item in os.scandir(f'./data/fashion/{image_type}/'):
        if item.is_file() and not item.name.startswith('.'):
            # with open(f'{image_type}_df.pickle', 'ab') as handle:
            image = Image.open(f"data/fashion/{image_type}/{item.name}")
            image = image.resize(size= (HEIGHT,WIDTH))
            image_to_array = np.asarray(image)/255
            if len(image_to_array.shape)== 3 and image_to_array.shape[2] ==3: #excludes images that do not have 3 channels i.e greyscale
                count+=1  
                pickle_list.append(image_to_array)
                if count >= DATA_LIMIT:
                    break
    print("Processed:", count, "images.")
    return pickle_list


def pickle_processed_data(image_type,processed_data):
    with open(f'{image_type}_df.pickle', 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumped processed data into pickle succesfully.")
    

def unpickle_data(image_type):
    file = open(f"{image_type}_df.pickle","rb")
    unpickled_data = pickle.load(file)
    return unpickled_data


image_data = process_data(FOLDER)
pickle_processed_data(FOLDER,image_data)

unpickled_data = unpickle_data(FOLDER)