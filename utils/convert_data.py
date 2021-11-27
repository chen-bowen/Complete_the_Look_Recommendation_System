from PIL import Image
import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset

def image_to_array(image_type):
    # pickle_list = []
    count = 0
    for item in os.scandir(f'./data/fashion/{image_type}/'):
        count+=1
        if item.is_file() and not item.name.startswith('.'):
            with open(f'{image_type}_df.pickle', 'wb') as handle:
                image = Image.open(f"data/fashion/{image_type}/{item.name}")
                image_to_array = np.asarray(image)/255 
                # pickle_list.append(image_to_array)
                pickle.dump(image_to_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("dumped:", count)
    # breakpoint()
    # with open(f'{image_type}_df.pickle', 'wb') as handle:
    #     pickle.dump(pickle_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadData(image_type):
    
    # db = pickle.load(dbfile)
    i = 1
    with open(f'{image_type}_df.pickle', 'rb') as db:
        for image in range(len(pickle.load(db))):
            i += 1
            # print(image)
    print(i)
    

# image_to_array("product")
loadData("product")