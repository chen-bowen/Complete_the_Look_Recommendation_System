# import os

# import matplotlib.pyplot as plt
# import pandas as pd
import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.io import read_image
# from torchvision.transforms import ToTensor

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class FashionProductDataset(Dataset):
    def __init__(self, X_Train, transform=None):
        self.X_Train = X_Train
        # self.Y_Train = Y_Train
        self.transform = transform

    def __len__(self):
        return len(self.X_Train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X_Train[idx]
        # y = self.Y_Train[idx]

        if self.transform:
            x = self.transform(x)
            # y = self.transform(y)

        # return x, y
        return x