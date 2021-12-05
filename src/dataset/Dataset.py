import os

import pandas as pd
from config import config as cfg
from PIL import Image
from torch.utils.data import Dataset


class ProductAndSceneDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None, subset=None):
        self.image_dir = image_dir
        self.metadata = (
            pd.read_csv(metadata_file)
            if not subset
            else pd.read_csv(metadata_file)[pd.read_csv(metadata_file)["image_type"] == subset]
        )
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        img_id = self.metadata.iloc[index, 0]
        img = Image.open(
            os.path.join(cfg.PACKAGE_ROOT, "dataset/", self.metadata.loc[img_id, "image_path"])
        ).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img
