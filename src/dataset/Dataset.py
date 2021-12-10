import ast
import os

import pandas as pd
from PIL import Image, ImageFile
from src.config import config as cfg
from src.utils.image_utils import bounding_box_process
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FashionProductSTLDataset(Dataset):
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


class FashionProductCTLTripletDataset(Dataset):
    def __init__(self, image_dir, metadata_file, data_type="train", transform=None):
        self.image_dir = image_dir
        self.data_type = data_type
        self.transform = transform
        self.metadata = pd.read_csv(metadata_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        triplet_id = self.metadata.reset_index().iloc[index, 0]
        # get the anchor, postive and negative image and save to img triplets
        img_triplets = []
        for img_type in ["anchor", "pos", "neg"]:
            img_src = Image.open(
                os.path.join(
                    cfg.PACKAGE_ROOT,
                    "dataset/data/fashion_v2/",
                    self.data_type,
                    self.metadata.loc[triplet_id, f"image_signature_{img_type}"] + ".jpg",
                )
            ).convert("RGB")
            img_boundingbox = bounding_box_process(
                img_src,
                [
                    self.metadata.loc[triplet_id, f"bounding_box_{cord}_{img_type}"]
                    for cord in ["x", "y", "w", "h"]
                ],
            )
            img = img_src.crop(img_boundingbox)
            img_triplets.append(img)

        if self.transform is not None:
            img_triplets = [self.transform(img) for img in img_triplets]

        return tuple(img_triplets)
