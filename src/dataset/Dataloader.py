import json
import os

import pandas as pd
from src.config import config as cfg
from src.dataset.Dataset import FashionProductSTLDataset
from src.utils.image_utils import convert_to_url
from torch.utils.data import DataLoader
from torchvision import transforms


class FashionProductSTLDataloader:
    def __init__(self):
        self.build_metadata_csv()

    def build_metadata_csv(self):
        """Creates the metadata csv file that could be used for image data generator"""
        # if the file exists, skip

        if os.path.exists(f"{cfg.DATASET_DIR}/dataset_metadata_stl.csv"):
            return

        images_df = []

        # get the fashion categories
        with open(f"{cfg.DATASET_DIR}/data/STL-Dataset/fashion-cat.json") as f:
            product_types = json.load(f)

        # navigate within each folder
        for class_folder_name in os.listdir(cfg.RAW_DATA_FOLDER):
            if not class_folder_name.startswith("."):
                class_folder_path = os.path.join(cfg.RAW_DATA_FOLDER, class_folder_name)

                # collect every image path
                for product_id, image_name in enumerate(os.listdir(class_folder_path)):
                    if not image_name.startswith("."):
                        # get image path
                        img_path = os.path.join(class_folder_path, image_name).split("dataset")[1]
                        # get image product type
                        product_type = product_types[image_name.split(".")[0]]
                        row = pd.DataFrame(
                            [
                                product_id,
                                f".{img_path}",
                                product_type,
                                convert_to_url(image_name.split(".")[0]),
                                class_folder_name,
                            ]
                        ).T
                        images_df.append(row)

        # concatenate the final df
        images_df = pd.concat(images_df, axis=0, ignore_index=True)
        images_df.columns = ["product_id", "image_path", "product_type", "image_url", "image_type"]

        # save to csv
        images_df.to_csv(f"{cfg.DATASET_DIR}/dataset_metadata_stl.csv", index=False)

    def data_loader(self):
        # define transformations
        transformations = transforms.Compose(
            [
                transforms.Resize((cfg.HEIGHT, cfg.WIDTH)),
                # transforms.RandomCrop((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # create the dataset and the stl_dataloader
        dataset = FashionProductSTLDataset(
            cfg.RAW_DATA_FOLDER,
            f"{cfg.DATASET_DIR}/dataset_metadata_stl.csv",
            transform=transformations,
            subset="product",
        )
        return DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)


if __name__ == "__main__":
    dl = FashionProductSTLDataloader().data_loader()