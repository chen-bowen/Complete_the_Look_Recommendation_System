import json
import os

import pandas as pd
from config import config as cfg
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.Dataset import FashionProductDataset


def build_metadata_csv():
    """Creates the metadata csv file that could be used for image data generator"""

    # if the file exists, skip
    if os.path.exists(f"{cfg.DATASET_DIR}/dataset_metadata.csv"):
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
                        [product_id, f".{img_path}", product_type, class_folder_name]
                    ).T
                    images_df.append(row)

    # concatenate the final df
    images_df = pd.concat(images_df, axis=0, ignore_index=True)
    images_df.columns = ["product_id", "image_path", "product_type", "image_type"]

    # save to csv
    images_df.to_csv(f"{cfg.DATASET_DIR}/dataset_metadata.csv", index=False)


def dataloader(metadata_csv=f"{cfg.DATASET_DIR}/dataset_metadata.csv"):

    print("Creating data loader...")

    # define transformations
    transformations = transforms.Compose(
        [
            transforms.Resize((cfg.HEIGHT, cfg.WIDTH)),
            # transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # create the dataset and the dataloader
    dataset = FashionProductDataset(
        cfg.RAW_DATA_FOLDER, metadata_csv, transform=transformations, subset="product"
    )
    data_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

    print("Data loader complete. Ready for use.")

    return data_loader


if __name__ == "__main__":

    build_metadata_csv()
    dl = dataloader()
    
