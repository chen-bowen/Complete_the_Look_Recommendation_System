import os

import pandas as pd
from config import config as cfg
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.Dataset import ProductAndSceneDataset


def build_metadata_csv():
    """Creates the metadata csv file that could be used for image data generator"""

    # if the file exists, skip
    if os.path.exists(f"{cfg.DATASET_DIR}/dataset_metadata.csv"):
        return

    images_df = []

    # navigate within each folder
    for class_folder_name in os.listdir(cfg.RAW_DATA_FOLDER):
        if not class_folder_name.startswith("."):
            class_folder_path = os.path.join(cfg.RAW_DATA_FOLDER, class_folder_name)

            # collect every image path
            for image_name in os.listdir(class_folder_path):
                if not image_name.startswith("."):
                    img_path = os.path.join(class_folder_path, image_name).split("dataset")[1]
                    tmp = pd.DataFrame([f".{img_path}", class_folder_name]).T
                    images_df.append(tmp)

    # concatenate the final df
    images_df = pd.concat(images_df, axis=0, ignore_index=True)
    images_df.columns = ["image_path", "image_type"]

    # save to csv
    images_df.to_csv(f"{cfg.DATASET_DIR}/dataset_metadata.csv")


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
    dataset = ProductAndSceneDataset(
        cfg.RAW_DATA_FOLDER, metadata_csv, transform=transformations, subset="product"
    )
    data_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

    print("Data loader complete. Ready for use.")

    return data_loader


if __name__ == "__main__":

    build_metadata_csv()
    dl = dataloader()
