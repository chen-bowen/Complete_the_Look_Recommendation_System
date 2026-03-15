"""Data loaders for STL, CTL, Street2Shop, and Polyvore.

Builds metadata CSVs, samples triplets, and provides PyTorch DataLoaders
with standard transforms (resize, normalize).
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from src.config import config as cfg
from src.config import get_simple_logger
from src.schemas.dataset_schemas import (FashionProductCTLSingleDataset,
                                         FashionProductCTLTripletDataset,
                                         FashionProductSTLDataset,
                                         PolyvoreTripletDataset,
                                         Street2ShopTripletDataset)
from src.utils import convert_to_url

logger = get_simple_logger(__name__)

class FashionProductSTLDataloader:
    """Dataloader for STL (Shop the Look) single-image dataset."""

    def __init__(self) -> None:
        """Build metadata CSV if needed, then ready for data_loader()."""
        self.build_metadata_csv()

    def build_metadata_csv(self) -> None:
        """Create dataset_metadata_stl.csv with product_id, image_path, product_type, image_url, image_type."""
        # if the file exists, skip
        if os.path.exists(f"{cfg.DATASET_DIR}/metadata/dataset_metadata_stl.csv"):
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
        images_df.to_csv(f"{cfg.DATASET_DIR}/metadata/dataset_metadata_stl.csv", index=False)

    def data_loader(self) -> DataLoader:
        """Return DataLoader for FashionProductSTLDataset (product subset)."""
        # define transformations
        transformations = transforms.Compose(
            [
                transforms.Resize((cfg.HEIGHT, cfg.WIDTH)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # create the dataset and the stl_dataloader
        dataset = FashionProductSTLDataset(
            cfg.RAW_DATA_FOLDER,
            f"{cfg.DATASET_DIR}/metadata/dataset_metadata_stl.csv",
            transform=transformations,
            subset="product",
        )
        return DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=5)


MAX_TRIPLETS_PER_OUTFIT = None  # maximum number of triplets sampled from a single outfit
SKIP_IF_POS_SAME_CATEGORY_AS_ANCHOR = True  # whether or not anchor and pos/neg must be from different categories


class FashionCompleteTheLookDataloader:
    """Dataloader for CTL (Complete the Look) triplets and single images."""

    def __init__(
        self,
        image_type="train",
        batch_size=cfg.BATCH_SIZE,
        num_workers=10,
        use_polyvore_in_compatibility: bool = False,
        polyvore_root: Path | str | None = None,
    ):
        self.image_type = image_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_polyvore_in_compatibility = use_polyvore_in_compatibility
        self.polyvore_root = Path(polyvore_root or str(cfg.DATASET_DIR / "data" / "polyvore"))
        self.build_metadata_csv()

    @property
    def img_file_map(self) -> dict:
        """Map image_type to raw TSV path (train/test)."""
        return {
            "train": f"{cfg.DATASET_DIR}/data/complete-the-look-dataset/datasets/raw_train.tsv",
            "test": f"{cfg.DATASET_DIR}/data/complete-the-look-dataset/datasets/raw_test.tsv",
        }

    @staticmethod
    def sample_triplets(data_by_sig: dict, data_by_cat: dict) -> pd.DataFrame:
        """Sample triplets: anchor/pos same outfit different category; neg same category different outfit."""
        triplets = []
        cnt = 0

        for sig, items in data_by_sig.items():
            pairs_from_outfit = 0

            # shuffle items
            random.shuffle(items)
            for anchor in range(0, len(items) - 1):
                for pos in range(anchor + 1, len(items)):

                    # filter restriction
                    if MAX_TRIPLETS_PER_OUTFIT and pairs_from_outfit >= MAX_TRIPLETS_PER_OUTFIT:
                        continue

                    # skip id positive image the same as anchor
                    if SKIP_IF_POS_SAME_CATEGORY_AS_ANCHOR and items[anchor]["product_type"] == items[pos]["product_type"]:
                        continue

                    # sample negative from the same category as positive (but different outfit)
                    neg_sample = random.choice(data_by_cat[items[pos]["product_type"]])
                    while neg_sample["image_signature"] == sig:
                        neg_sample = random.choice(data_by_cat[items[pos]["product_type"]])

                    # add triplets
                    triplets.append(
                        {
                            "image_signature_anchor": sig,
                            "bounding_box_x_anchor": items[anchor]["x"],
                            "bounding_box_y_anchor": items[anchor]["y"],
                            "bounding_box_w_anchor": items[anchor]["w"],
                            "bounding_box_h_anchor": items[anchor]["h"],
                            "anchor_product_type": items[anchor]["product_type"],
                            "image_signature_pos": sig,
                            "bounding_box_x_pos": items[pos]["x"],
                            "bounding_box_y_pos": items[pos]["y"],
                            "bounding_box_w_pos": items[pos]["w"],
                            "bounding_box_h_pos": items[pos]["h"],
                            "pos_product_type": items[pos]["product_type"],
                            "image_signature_neg": neg_sample["image_signature"],
                            "bounding_box_x_neg": neg_sample["x"],
                            "bounding_box_y_neg": neg_sample["y"],
                            "bounding_box_w_neg": neg_sample["w"],
                            "bounding_box_h_neg": neg_sample["h"],
                            "neg_product_type": neg_sample["product_type"],
                        }
                    )

                    pairs_from_outfit += 1

                    # print info
                    cnt += 1
                    if cnt % 100000 == 0:
                        logger.info("num_triplets={}".format(cnt))
                        logger.info("current_row={}".format(list(triplets)[-1]))

        logger.info("Done! Total number triplets : {}".format(cnt))

        # convert to dataframe
        triplets = pd.DataFrame(triplets).drop_duplicates()

        return triplets

    def build_metadata_csv(self) -> None:
        """Create dataset_metadata_ctl_triplets.csv and dataset_metadata_ctl_single.csv if missing."""
        # if the file exists, skip
        if os.path.exists(f"{cfg.DATASET_DIR}/metadata/dataset_metadata_ctl_triplets.csv") and os.path.exists(
            f"{cfg.DATASET_DIR}/metadata/dataset_metadata_ctl_single.csv"
        ):
            return

        # read csv metadata file
        image_meta_df = pd.read_csv(self.img_file_map["train"], sep="\t", header=None, skiprows=1)
        image_meta_df.columns = "image_signature x  y  w  h product_type".split()

        # filter the image metadata df to contain only images that existed
        existing_images_name = [filename.split(".")[0] for filename in os.listdir(f"{cfg.DATASET_DIR}/data/fashion_v2/train")]
        image_meta_df = image_meta_df[image_meta_df["image_signature"].isin(existing_images_name)]

        if not os.path.exists(f"{cfg.DATASET_DIR}/metadata/dataset_metadata_ctl_triplets.csv"):
            # group by image signature and product type
            image_meta_df["img_info"] = image_meta_df.apply(
                lambda x: {
                    "image_signature": x["image_signature"],
                    "x": x["x"],
                    "y": x["y"],
                    "w": x["w"],
                    "h": x["h"],
                    "product_type": x["product_type"],
                },
                axis=1,
            )
            image_meta_by_signature = (
                image_meta_df[["image_signature", "img_info"]].groupby("image_signature").agg({"img_info": list})
            ).to_dict()["img_info"]

            image_meta_by_product_type = (
                image_meta_df[["product_type", "img_info"]].groupby("product_type").agg({"img_info": list})
            ).to_dict()["img_info"]

            # sample triplets
            triplets = self.sample_triplets(image_meta_by_signature, image_meta_by_product_type)

            # set 90% full dataset to train and 10% to validation
            data_type = np.array(["train"] * len(triplets))
            validation_indices = random.choices(np.arange(len(triplets)), k=int(cfg.VALIDATION_PCNT * len(triplets)))
            data_type[validation_indices] = "validation"
            triplets["image_type"] = data_type

            # save to csv
            triplets.to_csv(f"{cfg.DATASET_DIR}/metadata/dataset_metadata_ctl_triplets.csv", index=False)

        if not os.path.exists(f"{cfg.DATASET_DIR}/metadata/dataset_metadata_ctl_single.csv"):
            # create single image singature
            image_meta_df["image_single_signature"] = image_meta_df[["image_signature", "product_type"]].agg("_".join, axis=1)

            image_meta_df["image_type"] = "train"

            # read in test csv and concat with the image meta df
            image_meta_test_df = pd.read_csv(self.img_file_map["test"], sep="\t", header=None, skiprows=1)
            image_meta_test_df.columns = "image_signature x  y  w  h product_type".split()
            image_meta_test_df["image_type"] = "test"

            # add image single signatureto image meta test df
            image_meta_test_df["image_single_signature"] = image_meta_test_df[["image_signature", "product_type"]].agg("_".join, axis=1)

            # merge image metadata df train and test
            image_meta_df = pd.concat([image_meta_df, image_meta_test_df]).reset_index()
            image_meta_df["product_id"] = image_meta_df.index

            # add image path
            image_meta_df["image_path"] = image_meta_df.apply(
                lambda row: f"./data/fashion_v2/{row['image_type']}_single/{row['image_single_signature']}.jpg",
                axis=1,
            )

            # save to csv
            image_meta_df[
                [
                    "product_id",
                    "image_single_signature",
                    "image_path",
                    "x",
                    "y",
                    "w",
                    "h",
                    "product_type",
                    "image_type",
                ]
            ].drop_duplicates().to_csv(f"{cfg.DATASET_DIR}/metadata/dataset_metadata_ctl_single.csv", index=False)

    def triplet_data_loader(self) -> DataLoader:
        """DataLoader for (anchor, pos, neg) triplets. Optionally merged with Polyvore if enabled."""
        transformations = transforms.Compose(
            [
                transforms.Resize((cfg.HEIGHT, cfg.WIDTH)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        ctl_dataset = FashionProductCTLTripletDataset(
            cfg.RAW_DATA_FOLDER,
            f"{cfg.DATASET_DIR}/metadata/dataset_metadata_ctl_triplets.csv",
            data_type=self.image_type,
            transform=transformations,
        )

        if self.use_polyvore_in_compatibility:
            polyvore_csv = self.polyvore_root / "triplets.csv"
            if polyvore_csv.exists():
                polyvore_dataset = PolyvoreTripletDataset(
                    root=self.polyvore_root,
                    triplets_csv=polyvore_csv,
                    split="train" if self.image_type in ("train", "validation") else "test",
                    transform=transformations,
                )
                dataset = ConcatDataset([ctl_dataset, polyvore_dataset])
            else:
                dataset = ctl_dataset
        else:
            dataset = ctl_dataset

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def single_data_loader(self) -> DataLoader:
        """DataLoader for single CTL images (embedding extraction)."""
        # define transformations
        transformations = transforms.Compose(
            [
                transforms.Resize((cfg.HEIGHT, cfg.WIDTH)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # create the dataset and the ctl single dataloader
        dataset = FashionProductCTLSingleDataset(
            cfg.RAW_DATA_FOLDER,
            f"{cfg.DATASET_DIR}/metadata/dataset_metadata_ctl_single.csv",
            data_type=self.image_type,
            transform=transformations,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class Street2ShopDataloader:
    """Dataloader for Street2Shop triplet training (street -> shop retrieval)."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """Initialize with root (path to street2shop/), split, batch_size, num_workers."""
        self.root = Path(root or str(cfg.DATASET_DIR / "data" / "street2shop"))
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def triplet_data_loader(self) -> DataLoader:
        """DataLoader yielding (anchor_street, pos_shop, neg_shop) triplets."""
        transformations = transforms.Compose(
            [
                transforms.Resize((cfg.HEIGHT, cfg.WIDTH)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = Street2ShopTripletDataset(
            root=self.root,
            pairs_csv=self.root / "pairs.csv",
            split=self.split,
            transform=transformations,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    # dl = FashionProductSTLDataloader().data_loader()
    dl2 = FashionCompleteTheLookDataloader().triplet_data_loader()
    dl3 = FashionCompleteTheLookDataloader().single_data_loader()
