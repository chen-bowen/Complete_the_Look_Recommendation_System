import json
import os
import random

import pandas as pd
from src.config import config as cfg
from src.dataset.Dataset import FashionProductCTLTripletDataset, FashionProductSTLDataset
from src.utils.image_utils import convert_to_url
from torch.utils.data import DataLoader
from torchvision import transforms


class FashionProductSTLDataloader:
    def __init__(self):
        self.build_metadata_csv()

    def build_metadata_csv(self):
        """
        Creates the metadata csv file that could be used for image data generator,
        metadata includes the following fields
        "product_id", "image_path", "product_type", "image_url", "image_type"
        """
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
        """Dataloader for FashionProductSTLDataset"""
        # define transformations
        transformations = transforms.Compose(
            [
                transforms.Resize((cfg.HEIGHT, cfg.WIDTH)),
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
        return DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=5)


MAX_TRIPLETS_PER_OUTFIT = None  # maximum number of triplets sampled from a single outfit
SKIP_IF_POS_SAME_CATEGORY_AS_ANCHOR = (
    True  # whether or not anchor and pos/neg must be from different categories
)


class FashionCompleteTheLookDataloader:
    def __init__(self, image_type="train", batch_size=cfg.BATCH_SIZE, num_workers=8):
        self.image_type = image_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.build_metadata_csv()

    @staticmethod
    def sample_triplets(data_by_sig, data_by_cat):
        """
        Sample triplets where <anchor, pos> are from the same outfit but different categories
        and <pos, neg> are from the same category but different outfits.
        """
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
                    if (
                        SKIP_IF_POS_SAME_CATEGORY_AS_ANCHOR
                        and items[anchor]["product_type"] == items[pos]["product_type"]
                    ):
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
                        print("num_triplets={}".format(cnt))
                        print("current_row={}".format(list(triplets)[-1]))

        print("Done! Total number triplets : {}".format(cnt))

        # convert to dataframe
        triplets = pd.DataFrame(triplets).drop_duplicates()

        return triplets

    def build_metadata_csv(self):
        """
        Creates the metadata csv file that could be used for image data generator，
        metadata includes a triplet of anchor, postive and negative image
        """
        # if the file exists, skip
        if os.path.exists(f"{cfg.DATASET_DIR}/dataset_metadata_ctl_triplets.csv"):
            return

        img_file_map = {
            "train": f"{cfg.DATASET_DIR}/data/complete-the-look-dataset/datasets/raw_train.tsv",
            "test": f"{cfg.DATASET_DIR}/data/complete-the-look-dataset/datasets/raw_test.tsv",
        }

        # read csv metadata file
        image_meta_df = pd.read_csv(
            img_file_map[self.image_type], sep="\t", header=None, skiprows=1
        )
        image_meta_df.columns = "image_signature x  y  w  h product_type".split()

        # filter the image metadata df to contain only images that existed
        existing_images_name = [
            filename.split(".")[0]
            for filename in os.listdir(f"{cfg.DATASET_DIR}/data/fashion_v2/train")
        ]
        image_meta_df = image_meta_df[image_meta_df["image_signature"].isin(existing_images_name)]

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
            image_meta_df[["image_signature", "img_info"]]
            .groupby("image_signature")
            .agg({"img_info": list})
        ).to_dict()["img_info"]

        image_meta_by_product_type = (
            image_meta_df[["product_type", "img_info"]]
            .groupby("product_type")
            .agg({"img_info": list})
        ).to_dict()["img_info"]

        # sample triplets
        triplets = self.sample_triplets(image_meta_by_signature, image_meta_by_product_type)

        # save to csv
        triplets.to_csv(f"{cfg.DATASET_DIR}/dataset_metadata_ctl_triplets.csv", index=False)

    def data_loader(self):
        """Dataloader for FashionProductCTLTripletDataset"""
        # define transformations
        transformations = transforms.Compose(
            [
                transforms.Resize((cfg.HEIGHT, cfg.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # create the dataset and the stl_dataloader
        dataset = FashionProductCTLTripletDataset(
            cfg.RAW_DATA_FOLDER,
            f"{cfg.DATASET_DIR}/dataset_metadata_ctl_triplets.csv",
            transform=transformations,
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )


if __name__ == "__main__":
    # dl = FashionProductSTLDataloader().data_loader()
    dl2 = FashionCompleteTheLookDataloader().data_loader()
