"""Dataset schemas: PyTorch Dataset classes defining data structure for each source."""

import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import config as cfg


class FashionProductSTLDataset(Dataset):
    """STL (Shop the Look) single-image dataset for similar-product retrieval."""

    def __init__(
        self,
        image_dir: str | Path,
        metadata_file: str | Path,
        transform=None,
        subset: str | None = None,
    ):
        """Load STL metadata and optionally filter by image_type (e.g. 'product')."""
        self.image_dir = image_dir
        self.metadata = (
            pd.read_csv(metadata_file)
            if not subset
            else pd.read_csv(metadata_file)[
                pd.read_csv(metadata_file)["image_type"] == subset
            ]
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        """Return transformed image tensor for the given index."""
        img_row = self.metadata.iloc[index]
        image_path = img_row["image_path"]
        img = Image.open(os.path.join(cfg.PACKAGE_ROOT, "dataset", image_path)).convert(
            "RGB"
        )

        if self.transform is not None:
            img = self.transform(img)

        return img


class FashionProductCTLTripletDataset(Dataset):
    """CTL (Complete the Look) triplet dataset: (anchor, pos, neg) for compatibility training."""

    def __init__(
        self,
        image_dir: str | Path,
        metadata_file: str | Path,
        data_type: str = "train",
        transform=None,
    ):
        """Load CTL triplet metadata. data_type: train, validation, or test."""
        self.image_dir = image_dir
        self.data_type = data_type
        self.transform = transform
        self.metadata = pd.read_csv(metadata_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        triplet_id = self.metadata.reset_index().iloc[index, 0]
        data_src_folder = (
            "train" if self.data_type in ["train", "validation"] else "test"
        )
        img_triplets = []
        for img_type in ["anchor", "pos", "neg"]:
            img = Image.open(
                os.path.join(
                    cfg.PACKAGE_ROOT,
                    "dataset/data/fashion_v2/",
                    f"{data_src_folder}_single",
                    self.metadata.loc[triplet_id, f"image_signature_{img_type}"]
                    + "_"
                    + self.metadata.loc[triplet_id, f"{img_type}_product_type"]
                    + ".jpg",
                )
            ).convert("RGB")

            img_triplets.append(img)

        if self.transform is not None:
            img_triplets = [self.transform(img) for img in img_triplets]

        return tuple(img_triplets)


class FashionProductCTLSingleDataset(Dataset):
    """CTL single-image dataset for embedding extraction (one product per sample)."""

    def __init__(
        self,
        image_dir: str | Path,
        metadata_file: str | Path,
        data_type: str = "train",
        transform=None,
    ):
        """Load CTL single-image metadata. data_type: train, validation, or test."""
        self.image_dir = image_dir
        self.data_type = data_type
        self.transform = transform
        self.create_metadata(metadata_file)

    def create_metadata(self, metadata_file: str | Path) -> None:
        """Load metadata CSV and filter by data_type."""
        metadata = pd.read_csv(metadata_file)
        self.metadata = metadata[metadata["image_type"] == self.data_type]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        self.metadata = self.metadata[self.metadata["image_type"] == self.data_type]
        img_id = self.metadata.reset_index().iloc[index, 0]

        try:
            img = Image.open(
                os.path.join(
                    cfg.PACKAGE_ROOT,
                    "dataset/data/fashion_v2/",
                    f"{self.data_type}_single",
                    self.metadata.loc[img_id, "image_single_signature"] + ".jpg",
                )
            ).convert("RGB")
        except Exception:
            img_src = Image.open(
                os.path.join(
                    cfg.PACKAGE_ROOT,
                    "dataset/data/fashion_v2/",
                    f"{self.data_type}",
                    self.metadata.loc[img_id, "image_single_signature"].split("_")[0]
                    + ".jpg",
                )
            )
            bounding_box = self.metadata.iloc[img_id, 2:6].to_list()
            img = img_src.crop(bounding_box)

        if self.transform is not None:
            img = self.transform(img)

        return img


class PolyvoreTripletDataset(Dataset):
    """Polyvore (anchor, pos, neg) triplets for outfit compatibility."""

    def __init__(
        self,
        root: Path | str,
        triplets_csv: Path | str | None = None,
        split: str = "train",
        transform=None,
    ):
        """Load Polyvore triplets from CSV. Expects anchor_path, pos_path, neg_path, split columns."""
        self.root = Path(root)
        csv_path = Path(triplets_csv or self.root / "triplets.csv")
        self.split = split
        self.transform = transform

        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        anchor_path = self.root / row["anchor_path"]
        pos_path = self.root / row["pos_path"]
        neg_path = self.root / row["neg_path"]

        anchor = Image.open(anchor_path).convert("RGB")
        pos = Image.open(pos_path).convert("RGB")
        neg = Image.open(neg_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anchor, pos, neg


class Street2ShopTripletDataset(Dataset):
    """Street2Shop (anchor_street, pos_shop, neg_shop) triplets for retrieval."""

    def __init__(
        self,
        root: Path | str,
        pairs_csv: Path | str | None = None,
        split: str = "train",
        transform=None,
    ):
        """Load Street2Shop pairs. Builds triplets: (street, pos_shop, random_neg_shop)."""
        self.root = Path(root)
        pairs_path = Path(pairs_csv or self.root / "pairs.csv")
        self.split = split
        self.transform = transform

        df = pd.read_csv(pairs_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.shop_paths = self.df["shop_path"].unique().tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        street_path = self.root / row["street_path"]
        pos_shop_path = self.root / row["shop_path"]

        neg_shop_path = random.choice(self.shop_paths)
        while neg_shop_path == row["shop_path"]:
            neg_shop_path = random.choice(self.shop_paths)
        neg_shop_path = self.root / neg_shop_path

        anchor = Image.open(street_path).convert("RGB")
        pos = Image.open(pos_shop_path).convert("RGB")
        neg = Image.open(neg_shop_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anchor, pos, neg
