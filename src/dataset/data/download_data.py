"""Data download for STL and Complete the Look datasets.

DataDownloader: Download images from Pinterest, crop product regions for CTL.
Run from project root or src/dataset/data/; paths are relative to cwd.
"""

import concurrent.futures
import json
from pathlib import Path

import pandas as pd
import tqdm
import urllib.request
from PIL import Image

from src.utils.image_utils import convert_to_url


class DataDownloader:
    """Download and prepare STL and CTL image data from Pinterest."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        max_workers: int = 5,
    ):
        """Initialize downloader.

        Args:
            data_dir: Base directory (contains STL-Dataset/, complete-the-look-dataset/).
                      Default: current working directory.
            max_workers: Thread pool size for downloads.
        """
        self.data_dir = Path(data_dir or ".")
        self.max_workers = max_workers

    def download_stl(
        self, image_category: str = "fashion", image_type: str = "product"
    ) -> None:
        """Download Shop the Look (STL) images.

        Args:
            image_category: 'fashion' or 'home'.
            image_type: 'scene' or 'product'.
        """
        img_file_map = {
            "fashion": self.data_dir / "STL-Dataset" / "fashion.json",
            "home": self.data_dir / "STL-Dataset" / "home.json",
        }
        json_path = img_file_map.get(image_category)
        if not json_path or not json_path.exists():
            raise FileNotFoundError(
                f"STL metadata not found: {json_path}. "
                "Clone https://github.com/kang205/STL-Dataset into src/dataset/data/"
            )
        with open(json_path) as f:
            image_list = [json.loads(line) for line in f]

        out_dir = self.data_dir / image_category / image_type
        out_dir.mkdir(parents=True, exist_ok=True)
        existed = {p.stem for p in out_dir.glob("*.jpg")}
        to_download = [r for r in image_list if r.get("product") not in existed]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._download_stl_image, r, image_category, image_type
                ): r
                for r in to_download
            }
            for f in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                try:
                    f.result()
                except Exception:
                    pass

    def _download_stl_image(
        self, res: dict, image_category: str, image_type: str
    ) -> None:
        """Download a single STL image."""
        img_url = convert_to_url(res[image_type])
        out_path = (
            self.data_dir / image_category / image_type / f"{res[image_type]}.jpg"
        )
        try:
            urllib.request.urlretrieve(img_url, str(out_path))
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

    def download_ctl(
        self,
        image_category: str = "fashion_v2",
        image_type: str = "train",
    ) -> None:
        """Download Complete the Look (CTL) images and crop product regions.

        Args:
            image_category: Output folder name (e.g. fashion_v2).
            image_type: 'train' or 'test'.
        """
        file_map = {
            "train": self.data_dir / "complete-the-look-dataset/datasets/raw_train.tsv",
            "test": self.data_dir / "complete-the-look-dataset/datasets/raw_test.tsv",
        }
        tsv_path = file_map.get(image_type)
        if not tsv_path or not tsv_path.exists():
            raise FileNotFoundError(
                f"CTL metadata not found: {tsv_path}. "
                "Clone https://github.com/eileenforwhat/complete-the-look-dataset"
            )
        df = pd.read_csv(tsv_path, sep="\t", header=None, skiprows=1)
        df.columns = "image_signature bounding_x bounding_y bounding_width bounding_height product_type".split()
        df["bounding_boxes"] = df.apply(
            lambda r: [
                r["bounding_x"],
                r["bounding_y"],
                r["bounding_width"],
                r["bounding_height"],
            ],
            axis=1,
        )

        out_dir = self.data_dir / image_category / image_type
        out_single = self.data_dir / image_category / f"{image_type}_single"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_single.mkdir(parents=True, exist_ok=True)

        existed = {
            p.stem for p in (self.data_dir / image_category / image_type).glob("*.jpg")
        }
        df = df[~df["image_signature"].isin(existed)]
        records = (
            df.groupby("image_signature")
            .agg({"bounding_boxes": list, "product_type": list})
            .reset_index()
            .to_dict(orient="records")
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._download_ctl_image, r, image_category, image_type
                ): r
                for r in records
            }
            for f in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                try:
                    f.result()
                except Exception:
                    pass

    def _download_ctl_image(
        self,
        res: dict,
        image_category: str,
        image_type: str,
    ) -> None:
        """Download CTL scene image and crop product regions."""
        sig = res["image_signature"]
        img_url = convert_to_url(sig)
        scene_path = self.data_dir / image_category / image_type / f"{sig}.jpg"
        try:
            urllib.request.urlretrieve(img_url, str(scene_path))
            img = Image.open(scene_path)
            for bbox, product_type in zip(res["bounding_boxes"], res["product_type"]):
                x, y, w, h = bbox
                x_min = img.width * x
                y_min = img.height * y
                x_max = x_min + img.width * w
                y_max = y_min + img.height * h
                crop = img.crop([x_min, y_min, x_max, y_max])
                out_path = (
                    self.data_dir
                    / image_category
                    / f"{image_type}_single"
                    / f"{sig}_{product_type}.jpg"
                )
                crop.save(str(out_path))
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument(
        "--stl", action="store_true", help="Download STL fashion/product"
    )
    parser.add_argument("--ctl-train", action="store_true", help="Download CTL train")
    parser.add_argument("--ctl-test", action="store_true", help="Download CTL test")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    downloader = DataDownloader(data_dir=args.data_dir, max_workers=args.max_workers)
    if args.stl:
        downloader.download_stl("fashion", "product")
    if args.ctl_train:
        downloader.download_ctl("fashion_v2", "train")
    if args.ctl_test:
        downloader.download_ctl("fashion_v2", "test")
    if not (args.stl or args.ctl_train or args.ctl_test):
        # Default: CTL test (matches original __main__)
        downloader.download_ctl("fashion_v2", "test")
