"""Unified data preparation: STL/CTL, Street2Shop, Polyvore.

Options are in configs/data_prep.yaml. Override with --config <path>.

  uv run python -m src.data_pipeline.data_preparation stl_ctl
  uv run python -m src.data_pipeline.data_preparation street2shop
  uv run python -m src.data_pipeline.data_preparation polyvore
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from datasets import disable_progress_bars, enable_progress_bars, load_dataset
from PIL import Image
from tqdm.auto import tqdm

from src.config import get_simple_logger
from src.utils import convert_to_url

logger = get_simple_logger(__name__)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# --- STL/CTL ---
class DataDownloader:
    """Download and prepare STL/CTL (Pinterest), Street2Shop, and Polyvore (Hugging Face)."""

    def __init__(self, data_dir: Path | str | None = None, max_workers: int = 5):
        """Initialize the downloader.

        Args:
            data_dir: Base directory for datasets (STL-Dataset, complete-the-look-dataset).
            max_workers: Number of parallel download threads for STL/CTL.
        """
        self.data_dir = Path(data_dir or ".")
        self.max_workers = max_workers

    def download_stl(self, image_category: str = "fashion", image_type: str = "product") -> None:
        """Download Shop the Look (STL) product images from Pinterest.

        Requires STL-Dataset metadata (fashion.json) cloned into data_dir.
        Saves images to {data_dir}/{category}/{type}/.
        """
        img_file_map = {
            "fashion": self.data_dir / "STL-Dataset" / "fashion.json",
            "home": self.data_dir / "STL-Dataset" / "home.json",
        }
        json_path = img_file_map.get(image_category)
        if not json_path or not json_path.exists():
            raise FileNotFoundError(
                f"STL metadata not found: {json_path}. " "Clone https://github.com/kang205/STL-Dataset into src/dataset/data/"
            )
        with open(json_path) as f:
            image_list = [json.loads(line) for line in f]
        out_dir = self.data_dir / image_category / image_type
        out_dir.mkdir(parents=True, exist_ok=True)
        existed = {p.stem for p in out_dir.glob("*.jpg")}
        to_download = [r for r in image_list if r.get("product") not in existed]
        logger.info(f"STL {image_category}/{image_type}: downloading {len(to_download)} images...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._download_stl_image, r, image_category, image_type): r for r in to_download}
            for f in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="STL",
                unit=" img",
                file=sys.stderr,
                disable=False,
                mininterval=0.1,
            ):
                try:
                    f.result()
                except Exception:
                    pass

    def _download_stl_image(self, res: dict, image_category: str, image_type: str) -> None:
        """Download a single STL image from Pinterest CDN."""
        img_url = convert_to_url(res[image_type])
        out_path = self.data_dir / image_category / image_type / f"{res[image_type]}.jpg"
        try:
            urllib.request.urlretrieve(img_url, str(out_path))
        except Exception as e:
            logger.warning(f"Failed to download {img_url}: {e}")

    def download_ctl(
        self,
        image_category: str = "fashion_v2",
        image_type: str = "train",
    ) -> None:
        """Download Complete the Look (CTL) scenes and crop product images.

        Requires complete-the-look-dataset metadata (raw_train.tsv, raw_test.tsv).
        Downloads full scenes, then crops products by bounding box to {category}/{type}_single/.
        """
        file_map = {
            "train": self.data_dir / "complete-the-look-dataset/datasets/raw_train.tsv",
            "test": self.data_dir / "complete-the-look-dataset/datasets/raw_test.tsv",
        }
        tsv_path = file_map.get(image_type)
        if not tsv_path or not tsv_path.exists():
            raise FileNotFoundError(
                f"CTL metadata not found: {tsv_path}. " "Clone https://github.com/eileenforwhat/complete-the-look-dataset"
            )
        df = pd.read_csv(tsv_path, sep="\t", header=None, skiprows=1)
        df.columns = "image_signature bounding_x bounding_y bounding_width bounding_height product_type".split()
        df["bounding_boxes"] = df.apply(
            lambda r: [r["bounding_x"], r["bounding_y"], r["bounding_width"], r["bounding_height"]],
            axis=1,
        )
        out_dir = self.data_dir / image_category / image_type
        out_single = self.data_dir / image_category / f"{image_type}_single"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_single.mkdir(parents=True, exist_ok=True)
        existed = {p.stem for p in (self.data_dir / image_category / image_type).glob("*.jpg")}
        df = df[~df["image_signature"].isin(existed)]
        records = df.groupby("image_signature").agg({"bounding_boxes": list, "product_type": list}).reset_index().to_dict(orient="records")
        logger.info(f"CTL {image_type}: downloading {len(records)} scenes...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._download_ctl_image, r, image_category, image_type): r for r in records}
            for f in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"CTL {image_type}",
                unit=" img",
                file=sys.stderr,
                disable=False,
                mininterval=0.1,
            ):
                try:
                    f.result()
                except Exception:
                    pass

    def _download_ctl_image(self, res: dict, image_category: str, image_type: str) -> None:
        """Download a single CTL scene, crop products by bbox, save to _single."""
        sig = res["image_signature"]
        img_url = convert_to_url(sig)
        scene_path = self.data_dir / image_category / image_type / f"{sig}.jpg"
        try:
            urllib.request.urlretrieve(img_url, str(scene_path))
            img = Image.open(scene_path)
            for bbox, product_type in zip(res["bounding_boxes"], res["product_type"]):
                x, y, w, h = bbox
                x_min, y_min = img.width * x, img.height * y
                x_max = x_min + img.width * w
                y_max = y_min + img.height * h
                crop = img.crop([x_min, y_min, x_max, y_max])
                out_path = self.data_dir / image_category / f"{image_type}_single" / f"{sig}_{product_type}.jpg"
                crop.save(str(out_path))
        except Exception as e:
            logger.warning(f"Failed to download {img_url}: {e}")

    def _prepare_street2shop(
        self,
        out_dir: Path | str,
        max_pairs: int | None = 5000,
        split: str = "train",
        streaming: bool = True,
    ) -> None:
        """Prepare Street2Shop: stream from Hugging Face, save street/shop image pairs.

        Args:
            out_dir: Output directory (creates street/, shop/, pairs.csv).
            max_pairs: Maximum pairs to save. None = all.
            split: Dataset split (train/test).
            streaming: If True, stream without caching full dataset locally.
        """
        out_dir = Path(out_dir)
        street_dir = out_dir / "street"
        shop_dir = out_dir / "shop"
        street_dir.mkdir(parents=True, exist_ok=True)
        shop_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading Street2Shop from Hugging Face (petr7555/street2shop)...")
        if streaming:
            logger.info("Using streaming mode (no full dataset cached locally)")
        ds = load_dataset("petr7555/street2shop", split=split, streaming=streaming)
        if streaming and max_pairs is not None:
            ds = ds.take(max_pairs)
        elif not streaming and max_pairs is not None:
            ds = ds.select(range(min(max_pairs, len(ds))))
        total = max_pairs if (streaming and max_pairs) else (len(ds) if not streaming else None)
        rows = []

        def save_image(img_or_url, path: Path) -> bool:
            """Save image from PIL, bytes, or URL to path. Returns True on success."""
            if path.exists():
                return True
            try:
                if hasattr(img_or_url, "save"):
                    img_or_url.save(str(path))
                    return True
                if isinstance(img_or_url, bytes):
                    with open(path, "wb") as f:
                        f.write(img_or_url)
                    return True
                if isinstance(img_or_url, str) and img_or_url.startswith("http"):
                    urllib.request.urlretrieve(img_or_url, str(path))
                    return True
            except Exception:
                pass
            return False

        ds_iter = iter(ds)
        pbar = tqdm(total=total, desc="Street2Shop", unit=" pairs", mininterval=0.5, file=sys.stderr)
        i = 0
        while True:
            try:
                row = next(ds_iter)
            except StopIteration:
                break
            except Exception as e:
                logger.warning(f"Skipping row (corrupted image): {e}")
                continue

            pbar.update(1)
            street_id = str(row.get("street_photo_id", f"street_{i}"))
            shop_id = str(row.get("shop_photo_id", f"shop_{i}"))
            street_path = street_dir / f"{street_id}.jpg"
            shop_path = shop_dir / f"{shop_id}.jpg"

            street_ok = save_image(
                row.get("street_photo_image") or row.get("street_photo_url", ""),
                street_path,
            )
            shop_ok = save_image(
                row.get("shop_photo_image") or row.get("shop_photo_url", ""),
                shop_path,
            )
            if street_ok and shop_ok:
                rows.append(
                    {
                        "street_path": f"street/{street_id}.jpg",
                        "shop_path": f"shop/{shop_id}.jpg",
                        "split": split,
                    }
                )
            i += 1
        pbar.close()
        df = pd.DataFrame(rows)
        pairs_path = out_dir / "pairs.csv"
        if pairs_path.exists():
            existing = pd.read_csv(pairs_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(pairs_path, index=False)
        logger.info(f"Saved {len(df)} pairs to {pairs_path}")

    def _prepare_polyvore(
        self,
        out_dir: Path | str,
        max_triplets: int | None = 10000,
        download_images: bool = False,
        streaming: bool = True,
    ) -> None:
        """Prepare Polyvore triplets (anchor, pos, neg) for compatibility training.

        Uses owj0421/polyvore-outfits for metadata. If download_images: True,
        fetches images from Marqo/polyvore. Otherwise writes triplets.csv only.

        Args:
            out_dir: Output directory (creates images/, triplets.csv).
            max_triplets: Maximum triplets to generate. None = all.
            download_images: If True, download images from Marqo/polyvore.
            streaming: If True, stream without caching full dataset locally.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        if download_images:
            self._prepare_polyvore_with_images(out_dir, images_dir, max_triplets, streaming)
            return

        logger.info("Loading Polyvore from Hugging Face (owj0421/polyvore-outfits)...")
        if streaming:
            logger.info("Using streaming mode (no full dataset cached locally)")
        disable_progress_bars()
        try:
            ds = load_dataset("owj0421/polyvore-outfits", "disjoint_default", streaming=False) if not streaming else None
        finally:
            enable_progress_bars()

        def _item_id(it: dict) -> str:
            """Format item dict to string ID (e.g. 132621870_1)."""
            return f"{it['item_id']}_{it['index']}"

        compat_pairs = []
        all_items_by_split = {"train": set(), "validation": set(), "test": set()}
        for split_name in ["train", "validation", "test"]:
            if streaming:
                try:
                    split_ds = load_dataset(
                        "owj0421/polyvore-outfits",
                        "disjoint_default",
                        split=split_name,
                        streaming=True,
                    )
                except Exception:
                    continue
                rows = split_ds
            else:
                if ds is None or split_name not in ds:
                    continue
                rows = ds[split_name]
            for row in rows:
                raw_items = row.get("items", [])
                if len(raw_items) >= 2:
                    items = [_item_id(it) for it in raw_items]
                    all_items_by_split[split_name].update(items)
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            compat_pairs.append((items[i], items[j], split_name))
        all_items = list(all_items_by_split["train"] | all_items_by_split["validation"] | all_items_by_split["test"])
        triplets = []
        for anchor, pos, split in compat_pairs:
            neg = random.choice(all_items)
            while neg == pos or neg == anchor:
                neg = random.choice(all_items)
            triplets.append(
                {
                    "anchor_path": f"images/{anchor}.jpg",
                    "pos_path": f"images/{pos}.jpg",
                    "neg_path": f"images/{neg}.jpg",
                    "split": split,
                }
            )
        if max_triplets is not None:
            random.shuffle(triplets)
            triplets = triplets[:max_triplets]
        df = pd.DataFrame(triplets)
        triplets_path = out_dir / "triplets.csv"
        df.to_csv(triplets_path, index=False)
        logger.info(f"Saved {len(df)} triplets to {triplets_path}")
        logger.info(
            "Note: Add --download-images to fetch images from Marqo/polyvore, or place images manually (e.g. from xthan/polyvore-dataset)"
        )

    def _prepare_polyvore_with_images(
        self,
        out_dir: Path,
        images_dir: Path,
        max_triplets: int | None,
        streaming: bool = True,
    ) -> None:
        """Prepare Polyvore with images from Marqo/polyvore (HF dataset with images).

        Two-pass streaming: first builds outfit structure, second saves only needed images.
        """

        logger.info("Loading Polyvore from Hugging Face (Marqo/polyvore, with images)...")
        if streaming:
            logger.info("Using streaming mode (no full dataset cached locally)")

        ds = load_dataset("Marqo/polyvore", split="data", streaming=streaming)

        # Pass 1: Build outfit structure (item_ID only; discard images to avoid memory)
        outfits: dict[str, list[str]] = {}
        for row in tqdm(ds, desc="Grouping outfits", unit=" items"):
            item_id = str(row.get("item_ID", ""))
            if "_" not in item_id:
                continue
            parts = item_id.rsplit("_", 1)
            set_id = parts[0]
            if set_id not in outfits:
                outfits[set_id] = []
            outfits[set_id].append(item_id)

        compat_pairs = []
        all_items = []
        for items in outfits.values():
            if len(items) < 2:
                continue
            all_items.extend(items)
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    compat_pairs.append((items[i], items[j]))

        all_items = list(set(all_items))
        triplets = []
        for anchor_id, pos_id in compat_pairs:
            neg_id = random.choice(all_items)
            while neg_id in (pos_id, anchor_id):
                neg_id = random.choice(all_items)
            triplets.append((anchor_id, pos_id, neg_id))

        if max_triplets is not None:
            random.shuffle(triplets)
            triplets = triplets[:max_triplets]

        needed = {aid for t in triplets for aid in (t[0], t[1], t[2])}

        # Pass 2: Stream again, save only images we need
        disable_progress_bars()
        try:
            ds2 = load_dataset("Marqo/polyvore", split="data", streaming=streaming)
        finally:
            enable_progress_bars()
        saved_count = 0
        for row in tqdm(ds2, desc="Saving images", unit=" img"):
            item_id = str(row.get("item_ID", ""))
            if item_id not in needed:
                continue
            img = row.get("image")
            if img is None:
                continue
            out_path = images_dir / f"{item_id}.jpg"
            if out_path.exists():
                needed.discard(item_id)
                saved_count += 1
                continue
            if hasattr(img, "save"):
                img.save(str(out_path))
            else:
                Image.open(img).convert("RGB").save(str(out_path))
            saved_count += 1
            needed.discard(item_id)

        n_triplets = len(triplets)
        splits = ["train"] * (n_triplets // 3) + ["validation"] * (n_triplets // 3) + ["test"] * (n_triplets - 2 * (n_triplets // 3))
        random.shuffle(splits)

        df = pd.DataFrame(
            [
                {
                    "anchor_path": f"images/{a}.jpg",
                    "pos_path": f"images/{p}.jpg",
                    "neg_path": f"images/{neg}.jpg",
                    "split": s,
                }
                for (a, p, neg), s in zip(triplets, splits)
            ]
        )
        triplets_path = out_dir / "triplets.csv"
        df.to_csv(triplets_path, index=False)
        logger.info(f"Saved {len(df)} triplets and {saved_count} images to {out_dir}")


if __name__ == "__main__":
    # CLI: load config from YAML, dispatch to DataDownloader by subcommand
    from src.config import load_config

    parser = argparse.ArgumentParser(description="Prepare datasets: stl_ctl, street2shop, polyvore")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_prep.yaml",
        help="Path to data prep config YAML",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("stl_ctl", help="STL and CTL from Pinterest")
    subparsers.add_parser("street2shop", help="Street2Shop from Hugging Face")
    subparsers.add_parser("polyvore", help="Polyvore triplets from Hugging Face")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "stl_ctl":
        c = cfg.get("stl_ctl", {})
        data_dir = c.get("data_dir", ".")
        do_stl = c.get("stl", False)
        do_ctl_train = c.get("ctl_train", False)
        do_ctl_test = c.get("ctl_test", True)
        max_workers = c.get("max_workers", 5)
        if not (do_stl or do_ctl_train or do_ctl_test):
            do_ctl_test = True
        dl = DataDownloader(data_dir=data_dir, max_workers=max_workers)
        if do_stl:
            dl.download_stl("fashion", "product")
        if do_ctl_train:
            dl.download_ctl("fashion_v2", "train")
        if do_ctl_test:
            dl.download_ctl("fashion_v2", "test")
    elif args.cmd == "street2shop":
        c = cfg.get("street2shop", {})
        DataDownloader()._prepare_street2shop(
            out_dir=c.get("out_dir", "src/dataset/data/street2shop"),
            max_pairs=c.get("max_pairs", 5000),
            split=c.get("split", "train"),
            streaming=c.get("streaming", True),
        )
    elif args.cmd == "polyvore":
        c = cfg.get("polyvore", {})
        DataDownloader()._prepare_polyvore(
            out_dir=c.get("out_dir", "src/dataset/data/polyvore"),
            max_triplets=c.get("max_triplets", 10000),
            download_images=c.get("download_images", False),
            streaming=c.get("streaming", True),
        )
