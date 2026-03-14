"""Unified data preparation: STL/CTL, Street2Shop, Polyvore.

Run from project root:
  uv run python -m src.data_pipeline.data_preparation stl_ctl --ctl-test --data-dir src/dataset/data
  uv run python -m src.data_pipeline.data_preparation street2shop --out-dir src/dataset/data/street2shop
  uv run python -m src.data_pipeline.data_preparation polyvore --out-dir src/dataset/data/polyvore
  uv run python -m src.data_pipeline.data_preparation polyvore --download-images  # includes images (~2.5GB)
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
from tqdm.auto import tqdm
from PIL import Image

from src.utils import convert_to_url

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# --- STL/CTL ---
class DataDownloader:
    """Download and prepare STL and CTL image data from Pinterest."""

    def __init__(self, data_dir: Path | str | None = None, max_workers: int = 5):
        self.data_dir = Path(data_dir or ".")
        self.max_workers = max_workers

    def download_stl(self, image_category: str = "fashion", image_type: str = "product") -> None:
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
        print(f"STL {image_category}/{image_type}: downloading {len(to_download)} images...", flush=True)
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
        img_url = convert_to_url(res[image_type])
        out_path = self.data_dir / image_category / image_type / f"{res[image_type]}.jpg"
        try:
            urllib.request.urlretrieve(img_url, str(out_path))
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

    def download_ctl(
        self,
        image_category: str = "fashion_v2",
        image_type: str = "train",
    ) -> None:
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
        print(f"CTL {image_type}: downloading {len(records)} scenes...", flush=True)
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
            print(f"Failed to download {img_url}: {e}")


# --- Street2Shop ---
def _prepare_street2shop(
    out_dir: Path | str,
    max_pairs: int | None = 5000,
    split: str = "train",
) -> None:
    from datasets import disable_progress_bars, enable_progress_bars, load_dataset

    out_dir = Path(out_dir)
    street_dir = out_dir / "street"
    shop_dir = out_dir / "shop"
    street_dir.mkdir(parents=True, exist_ok=True)
    shop_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Street2Shop from Hugging Face (petr7555/street2shop)...")
    disable_progress_bars()
    try:
        ds = load_dataset("petr7555/street2shop", split=split)
    finally:
        enable_progress_bars()
    os.environ.pop("HF_DATASETS_DISABLE_PROGRESS_BARS", None)
    if max_pairs is not None:
        ds = ds.select(range(min(max_pairs, len(ds))))
    total = len(ds)
    rows = []
    it = tqdm(
        enumerate(ds),
        total=total,
        desc="Street2Shop",
        unit=" pairs",
        mininterval=0.5,
        file=sys.stderr,
    )
    for i, row in it:
        street_id = str(row.get("street_photo_id", f"street_{i}"))
        shop_id = str(row.get("shop_photo_id", f"shop_{i}"))
        street_path = street_dir / f"{street_id}.jpg"
        shop_path = shop_dir / f"{shop_id}.jpg"

        def save_image(img_or_url, path: Path) -> bool:
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
    df = pd.DataFrame(rows)
    pairs_path = out_dir / "pairs.csv"
    if pairs_path.exists():
        existing = pd.read_csv(pairs_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(pairs_path, index=False)
    print(f"Saved {len(df)} pairs to {pairs_path}")


# --- Polyvore ---
def _prepare_polyvore(
    out_dir: Path | str,
    max_triplets: int | None = 10000,
    download_images: bool = False,
) -> None:
    from datasets import disable_progress_bars, enable_progress_bars, load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if download_images:
        _prepare_polyvore_with_images(out_dir, images_dir, max_triplets)
        return

    print("Loading Polyvore from Hugging Face (owj0421/polyvore-outfits)...")
    disable_progress_bars()
    try:
        ds = load_dataset("owj0421/polyvore-outfits", "disjoint_default")
    finally:
        enable_progress_bars()

    def _item_id(it: dict) -> str:
        return f"{it['item_id']}_{it['index']}"

    compat_pairs = []
    all_items_by_split = {"train": set(), "validation": set(), "test": set()}
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            raw_items = row.get("items", [])
            if len(raw_items) >= 2:
                items = [_item_id(it) for it in raw_items]
                all_items_by_split[split_name].update(items)
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        compat_pairs.append((items[i], items[j], split_name))
    all_items = list(
        all_items_by_split["train"] | all_items_by_split["validation"] | all_items_by_split["test"]
    )
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
    print(f"Saved {len(df)} triplets to {triplets_path}")
    print("Note: Add --download-images to fetch images from Marqo/polyvore, or place images manually (e.g. from xthan/polyvore-dataset)")


def _prepare_polyvore_with_images(
    out_dir: Path,
    images_dir: Path,
    max_triplets: int | None,
) -> None:
    """Prepare Polyvore with images from Marqo/polyvore (HF dataset with images)."""
    from datasets import disable_progress_bars, enable_progress_bars, load_dataset

    print("Loading Polyvore from Hugging Face (Marqo/polyvore, with images)...")
    disable_progress_bars()
    try:
        ds = load_dataset("Marqo/polyvore", split="data")
    finally:
        enable_progress_bars()

    # Group by outfit (item_ID prefix before last underscore, e.g. 100002074_1 -> 100002074)
    outfits: dict[str, list[tuple[str, object]]] = {}
    for row in tqdm(ds, desc="Grouping outfits", unit=" items"):
        item_id = str(row.get("item_ID", ""))
        if "_" not in item_id:
            continue
        img = row.get("image")
        if img is None:
            continue
        parts = item_id.rsplit("_", 1)
        set_id = parts[0]
        if set_id not in outfits:
            outfits[set_id] = []
        outfits[set_id].append((item_id, img))

    compat_pairs = []
    all_items = []
    for items in outfits.values():
        if len(items) < 2:
            continue
        ids = [x[0] for x in items]
        all_items.extend(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                compat_pairs.append((ids[i], ids[j]))

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

    # Save images and build triplet rows
    needed = {aid for t in triplets for aid in (t[0], t[1], t[2])}
    id_to_img = {}
    for set_id, items in outfits.items():
        for item_id, img in items:
            if item_id in needed:
                id_to_img[item_id] = img

    for item_id, img in tqdm(id_to_img.items(), desc="Saving images", unit=" img"):
        out_path = images_dir / f"{item_id}.jpg"
        if not out_path.exists():
            if hasattr(img, "save"):
                img.save(str(out_path))
            else:
                Image.open(img).convert("RGB").save(str(out_path))

    n_triplets = len(triplets)
    splits = (
        ["train"] * (n_triplets // 3)
        + ["validation"] * (n_triplets // 3)
        + ["test"] * (n_triplets - 2 * (n_triplets // 3))
    )
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
    print(f"Saved {len(df)} triplets and {len(id_to_img)} images to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets: stl_ctl, street2shop, polyvore")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # stl_ctl
    p_stl = subparsers.add_parser("stl_ctl", help="STL and CTL from Pinterest")
    p_stl.add_argument("--data-dir", type=str, default=".")
    p_stl.add_argument("--stl", action="store_true")
    p_stl.add_argument("--ctl-train", action="store_true")
    p_stl.add_argument("--ctl-test", action="store_true")
    p_stl.add_argument("--max-workers", type=int, default=5)

    # street2shop
    p_s2s = subparsers.add_parser("street2shop", help="Street2Shop from Hugging Face")
    p_s2s.add_argument("--out-dir", type=str, default="src/dataset/data/street2shop")
    p_s2s.add_argument("--max-pairs", type=int, default=5000)
    p_s2s.add_argument("--split", type=str, default="train")

    # polyvore
    p_pv = subparsers.add_parser("polyvore", help="Polyvore triplets from Hugging Face")
    p_pv.add_argument("--out-dir", type=str, default="src/dataset/data/polyvore")
    p_pv.add_argument("--max-triplets", type=int, default=10000)
    p_pv.add_argument("--download-images", action="store_true", help="Download images from Marqo/polyvore (~2.5GB)")

    args = parser.parse_args()

    if args.cmd == "stl_ctl":
        dl = DataDownloader(data_dir=args.data_dir, max_workers=args.max_workers)
        if args.stl:
            dl.download_stl("fashion", "product")
        if args.ctl_train:
            dl.download_ctl("fashion_v2", "train")
        if args.ctl_test:
            dl.download_ctl("fashion_v2", "test")
        if not (args.stl or args.ctl_train or args.ctl_test):
            dl.download_ctl("fashion_v2", "test")
    elif args.cmd == "street2shop":
        _prepare_street2shop(
            out_dir=args.out_dir,
            max_pairs=args.max_pairs,
            split=args.split,
        )
    elif args.cmd == "polyvore":
        _prepare_polyvore(
            out_dir=args.out_dir,
            max_triplets=args.max_triplets,
            download_images=args.download_images,
        )
