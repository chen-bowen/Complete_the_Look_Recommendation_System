import concurrent.futures
import json
import urllib.request
from os import makedirs, path

import pandas as pd
import tqdm
from PIL import Image


def convert_to_url(signature):
    """convert image"""
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s.jpg"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


def download_and_save_image_stl(res, image_category, image_type):
    """Get image from signature, image category and image type"""
    img_url = convert_to_url(res[image_type])

    if not path.exists(f"{image_category}/{image_type}"):
        makedirs(f"{image_category}/{image_type}")

    try:
        urllib.request.urlretrieve(
            img_url,
            f"{image_category}/{image_type}/{res[image_type]}.jpg",
        )
    except Exception:
        print(f"Failed to download image {img_url}")
        pass


def get_images_stl(image_category, image_type):
    """Get images from STL dataset given category (fashion/home) and type (scene/product)"""
    img_file_map = {
        "fashion": "./STL-Dataset/fashion.json",
        "home": "./STL-Dataset/home.json",
    }
    img_file = open(img_file_map[image_category])
    image_list = img_file.readlines()
    # use thread pooling to speed up processing downloading images
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(
                download_and_save_image_stl, json.loads(res), image_category, image_type
            )
            for res in image_list
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_url), total=len(future_to_url)
        ):
            try:
                url = future_to_url[future]
            except Exception as e:
                pass


def download_and_save_inages_complete_the_look(res, image_category, image_type):
    """Get image from signature, image category and image type"""
    img_url = convert_to_url(res["image_signature"])

    if not path.exists(f"{image_category}/{image_type}"):
        makedirs(f"{image_category}/{image_type}")

    if not path.exists(f"{image_category}/{image_type}_single"):
        makedirs(f"{image_category}/{image_type}_single")

    try:
        # retrieve the cluster product photo and save to train/test
        urllib.request.urlretrieve(
            img_url,
            f"{image_category}/{image_type}/{res['image_signature']}.jpg",
        )
        # crop out the single product photo and save to train_single/test_single
        img = Image.open(f"{image_category}/{image_type}/{res['image_signature']}.jpg")

        for bounding_box, product_type in zip(res["bounding_boxes"], res["product_type"]):

            # convert bounding box to its coordinates
            img_height, img_width = img.size
            x, y, w, h = bounding_box

            # get bouding box coordinates
            x_min = img_width * x
            y_min = img_height * y

            x_max = x_min + img_width * w
            y_max = y_min + img_height * h

            # crop and save the images
            img_single = img.crop([x_min, y_min, x_max, y_max])
            img_single.save(
                f"{image_category}/{image_type}_single/{res['image_signature']}_{product_type}.jpg"
            )

    except Exception as e:

        print("error:", e)
        breakpoint()
        print(f"Failed to download image {img_url}")
        pass


def get_images_complete_the_look(image_category, image_type):
    img_file_map = {
        "train": "./complete-the-look-dataset/datasets/raw_train.tsv",
        "test": "./complete-the-look-dataset/datasets/raw_test.tsv",
        "triplet_train_1": "./complete-the-look-dataset/datasets/triplet_train_p1.tsv",
        "triplet_train_2": "./complete-the-look-dataset/datasets/triplet_train_p2.tsv",
    }
    image_meta_df = pd.read_csv(img_file_map[image_type], sep="\t", header=None, skiprows=1)
    image_meta_df.columns = "image_signature bounding_x  bounding_y  bounding_width  bounding_height product_type".split()
    image_meta_df["bounding_boxes"] = image_meta_df.apply(
        lambda x: list(
            [x["bounding_x"], x["bounding_y"], x["bounding_width"], x["bounding_height"]]
        ),
        axis=1,
    )
    image_list = (
        image_meta_df[["image_signature", "bounding_boxes", "product_type"]]
        .groupby(["image_signature"])
        .agg({"bounding_boxes": list, "product_type": list})
        .reset_index()
        .to_dict(orient="records")
    )

    # use thread pooling to speed up processing downloading images
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(
                download_and_save_inages_complete_the_look, res, image_category, image_type
            )
            for res in image_list
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_url), total=len(future_to_url)
        ):
            try:
                url = future_to_url[future]
            except Exception as e:
                pass


if __name__ == "__main__":
    # get_images_stl("fashion", "product")
    get_images_complete_the_look("fashion_v2", "train")
