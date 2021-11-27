import concurrent.futures
import json
import urllib.request
from os import mkdir, path

import tqdm


def convert_to_url(signature):
    """ convert image """
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s.jpg"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


def download_and_save_image(res, image_category, image_type):
    """ Get image from signature, image category and image type """
    img_url = convert_to_url(res[image_type])

    if not path.exists(f"data/{image_category}/{image_type}"):
        mkdir(f"data/{image_category}/{image_type}")

    try:
        urllib.request.urlretrieve(
            img_url, f"data/{image_category}/{image_type}/{res[image_type]}.jpg",
        )
    except Exception:
        print(f"Failed to download image {img_url}")
        pass


def get_images(image_category, image_type):
    """ Get images given category (fashion/home) and type (scene/product)"""
    img_file_map = {"fashion": "./STL-Dataset/fashion.json", "home": "./STL-Dataset/home.json"}
    img_file = open(img_file_map[image_category])
    image_list = img_file.readlines()
    # use thread pooling to speed up processing downloading images
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(download_and_save_image, json.loads(res), image_category, image_type)
            for res in image_list
        }
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_url)):
            try:
                url = future_to_url[future]
            except Exception as e:
                pass


if __name__ == "__main__":
    get_images("fashion", "product")
