# import libraries
import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from src.config import config as cfg


def bounding_box_process(img, bounding_box):
    """
    takes in the image and bounding box information,
    and returns bounding boxes in x_min, y_min, x_max, y_max
    """
    # get image shape and bounding box information
    img_height, img_width = img.size
    x, y, w, h = bounding_box

    # get bouding box coordinates
    x_min = img_width * x
    y_min = img_height * y

    x_max = x_min + img_width * w
    y_max = y_min + img_height * h

    return [x_min, y_min, x_max, y_max]


def convert_to_url(signature):
    """convert image"""
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s.jpg"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


def display_recommended_products(im1, im2, im3, im4, im5, im6, simlarity_scores, save_image=True):

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 2  # 2
    columns = 5  # 2

    # reading images
    input_image_size = Image.open(f"{cfg.DATASET_DIR}/{im1}").size
    Image1 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im1}"))
    Image2 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im2}").resize(input_image_size))
    Image3 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im3}").resize(input_image_size))
    Image4 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im4}").resize(input_image_size))
    Image5 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im5}").resize(input_image_size))
    Image6 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im6}").resize(input_image_size))

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(Image1)
    plt.axis("off")
    plt.title("Selected Product")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 6)

    # showing image
    plt.imshow(Image2)
    plt.axis("off")
    plt.title(f"Option #1 \n Score: {round(simlarity_scores[0],2)}")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 7)

    # showing image
    plt.imshow(Image3)
    plt.axis("off")
    plt.title(f"Option #2 \n Score: {round(simlarity_scores[1],2)}")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 8)

    # showing image
    plt.imshow(Image4)
    plt.axis("off")
    plt.title(f"Option #3 \n Score: {round(simlarity_scores[2],2)}")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 9)

    # showing image
    plt.imshow(Image5)
    plt.axis("off")
    plt.title(f"Option #4 \n Score: {round(simlarity_scores[3],2)}")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 10)

    # showing image
    plt.imshow(Image6)
    plt.axis("off")
    plt.title(f"Option #5 \n Score: {round(simlarity_scores[4],2)}")

    if save_image:
        # save all recommendation layout
        if not os.path.exists(cfg.RETURNED_IMAGE_DIR):
            os.makedirs(cfg.RETURNED_IMAGE_DIR)
        plt.savefig(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_all.png")

        # save individual recommendation image
        input_img = Image.open(f"{cfg.DATASET_DIR}/{im1}")
        input_img.save(f"{cfg.RETURNED_IMAGE_DIR}/input_product.png")

        for i, im_path in enumerate([im2, im3, im4, im5, im6]):
            img = Image.open(f"{cfg.DATASET_DIR}/{im_path}").resize(input_img.size)
            img.save(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_{i+1}.png")

    else:
        return fig
