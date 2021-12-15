# import libraries
import os

import numpy as np
import requests
from matplotlib import pyplot as plt
from PIL import Image
from src.config import config as cfg


def bounding_box_process(img, bounding_box):
    """
    takes in the image and bounding box information,
    and returns bounding boxes in x_min, y_min, x_max, y_max
    """
    # get image shape and bounding box information
    img_width, img_height = img.size
    x, y, w, h = bounding_box

    # get bouding box coordinates
    x_min = img_width * x
    y_min = img_height * y

    x_max = x_min + img_width * w
    y_max = y_min + img_height * h

    return [x_min, y_min, x_max, y_max]


def convert_to_url(signature):
    """convert image"""
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


def plot_learning_curves(train_losses, validation_losses):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses)
    plt.plot(validation_losses)
    plt.legend(["train", "val"], loc="upper left")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Compatible Product Embedding Loss")
    plt.savefig(f"Compatible Product Embedding Loss.png")


def display_recommended_products(im1, im2, im3, im4, im5, im6, simlarity_scores, save_image=True):

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 2  # 2
    columns = 5  # 2
    if save_image:
        # reading images
        input_image_size = Image.open(f"{cfg.DATASET_DIR}/{im1}").size
        Image1 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im1}"))
        Image2 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im2}").resize(input_image_size))
        Image3 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im3}").resize(input_image_size))
        Image4 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im4}").resize(input_image_size))
        Image5 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im5}").resize(input_image_size))
        Image6 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im6}").resize(input_image_size))
    else:
        input_image_size = Image.open(
            requests.get(convert_to_url(im1.split("/")[-1]), stream=True).raw
        ).size
        Image1 = np.asarray(
            Image.open(requests.get(convert_to_url(im1.split("/")[-1]), stream=True).raw)
        )
        Image2 = np.asarray(
            Image.open(requests.get(convert_to_url(im2.split("/")[-1]), stream=True).raw).resize(
                input_image_size
            )
        )
        Image3 = np.asarray(
            Image.open(requests.get(convert_to_url(im3.split("/")[-1]), stream=True).raw).resize(
                input_image_size
            )
        )
        Image4 = np.asarray(
            Image.open(requests.get(convert_to_url(im4.split("/")[-1]), stream=True).raw).resize(
                input_image_size
            )
        )
        Image5 = np.asarray(
            Image.open(requests.get(convert_to_url(im5.split("/")[-1]), stream=True).raw).resize(
                input_image_size
            )
        )
        Image6 = np.asarray(
            Image.open(requests.get(convert_to_url(im6.split("/")[-1]), stream=True).raw).resize(
                input_image_size
            )
        )
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
    # return figure if save image is false
    else:
        return fig


def display_recommended_products_one_row(
    im1, im2, im3, im4, im5, im6, simlarity_scores, product_id, save_image=True
):

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1  # 2
    columns = 7  # 2

    # reading images
    input_image_size = Image.open(f"{cfg.DATASET_DIR}/{im1}").size
    Image1 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im1}"))
    Image2 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im2}").resize(input_image_size))
    Image3 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im3}").resize(input_image_size))
    Image4 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im4}").resize(input_image_size))
    Image5 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im5}").resize(input_image_size))
    Image6 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/{im6}").resize(input_image_size))

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(Image1)
    plt.axis("off")
    plt.title(" ")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(Image2)
    plt.axis("off")
    plt.title(f"{round(simlarity_scores[0],2)}")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 4)

    # showing image
    plt.imshow(Image3)
    plt.axis("off")
    plt.title(f"{round(simlarity_scores[1],2)}")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 5)

    # showing image
    plt.imshow(Image4)
    plt.axis("off")
    plt.title(f"{round(simlarity_scores[2],2)}")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 6)

    # showing image
    plt.imshow(Image5)
    plt.axis("off")
    plt.title(f"{round(simlarity_scores[3],2)}")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 7)

    # showing image
    plt.imshow(Image6)
    plt.axis("off")
    plt.title(f"{round(simlarity_scores[4],2)}")

    if save_image:
        # save all recommendation layout
        if not os.path.exists(cfg.RETURNED_IMAGE_DIR):
            os.makedirs(cfg.RETURNED_IMAGE_DIR)
        plt.savefig(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_{product_id}.png")

        """
        # save individual recommendation image
        input_img = Image.open(f"{cfg.DATASET_DIR}/{im1}")
        input_img.save(f"{cfg.RETURNED_IMAGE_DIR}/input_product.png")

        for i, im_path in enumerate([im2, im3, im4, im5, im6]):
            img = Image.open(f"{cfg.DATASET_DIR}/{im_path}").resize(input_img.size)
            img.save(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_{i+1}.png")
        """

    else:
        return fig


def display_compatible_images(im1, im2, im3, im4, im5, im6, product_id, save_image=True):

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1  # 2
    columns = 7  # 2

    # reading images
    input_image_size = Image.open(f"{cfg.DATASET_DIR}/data/fashion_v2/train_single/{im1}").size
    Image1 = np.asarray(Image.open(f"{cfg.DATASET_DIR}/data/fashion_v2/train_single/{im1}"))
    Image2 = np.asarray(
        Image.open(f"{cfg.DATASET_DIR}/data/fashion_v2/train_single/{im2}").resize(input_image_size)
    )
    Image3 = np.asarray(
        Image.open(f"{cfg.DATASET_DIR}/data/fashion_v2/train_single/{im3}").resize(input_image_size)
    )
    Image4 = np.asarray(
        Image.open(f"{cfg.DATASET_DIR}/data/fashion_v2/train_single/{im4}").resize(input_image_size)
    )
    Image5 = np.asarray(
        Image.open(f"{cfg.DATASET_DIR}/data/fashion_v2/train_single/{im5}").resize(input_image_size)
    )
    Image6 = np.asarray(
        Image.open(f"{cfg.DATASET_DIR}/data/fashion_v2/train_single/{im6}").resize(input_image_size)
    )

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(Image1)
    plt.axis("off")
    plt.title(" ")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(Image2)
    plt.axis("off")
    plt.title(" ")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 4)

    # showing image
    plt.imshow(Image3)
    plt.axis("off")
    plt.title(" ")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 5)

    # showing image
    plt.imshow(Image4)
    plt.axis("off")
    plt.title(" ")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 6)

    # showing image
    plt.imshow(Image5)
    plt.axis("off")
    plt.title(" ")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 7)

    # showing image
    plt.imshow(Image6)
    plt.axis("off")
    plt.title(" ")

    if save_image:
        # save all recommendation layout
        if not os.path.exists(cfg.RETURNED_IMAGE_DIR):
            os.makedirs(cfg.RETURNED_IMAGE_DIR)
        plt.savefig(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_{product_id}.png")

        """
        # save individual recommendation image
        input_img = Image.open(f"{cfg.DATASET_DIR}/{im1}")
        input_img.save(f"{cfg.RETURNED_IMAGE_DIR}/input_product.png")

        for i, im_path in enumerate([im2, im3, im4, im5, im6]):
            img = Image.open(f"{cfg.DATASET_DIR}/{im_path}").resize(input_img.size)
            img.save(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_{i+1}.png")
        """

    else:
        return fig


if __name__ == "__main__":

    output = {
        "input_product": {
            "product_id": 32980,
            "image_single_signature": "174d42a13a50ab3fbb61615bd26b399c_Skirts",
            "product_type": "Skirts",
        },
        "recommended_compatible_products": [
            {
                "product_id": 409990,
                "image_single_signature": "e7ffaf2f4ab71f03ec0dc64fd07861ff_Handbags",
                "product_type": "Handbags",
                "compatibility_score": 0.9992280602455139,
            },
            {
                "product_id": 36287,
                "image_single_signature": "192396af62d98cdea94153465b975d2c_Dresses",
                "product_type": "Dresses",
                "compatibility_score": 0.9991121292114258,
            },
            {
                "product_id": 51877,
                "image_single_signature": "21a6d5501760e74d9b69b999d5746ce2_Shorts",
                "product_type": "Shorts",
                "compatibility_score": 0.9990701675415039,
            },
            {
                "product_id": 426039,
                "image_single_signature": "f0de75fa8c7113124ec22bb54ca48704_Coats & Jackets",
                "product_type": "Coats & Jackets",
                "compatibility_score": 0.9990623593330383,
            },
            {
                "product_id": 291004,
                "image_single_signature": "a6a978a3fc6c0593808fac941acd8d94_Hats",
                "product_type": "Hats",
                "compatibility_score": 0.9989411234855652,
            },
        ],
    }
    product_id = output["input_product"]["product_id"]
    im1 = output["input_product"]["image_single_signature"] + ".jpg"
    im2 = output["recommended_compatible_products"][0]["image_single_signature"] + ".jpg"
    im3 = output["recommended_compatible_products"][1]["image_single_signature"] + ".jpg"
    im4 = output["recommended_compatible_products"][2]["image_single_signature"] + ".jpg"
    im5 = output["recommended_compatible_products"][3]["image_single_signature"] + ".jpg"
    im6 = output["recommended_compatible_products"][4]["image_single_signature"] + ".jpg"
    print(product_id)
    print(im1)
    print(im2)
    print(im3)
    print(im4)
    print(im5)
    print(im6)
    display_compatible_images(im1, im2, im3, im4, im5, im6, product_id, save_image=True)
