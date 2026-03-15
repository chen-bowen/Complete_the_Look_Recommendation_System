"""Utilities: image helpers, similarity, model init."""

import os

import numpy as np
import requests
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image

from src.config import config as cfg


# --- Model utils ---
def init_weights(m: nn.Module) -> None:
    """Apply Xavier initialization on linear layer weights. Skips non-Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


# --- Similarity ---
def calculate_similarity(source_vector, destination_vector, sim_function):
    """Compute similarity between source and destination vectors.

    Args:
        source_vector: 1D tensor (query)
        destination_vector: 2D tensor (candidates)
        sim_function: 'cosine' or 'euclidean'

    Returns:
        Similarity scores (higher = more similar).
    """
    if sim_function == "cosine":
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(source_vector, destination_vector)
    if sim_function == "euclidean":
        pdist = nn.PairwiseDistance(p=2)
        return -pdist(source_vector, destination_vector)
    return torch.zeros(destination_vector.size(0))


# --- Image utils ---
def convert_to_url(signature: str, suffix: str = ".jpg") -> str:
    """Build Pinterest CDN URL from image signature.

    Args:
        signature: Image ID (e.g. from STL/CTL metadata). Strips .jpg if present.
        suffix: URL suffix (default .jpg).

    Returns:
        Full URL for the image.
    """
    if isinstance(signature, str) and signature.endswith(".jpg"):
        signature = signature[:-4]
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature) + suffix


def plot_learning_curves(train_losses: list, validation_losses: list) -> None:
    """Plot train and validation loss curves and save to PNG."""
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses)
    plt.plot(validation_losses)
    plt.legend(["train", "val"], loc="upper left")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Compatible Product Embedding Loss")
    plt.savefig("Compatible Product Embedding Loss.png")


def display_recommended_products(
    im1: str,
    im2: str,
    im3: str,
    im4: str,
    im5: str,
    im6: str,
    simlarity_scores: list,
    save_image: bool = True,
):
    """Display query image (im1) and top 5 recommendations (im2–im6) in a grid.

    Args:
        im1: Path to query/selected product image.
        im2–im6: Paths to recommended product images.
        simlarity_scores: Similarity scores for each recommendation.
        save_image: If True, save to RETURNED_IMAGE_DIR; else return fig.
    """
    fig = plt.figure(figsize=(10, 7))
    rows, columns = 2, 5
    if save_image:
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
            Image.open(requests.get(convert_to_url(im2.split("/")[-1]), stream=True).raw
            ).resize(input_image_size)
        )
        Image3 = np.asarray(
            Image.open(requests.get(convert_to_url(im3.split("/")[-1]), stream=True).raw
            ).resize(input_image_size)
        )
        Image4 = np.asarray(
            Image.open(requests.get(convert_to_url(im4.split("/")[-1]), stream=True).raw
            ).resize(input_image_size)
        )
        Image5 = np.asarray(
            Image.open(requests.get(convert_to_url(im5.split("/")[-1]), stream=True).raw
            ).resize(input_image_size)
        )
        Image6 = np.asarray(
            Image.open(requests.get(convert_to_url(im6.split("/")[-1]), stream=True).raw
            ).resize(input_image_size)
        )
    fig.add_subplot(rows, columns, 2)
    plt.imshow(Image1)
    plt.axis("off")
    plt.title("Selected Product")
    for i, (img, score) in enumerate(
        zip([Image2, Image3, Image4, Image5, Image6], simlarity_scores)
    ):
        fig.add_subplot(rows, columns, 6 + i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Option #{i + 1}\nScore: {round(score, 2)}")
    if save_image:
        if not os.path.exists(cfg.RETURNED_IMAGE_DIR):
            os.makedirs(cfg.RETURNED_IMAGE_DIR)
        plt.savefig(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_all.png")
        input_img = Image.open(f"{cfg.DATASET_DIR}/{im1}")
        input_img.save(f"{cfg.RETURNED_IMAGE_DIR}/input_product.png")
        for i, im_path in enumerate([im2, im3, im4, im5, im6]):
            img = Image.open(f"{cfg.DATASET_DIR}/{im_path}").resize(input_img.size)
            img.save(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_{i + 1}.png")
    else:
        return fig


def display_compatible_images(
    im1: str,
    im2: str,
    im3: str,
    im4: str,
    im5: str,
    im6: str,
    product_id: str,
    save_image: bool = True,
):
    """Display query image and 5 compatible products in a row.

    Args:
        im1–im6: Filenames under fashion_v2/train_single/.
        product_id: Used for output filename.
        save_image: If True, save to RETURNED_IMAGE_DIR; else return fig.
    """
    fig = plt.figure(figsize=(10, 7))
    rows, columns = 1, 7
    base = f"{cfg.DATASET_DIR}/data/fashion_v2/train_single"
    input_image_size = Image.open(f"{base}/{im1}").size
    images = [
        np.asarray(Image.open(f"{base}/{im}").resize(input_image_size))
        for im in [im1, im2, im3, im4, im5, im6]
    ]
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, 1 + i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(" ")
    if save_image:
        if not os.path.exists(cfg.RETURNED_IMAGE_DIR):
            os.makedirs(cfg.RETURNED_IMAGE_DIR)
        plt.savefig(f"{cfg.RETURNED_IMAGE_DIR}/recommendation_{product_id}.png")
    else:
        return fig
