import pathlib

import src
import torch

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent

BATCHES = 8

BATCH_SIZE = 32

# RAW_DATA_FOLDER = PACKAGE_ROOT / "dataset/data/fashion"
RAW_DATA_FOLDER = "dataset/data/fashion"
DATASET_DIR = PACKAGE_ROOT / "dataset"
RETURNED_IMAGE_DIR = PACKAGE_ROOT / "images"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models/trained_models"
HEIGHT = 100
WIDTH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
