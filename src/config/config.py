import pathlib

import src

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent

BATCHES = 8
BATCH_SIZE = 64
RAW_DATA_FOLDER = PACKAGE_ROOT / "dataset/data/fashion"
DATASET_DIR = PACKAGE_ROOT / "dataset"
RETURNED_IMAGE_DIR = PACKAGE_ROOT / "images"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models"
HEIGHT = 56
WIDTH = 56
