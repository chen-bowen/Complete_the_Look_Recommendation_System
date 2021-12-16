import pathlib

import src
import torch

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent

VALIDATION_PCNT = 0.1

BATCH_SIZE = 64
NUM_EPOCHS = 1
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
DROPOUT = 0.4
LERANING_RATE = 0.0005
MARGIN = 0.2

RAW_DATA_FOLDER = PACKAGE_ROOT / "dataset/data/fashion"
DATASET_DIR = PACKAGE_ROOT / "dataset"
RETURNED_IMAGE_DIR = PACKAGE_ROOT / "images"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models/trained_models"
HEIGHT = 64
WIDTH = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
