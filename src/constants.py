"""Project-wide constants: paths, model defaults, and image sizes.

Used by config, data loaders, models, and inference. Paths are resolved relative
to the package root.
"""

import pathlib

import src
import torch

# --- Paths ---
PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / "dataset"
RAW_DATA_FOLDER = PACKAGE_ROOT / "dataset/data/fashion"
STREET2SHOP_ROOT = DATASET_DIR / "data" / "street2shop"
POLYVORE_ROOT = DATASET_DIR / "data" / "polyvore"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models/trained_models"
CACHED_EMBEDDINGS_DIR = PACKAGE_ROOT / "features/cached_embeddings"
RETURNED_IMAGE_DIR = PACKAGE_ROOT / "images"

# --- Training hyperparameters ---
VALIDATION_PCNT = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 1
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
DROPOUT = 0.4
LEARNING_RATE = 0.0001
MARGIN = 1.0

# --- Image dimensions ---
HEIGHT = 224
WIDTH = 224

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
