"""Configuration and constants for the Complete the Look project.

Re-exports from src.constants for backward compatibility. Use load_config()
to load YAML configs for training, inference, or data prep.
"""

import logging
import pathlib
import sys
from typing import Any

import yaml

from src import constants

# --- Logging ---
FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"
)
# Message only (no level, timestamp, or name)
SIMPLE_FORMATTER = logging.Formatter("%(message)s")


def get_console_handler() -> logging.StreamHandler:
    """Return a console handler for logging with standard format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)
    return handler


def get_simple_logger(name: str) -> logging.Logger:
    """Return a logger that outputs only the message (no INFO/level prefix)."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(SIMPLE_FORMATTER)
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log

# --- Re-export constants (backward compatibility) ---
PACKAGE_ROOT = constants.PACKAGE_ROOT
VALIDATION_PCNT = constants.VALIDATION_PCNT
BATCH_SIZE = constants.BATCH_SIZE
NUM_EPOCHS = constants.NUM_EPOCHS
HIDDEN_DIM = constants.HIDDEN_DIM
EMBEDDING_DIM = constants.EMBEDDING_DIM
DROPOUT = constants.DROPOUT
LEARNING_RATE = constants.LEARNING_RATE
MARGIN = constants.MARGIN
RAW_DATA_FOLDER = constants.RAW_DATA_FOLDER
DATASET_DIR = constants.DATASET_DIR
STREET2SHOP_ROOT = constants.STREET2SHOP_ROOT
POLYVORE_ROOT = constants.POLYVORE_ROOT
RETURNED_IMAGE_DIR = constants.RETURNED_IMAGE_DIR
TRAINED_MODEL_DIR = constants.TRAINED_MODEL_DIR
HEIGHT = constants.HEIGHT
WIDTH = constants.WIDTH
device = constants.DEVICE


def load_config(
    config_path: str | pathlib.Path | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML config and merge with defaults.

    Args:
        config_path: Path to YAML file. If None, returns defaults only.
        defaults: Default values to merge. Config file overrides these.

    Returns:
        Merged config dict.
    """
    result = dict(defaults or {})
    if config_path is None:
        return result
    path = pathlib.Path(config_path)
    if not path.exists():
        return result
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    result.update(loaded)
    return result
