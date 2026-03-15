"""Complete the Look: fashion product recommendation (similar + compatible)."""

import logging

from src import features, utils
from src.config import config, get_console_handler

__all__ = ["config", "features", "utils"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_handler())
logger.propagate = False
