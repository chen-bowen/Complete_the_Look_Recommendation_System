import logging

from src import dataset, features, utils
from src.config import config, logging_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False
