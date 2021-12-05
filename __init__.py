import logging

from complete_the_look import dataset, features, utils
from complete_the_look.config import config, logging_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False
