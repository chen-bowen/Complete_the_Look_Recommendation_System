import logging

# from src.config import config
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Configure logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
