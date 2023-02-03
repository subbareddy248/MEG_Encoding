import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def set_logging_level(level):
    level = logging.basicConfig(level=logging._nameToLevel[level])
