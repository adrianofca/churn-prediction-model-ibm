"""Configura o logger raiz para emitir logs em formato JSON."""
import logging
import sys

from pythonjsonlogger import jsonlogger


def configure_logging():

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
