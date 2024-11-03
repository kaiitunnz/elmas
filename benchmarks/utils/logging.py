import logging

from agents.utils.logging import ColorFormatter


def init_logger() -> logging.Logger:
    logger = logging.getLogger("BenchmarkRunner")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(ColorFormatter())

    logger.addHandler(handler)
    return logger
