"""
Adapted from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output.
"""

import logging
from typing import Dict


class ColorFormatter(logging.Formatter):
    blue: str = "\x1b[34;20m"
    green: str = "\x1b[32;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    format_str: str = "[%(name)s | %(levelname)s] %(message)s"

    FORMATS: Dict[int, str] = {
        logging.DEBUG: blue + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger() -> logging.Logger:
    logger = logging.getLogger("Agents")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(ColorFormatter())

    logger.addHandler(handler)
    return logger
