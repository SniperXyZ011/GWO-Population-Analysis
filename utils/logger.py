"""
logger.py

Structured logging for the framework.
Provides file + console handlers with per-experiment log entries.
"""

import logging
import sys
from pathlib import Path

from configs.config import LOG_DIR, LOG_LEVEL, LOG_FORMAT


def setup_logger(
    name: str = "GwoPopulationAnalysis",
    log_file: str = "experiment.log",
    level: str = None,
) -> logging.Logger:
    """
    Set up a structured logger with file and console handlers.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str
        Log file name (inside LOG_DIR).
    level : str
        Logging level override.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    log_level = getattr(
        logging, (level or LOG_LEVEL).upper(), logging.INFO
    )

    logger.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_dir / log_file,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
