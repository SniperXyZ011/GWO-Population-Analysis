"""
logger.py

Structured logging for the framework.

Provides:
    - File + console handlers with per-experiment log entries
    - Optional JSON-lines structured log for machine-parseable analysis
    - Per-campaign log files with timestamps
    - Worker-aware log formatting
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from configs.config import LOG_DIR, LOG_LEVEL, LOG_FORMAT


def setup_logger(
    name: str = "GwoPopulationAnalysis",
    log_file: str = None,
    level: str = None,
    enable_jsonl: bool = True,
) -> logging.Logger:
    """
    Set up a structured logger with file and console handlers.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str, optional
        Log file name. If None, uses timestamped campaign file.
    level : str
        Logging level override.
    enable_jsonl : bool
        If True, also write machine-parseable JSON-lines log.

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

    # File handler — timestamped campaign log
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"campaign_{timestamp}.log"

    file_handler = logging.FileHandler(
        log_dir / log_file,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # JSON-lines handler for machine-parseable logs
    if enable_jsonl:
        jsonl_file = log_dir / log_file.replace(".log", ".jsonl")
        jsonl_handler = _JsonLinesHandler(jsonl_file)
        jsonl_handler.setLevel(log_level)
        logger.addHandler(jsonl_handler)

    return logger


class _JsonLinesHandler(logging.Handler):
    """
    Logging handler that writes structured JSON-lines.

    Each log entry is a single JSON object on one line,
    enabling easy parsing by monitoring tools.
    """

    def __init__(self, filepath: Path):
        super().__init__()
        self._filepath = Path(filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "pid": os.getpid(),
            }

            # Include extra fields if present
            for key in ("experiment", "metric", "value"):
                if hasattr(record, key):
                    entry[key] = getattr(record, key)

            with open(self._filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception:
            self.handleError(record)


def log_experiment_metric(
    logger: logging.Logger,
    experiment_name: str,
    metric: str,
    value: float,
    message: str = "",
):
    """
    Log a structured metric for an experiment.

    Parameters
    ----------
    logger : logging.Logger
    experiment_name : str
    metric : str
        Metric name (e.g., 'fe_per_second', 'best_score').
    value : float
    message : str
    """

    logger.info(
        message or f"{experiment_name} | {metric}={value}",
        extra={
            "experiment": experiment_name,
            "metric": metric,
            "value": value,
        },
    )
