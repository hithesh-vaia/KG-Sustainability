"""Logging setup: writes to stdout (unbuffered) and a timestamped file in output/logs/."""
from __future__ import annotations

import logging
import sys
from datetime import datetime

from .config import CONFIG

_CONFIGURED = False


def setup_logging(run: str = "run", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("graphrag")
    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    log_dir = CONFIG.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"{run}-{stamp}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("logging to %s", log_dir / f"{run}-{stamp}.log")
    _CONFIGURED = True
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("graphrag")
