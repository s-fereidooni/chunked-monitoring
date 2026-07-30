"""Logging setup. Call `setup_logging()` once per entry point."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging. Honours the LOG_LEVEL env var when `level` is None."""
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT)
    # The HF and HTTP stacks are chatty at INFO.
    for noisy in ("httpx", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
