"""Logging setup.

Console output is intentionally terse and readable::

    [INFO] Downloading video
    [INFO] Video duration: 01:42:18

A full, timestamped copy is written to the job workspace when one exists.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"

LOGGER_NAME = "clipper"


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("CLIPPER_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


class ConsoleFormatter(logging.Formatter):
    """``[LEVEL] message`` with optional ANSI colour."""

    def __init__(self, color: bool) -> None:
        super().__init__("%(message)s")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        level = record.levelname
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if self.color:
            tint = _LEVEL_COLORS.get(level, "")
            return f"{tint}[{level}]{_RESET} {message}"
        return f"[{level}] {message}"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure and return the application logger (idempotent)."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    numeric = getattr(logging, str(level).upper(), logging.INFO)

    for handler in list(logger.handlers):
        if getattr(handler, "_clipper_console", False):
            logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(numeric)
    console.setFormatter(ConsoleFormatter(_supports_color(sys.stderr)))
    console._clipper_console = True  # type: ignore[attr-defined]
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing = {
            getattr(h, "baseFilename", None)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        if str(log_file.resolve()) not in existing:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
            )
            logger.addHandler(file_handler)

    return logger


def get_logger(suffix: str = "") -> logging.Logger:
    """Return the shared application logger (optionally a child of it)."""

    if suffix:
        return logging.getLogger(f"{LOGGER_NAME}.{suffix}")
    return logging.getLogger(LOGGER_NAME)
