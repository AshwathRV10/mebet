"""Structured logging for data retrieval, predictions and model performance."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from .config import get_settings

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    settings.ensure_dirs()
    lvl = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    root = logging.getLogger("mebet")
    root.setLevel(lvl)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    fileh = logging.handlers.RotatingFileHandler(
        Path(settings.log_dir) / "mebet.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fileh.setFormatter(fmt)
    root.addHandler(fileh)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"mebet.{name}")
