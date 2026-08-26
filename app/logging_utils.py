"""Small, consistent logging setup for the inference runtime."""

from __future__ import annotations

import logging
import os


def _configured_level() -> int:
    if os.getenv("TTS_DEBUG", "0") == "1":
        return logging.DEBUG
    name = os.getenv("TTS_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


_root_logger = logging.getLogger("tontaube")
if not _root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _root_logger.addHandler(handler)
_root_logger.setLevel(_configured_level())
_root_logger.propagate = False


def get_logger(component: str) -> logging.Logger:
    """Return a Tontaube child logger using the configured runtime level."""
    return _root_logger.getChild(component)
