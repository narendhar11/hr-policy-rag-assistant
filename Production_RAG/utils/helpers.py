"""Shared utilities: logging setup and common helper functions."""

import logging
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_PATH = _LOG_DIR / "app.log"


def get_logger(name: str) -> logging.Logger:
    """Get a named logger that writes to Production_RAG/logs/app.log."""
    logging.basicConfig(
        filename=str(_LOG_PATH),
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(name)
