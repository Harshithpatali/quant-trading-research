"""
Production logging configuration.

This module centralizes application logging for FastAPI, Streamlit
backend services, data loading, model inference, and strategy logic.

Goals:
- consistent log format
- console logging for local/container environments
- rotating file logging for traditional deployments
- no secrets in log messages
- configurable log level through src.config
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from src.config import (
    LOG_FILE,
    LOG_LEVEL,
)


DEFAULT_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_LOG_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5


def _normalise_level(
    level: str,
) -> int:
    """Convert a textual log level into a logging constant."""
    normalized = level.strip().upper()

    value = getattr(
        logging,
        normalized,
        None,
    )

    if not isinstance(value, int):
        raise ValueError(
            f"Unsupported LOG_LEVEL: {level!r}"
        )

    return value


def configure_logging(
    level: str = LOG_LEVEL,
    log_file: Path = LOG_FILE,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Calling this function repeatedly is safe; existing handlers created
    by this module are replaced rather than duplicated.
    """
    numeric_level = _normalise_level(
        level
    )

    root_logger = logging.getLogger()

    root_logger.setLevel(
        numeric_level
    )

    # Remove handlers previously created by this module.
    for handler in list(
        root_logger.handlers
    ):
        if getattr(
            handler,
            "_quant_trading_handler",
            False,
        ):
            root_logger.removeHandler(
                handler
            )
            handler.close()

    formatter = logging.Formatter(
        fmt=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    # ---------------------------------------------------------------
    # Console handler
    # ---------------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(
        numeric_level
    )

    console_handler.setFormatter(
        formatter
    )

    console_handler._quant_trading_handler = True

    root_logger.addHandler(
        console_handler
    )

    # ---------------------------------------------------------------
    # Rotating file handler
    # ---------------------------------------------------------------

    log_path = Path(log_file)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )

    file_handler.setLevel(
        numeric_level
    )

    file_handler.setFormatter(
        formatter
    )

    file_handler._quant_trading_handler = True

    root_logger.addHandler(
        file_handler
    )

    logger = logging.getLogger(
        "quant_trading"
    )

    logger.info(
        "Logging configured at level %s.",
        logging.getLevelName(
            numeric_level
        ),
    )

    return logger


def get_logger(
    name: str,
) -> logging.Logger:
    """
    Return a named application logger.

    Example:
        logger = get_logger(__name__)
    """
    return logging.getLogger(
        name
    )


def log_prediction(
    logger: logging.Logger,
    *,
    prediction_date: str,
    signal: str,
    probability_up: float,
    threshold: float,
) -> None:
    """
    Log a production prediction without logging credentials or raw
    feature vectors.
    """
    logger.info(
        (
            "Prediction generated | "
            "date=%s | signal=%s | "
            "probability_up=%.6f | threshold=%.6f"
        ),
        prediction_date,
        signal,
        probability_up,
        threshold,
    )


__all__ = [
    "configure_logging",
    "get_logger",
    "log_prediction",
]
