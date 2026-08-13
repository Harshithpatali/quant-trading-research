"""
Application configuration for the S&P 500 Quant Trading system.

All environment-specific values and secrets are loaded from .env.
No credentials should be hard-coded in this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name!r} must be an integer."
        ) from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name!r} must be a number."
        ) from exc


def _get_required(name: str) -> str:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        raise ValueError(
            f"Required environment variable {name!r} is not configured."
        )

    return value.strip()


# -------------------------------------------------------------------
# Application
# -------------------------------------------------------------------

APP_NAME = os.getenv(
    "APP_NAME",
    "S&P 500 Quant Trading API",
)

APP_ENV = os.getenv(
    "APP_ENV",
    "development",
)

DEBUG = _get_bool(
    "DEBUG",
    default=False,
)

API_HOST = os.getenv(
    "API_HOST",
    "0.0.0.0",
)

API_PORT = _get_int(
    "API_PORT",
    default=8000,
)

STREAMLIT_PORT = _get_int(
    "STREAMLIT_PORT",
    default=8501,
)


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------

_cors_origins_raw = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
)

CORS_ORIGINS = [
    origin.strip()
    for origin in _cors_origins_raw.split(",")
    if origin.strip()
]


# -------------------------------------------------------------------
# Production model
# -------------------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / os.getenv(
    "MODEL_PATH",
    "models/production/sp500_direction_model.joblib",
)

MODEL_METADATA_PATH = PROJECT_ROOT / os.getenv(
    "MODEL_METADATA_PATH",
    "models/production/sp500_direction_model_metadata.json",
)

MODEL_CONFIG_PATH = PROJECT_ROOT / os.getenv(
    "MODEL_CONFIG_PATH",
    "models/production/sp500_frozen_research_config.json",
)

MODEL_THRESHOLD = _get_float(
    "MODEL_THRESHOLD",
    default=0.50,
)


# -------------------------------------------------------------------
# Market data
# -------------------------------------------------------------------

YAHOO_TICKER = os.getenv(
    "YAHOO_TICKER",
    "^GSPC",
)

MIN_HISTORY_DAYS = _get_int(
    "MIN_HISTORY_DAYS",
    default=253,
)


# -------------------------------------------------------------------
# Supabase
# -------------------------------------------------------------------
#
# Supabase is the production data source for this application.
# The secret key is loaded only from the environment.
#
# Do NOT place a real SUPABASE_KEY in source code.
#

SUPABASE_URL = _get_required(
    "SUPABASE_URL",
)

SUPABASE_KEY = _get_required(
    "SUPABASE_KEY",
)

SUPABASE_TABLE = os.getenv(
    "SUPABASE_TABLE",
    "sp500_daily",
)


# -------------------------------------------------------------------
# Local data paths
# -------------------------------------------------------------------

RAW_DATA_PATH = PROJECT_ROOT / os.getenv(
    "RAW_DATA_PATH",
    "data/raw/sp500_1950_present.csv",
)

INTERIM_DATA_PATH = PROJECT_ROOT / os.getenv(
    "INTERIM_DATA_PATH",
    "data/interim",
)

PROCESSED_DATA_PATH = PROJECT_ROOT / os.getenv(
    "PROCESSED_DATA_PATH",
    "data/processed",
)


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

LOG_FILE = PROJECT_ROOT / os.getenv(
    "LOG_FILE",
    "logs/application.log",
)


# -------------------------------------------------------------------
# Security
# -------------------------------------------------------------------

APP_SECRET_KEY = os.getenv(
    "APP_SECRET_KEY",
    "",
)

API_SECRET_KEY = os.getenv(
    "API_SECRET_KEY",
    "",
)


# -------------------------------------------------------------------
# Monitoring
# -------------------------------------------------------------------

ENABLE_HEALTH_CHECK = _get_bool(
    "ENABLE_HEALTH_CHECK",
    default=True,
)

ENABLE_REQUEST_LOGGING = _get_bool(
    "ENABLE_REQUEST_LOGGING",
    default=True,
)


# -------------------------------------------------------------------
# Startup validation
# -------------------------------------------------------------------

def validate_configuration() -> None:
    """
    Validate configuration required by the running application.

    This deliberately does not validate the existence of the model
    artifact because different application stages may call this
    function before the model-loading service is initialized.
    """

    if not SUPABASE_URL.startswith(
        "https://"
    ):
        raise ValueError(
            "SUPABASE_URL must use HTTPS."
        )

    if not SUPABASE_TABLE:
        raise ValueError(
            "SUPABASE_TABLE cannot be empty."
        )

    if MIN_HISTORY_DAYS < 253:
        raise ValueError(
            "MIN_HISTORY_DAYS must be at least 253 "
            "for the current feature set."
        )

    if not 0.0 <= MODEL_THRESHOLD <= 1.0:
        raise ValueError(
            "MODEL_THRESHOLD must be between 0 and 1."
        )

    if API_PORT < 1 or API_PORT > 65535:
        raise ValueError(
            "API_PORT must be between 1 and 65535."
        )

    if STREAMLIT_PORT < 1 or STREAMLIT_PORT > 65535:
        raise ValueError(
            "STREAMLIT_PORT must be between 1 and 65535."
        )


__all__ = [
    "PROJECT_ROOT",
    "APP_NAME",
    "APP_ENV",
    "DEBUG",
    "API_HOST",
    "API_PORT",
    "STREAMLIT_PORT",
    "CORS_ORIGINS",
    "MODEL_PATH",
    "MODEL_METADATA_PATH",
    "MODEL_CONFIG_PATH",
    "MODEL_THRESHOLD",
    "YAHOO_TICKER",
    "MIN_HISTORY_DAYS",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "SUPABASE_TABLE",
    "RAW_DATA_PATH",
    "INTERIM_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "LOG_LEVEL",
    "LOG_FILE",
    "APP_SECRET_KEY",
    "API_SECRET_KEY",
    "ENABLE_HEALTH_CHECK",
    "ENABLE_REQUEST_LOGGING",
    "validate_configuration",
]
