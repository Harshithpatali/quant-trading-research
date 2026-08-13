"""
FastAPI API security helpers.

P16 adds optional API-key protection for production prediction endpoints.

The secret is read from the environment through src.config.
It is never hard-coded and is never returned in API responses.

Behavior:
- If API_SECRET_KEY is empty, authentication is disabled. This keeps
  local development simple.
- If API_SECRET_KEY is configured, protected endpoints require the
  X-API-Key header.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status

from src.config import API_SECRET_KEY


API_KEY_HEADER = "X-API-Key"


def require_api_key(
    api_key: Annotated[
        str | None,
        Header(
            alias=API_KEY_HEADER,
            description="Production API key.",
        ),
    ] = None,
) -> None:
    """
    Validate the configured API key.

    Constant-time comparison is used to avoid leaking information
    through string-comparison timing differences.
    """

    # Local-development mode: authentication is intentionally optional.
    if not API_SECRET_KEY:
        return

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "API_KEY_REQUIRED",
                "message": "API authentication is required.",
            },
            headers={
                "WWW-Authenticate": "ApiKey",
            },
        )

    if not secrets.compare_digest(
        api_key,
        API_SECRET_KEY,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": "INVALID_API_KEY",
                "message": "The supplied API key is invalid.",
            },
        )


def authentication_enabled() -> bool:
    """Return whether API-key authentication is currently enabled."""
    return bool(API_SECRET_KEY)


__all__ = [
    "API_KEY_HEADER",
    "require_api_key",
    "authentication_enabled",
]
