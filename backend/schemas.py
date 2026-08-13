"""
Pydantic schemas for the S&P 500 Quant Trading API.

These models define the public API contract between the FastAPI backend
and clients such as the Streamlit frontend.

The schemas contain no model-training logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------

SignalType = Literal["LONG", "CASH"]


class APIStatus(BaseModel):
    """Generic API status response."""

    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "success",
        "error",
        "healthy",
        "unhealthy",
    ]


# ---------------------------------------------------------------------
# Prediction response
# ---------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Safe model metadata exposed to API consumers."""

    model_config = ConfigDict(
        extra="forbid"
    )

    type: str | None = None
    feature_count: int = Field(
        ge=1
    )


class DataInfo(BaseModel):
    """Information about the market data used for prediction."""

    model_config = ConfigDict(
        extra="forbid"
    )

    source: str
    rows_used: int = Field(
        ge=1
    )
    date_start: str | None = None
    date_end: str | None = None
    validation_passed: bool


class PredictionResponse(BaseModel):
    """Complete production prediction response."""

    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal["success"] = "success"

    requested_at: datetime

    prediction_date: str

    signal: SignalType

    prediction: Literal[0, 1]

    position: float = Field(
        ge=0.0,
        le=1.0,
    )

    probability_up: float = Field(
        ge=0.0,
        le=1.0,
    )

    probability_down: float = Field(
        ge=0.0,
        le=1.0,
    )

    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    model: ModelInfo

    data: DataInfo


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

class SupabaseHealth(BaseModel):
    """Supabase health information."""

    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "connected",
        "error",
        "unknown",
    ]

    table: str | None = None

    rows: int | None = Field(
        default=None,
        ge=0,
    )

    error: str | None = None


class ModelHealth(BaseModel):
    """Production model health information."""

    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "loaded",
        "error",
        "unknown",
    ]

    type: str | None = None

    feature_count: int | None = Field(
        default=None,
        ge=1,
    )

    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    error: str | None = None


class HealthResponse(BaseModel):
    """Complete application health response."""

    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "healthy",
        "unhealthy",
    ]

    supabase: SupabaseHealth | dict[str, Any]

    model: ModelHealth | dict[str, Any]


# ---------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard API error payload."""

    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal["error"] = "error"

    error_code: str

    message: str

    details: dict[str, Any] | None = None


# ---------------------------------------------------------------------
# API metadata
# ---------------------------------------------------------------------

class RootResponse(BaseModel):
    """Root API information."""

    model_config = ConfigDict(
        extra="forbid"
    )

    name: str

    version: str

    status: Literal["online"]

    documentation: str


class VersionResponse(BaseModel):
    """Application version response."""

    model_config = ConfigDict(
        extra="forbid"
    )

    version: str

    environment: str


__all__ = [
    "SignalType",
    "APIStatus",
    "ModelInfo",
    "DataInfo",
    "PredictionResponse",
    "SupabaseHealth",
    "ModelHealth",
    "HealthResponse",
    "ErrorResponse",
    "RootResponse",
    "VersionResponse",
]
