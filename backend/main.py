"""
FastAPI application for the S&P 500 Quant Trading production system.

API responsibilities:
    - expose the current production prediction
    - expose application/model/data health
    - expose basic application metadata

Business logic remains in src/services/prediction_service.py.
The API layer does not train or modify the model.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    RootResponse,
    VersionResponse,
)
from src.config import (
    APP_ENV,
    APP_NAME,
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
)
from src.services.prediction_service import (
    prediction_health_check,
    predict_current_market,
)


# ---------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------

APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    description=(
        "Production API for S&P 500 market-direction prediction "
        "using a frozen research model."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Global exception handling
# ---------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Return a consistent API error without exposing secrets.

    Detailed exception text is intentionally not returned to the
    external client because it can contain infrastructure information.
    """
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------

@app.get(
    "/",
    response_model=RootResponse,
    tags=["System"],
)
async def root() -> RootResponse:
    """Return basic API information."""
    return RootResponse(
        name=APP_NAME,
        version=APP_VERSION,
        status="online",
        documentation="/docs",
    )


# ---------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------

@app.get(
    "/version",
    response_model=VersionResponse,
    tags=["System"],
)
async def version() -> VersionResponse:
    """Return application version and environment."""
    return VersionResponse(
        version=APP_VERSION,
        environment=APP_ENV,
    )


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health() -> HealthResponse:
    """
    Check the availability of the Supabase data source and model.

    HTTP 200 means the application endpoint itself is responding.
    The returned `status` distinguishes healthy from unhealthy
    dependencies.
    """
    result = prediction_health_check()

    return HealthResponse(
        **result
    )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@app.get(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict() -> PredictionResponse:
    """
    Generate the latest S&P 500 production prediction.

    Workflow:
        Supabase
        -> validation
        -> feature engineering
        -> frozen model
        -> trading signal
    """
    try:
        result = predict_current_market()
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "DATA_VALIDATION_ERROR",
                "message": str(exc),
            },
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "MODEL_ARTIFACT_MISSING",
                "message": (
                    "Production model artifacts are unavailable."
                ),
            },
        ) from exc
    except (ConnectionError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "DEPENDENCY_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc

    return PredictionResponse(
        **result
    )


# ---------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
