"""
Production FastAPI application wrapper.

P13 hardens the API entry point by separating application creation from
the module-level server runner. This makes the backend easier to test,
reuse, and deploy with Uvicorn/Gunicorn.

The actual prediction logic remains in:
    src.services.prediction_service
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
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
    CORS_ORIGINS,
)
from src.services.prediction_service import (
    prediction_health_check,
    predict_current_market,
)


APP_VERSION = "1.0.0"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """
    Application startup/shutdown lifecycle.

    Startup intentionally does not generate a trading prediction.
    The model is loaded lazily by the prediction service so that the
    application can start cleanly and health diagnostics can identify
    missing dependencies.
    """
    logger.info(
        "Starting %s version %s in %s environment.",
        APP_NAME,
        APP_VERSION,
        APP_ENV,
    )

    yield

    logger.info(
        "Shutting down %s.",
        APP_NAME,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title=APP_NAME,
        description=(
            "Production API for S&P 500 market-direction prediction "
            "using a frozen research model."
        ),
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.exception_handler(HTTPExceptionSafe)
    async def safe_http_exception_handler(
        request: Request,
        exc: HTTPExceptionSafe,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=exc.message,
            ).model_dump(),
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """
        Return a safe generic error response.

        Do not expose internal exception details, credentials, file
        paths, or stack traces to API consumers.
        """
        logger.exception(
            "Unhandled API exception on %s %s",
            request.method,
            request.url.path,
        )

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="INTERNAL_SERVER_ERROR",
                message="An internal server error occurred.",
            ).model_dump(),
        )

    @application.get(
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

    @application.get(
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

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
    )
    async def health() -> HealthResponse:
        """Check Supabase and model availability."""
        result = prediction_health_check()

        return HealthResponse(
            **result
        )

    @application.get(
        "/api/v1/predict",
        response_model=PredictionResponse,
        tags=["Prediction"],
    )
    async def predict() -> PredictionResponse:
        """Generate the latest production S&P 500 prediction."""
        try:
            result = predict_current_market()

        except ValueError as exc:
            raise HTTPExceptionSafe(
                status_code=422,
                error_code="DATA_VALIDATION_ERROR",
                message=str(exc),
            )

        except FileNotFoundError as exc:
            raise HTTPExceptionSafe(
                status_code=503,
                error_code="MODEL_ARTIFACT_MISSING",
                message="Production model artifacts are unavailable.",
            )

        except (ConnectionError, RuntimeError) as exc:
            raise HTTPExceptionSafe(
                status_code=503,
                error_code="DEPENDENCY_UNAVAILABLE",
                message=str(exc),
            )

        return PredictionResponse(
            **result
        )

    return application


class HTTPExceptionSafe(Exception):
    """
    Internal exception used to keep endpoint error construction
    consistent without leaking implementation details.
    """

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(message)


app = create_app()


__all__ = [
    "APP_VERSION",
    "app",
    "create_app",
]
