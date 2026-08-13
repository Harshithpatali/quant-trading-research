"""
Production test suite for the S&P 500 Quant Trading application.

P23 covers deterministic, dependency-light tests that can run in
GitHub Actions without requiring a live Supabase connection.

Test areas:
    - OHLCV data-integrity validation
    - frozen signal decision rules
    - prediction response contract
    - configuration/security behavior
    - FastAPI application route registration

Live Supabase and model-inference tests remain outside this unit suite
because CI should not depend on production credentials or external data.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.schemas import PredictionResponse
from backend.security import require_api_key
from src.data.validator import validate_ohlcv
from src.strategy.signal import (
    generate_signal,
    get_position_size,
    signal_from_prediction,
    validate_signal,
)


# ---------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------

def valid_ohlcv_frame() -> pd.DataFrame:
    """Return a minimal valid OHLCV frame."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2026-08-10",
                    "2026-08-11",
                    "2026-08-12",
                ]
            ),
            "Open": [6400.0, 6420.0, 6440.0],
            "High": [6450.0, 6460.0, 6480.0],
            "Low": [6380.0, 6400.0, 6420.0],
            "Close": [6430.0, 6450.0, 6470.0],
            "Adj.Close": [6430.0, 6450.0, 6470.0],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )


# ---------------------------------------------------------------------
# OHLCV validation
# ---------------------------------------------------------------------

def test_valid_ohlcv_passes() -> None:
    """A valid chronological OHLCV dataset must pass."""
    df = valid_ohlcv_frame()

    report = validate_ohlcv(
        df,
        source_name="unit-test",
        raise_on_error=False,
        require_all_columns=True,
        require_sorted_dates=True,
        require_unique_dates=True,
    )

    assert report["valid"] is True
    assert not report["issues"]


def test_high_below_low_is_rejected() -> None:
    """High < Low must be treated as a data-integrity failure."""
    df = valid_ohlcv_frame()
    df.loc[1, "High"] = 6390.0

    with pytest.raises(ValueError):
        validate_ohlcv(
            df,
            source_name="invalid-high-low",
            raise_on_error=True,
            require_all_columns=True,
            require_sorted_dates=True,
            require_unique_dates=True,
        )


def test_duplicate_dates_are_rejected() -> None:
    """Duplicate trading dates must be rejected."""
    df = valid_ohlcv_frame()
    df.loc[2, "Date"] = df.loc[1, "Date"]

    with pytest.raises(ValueError):
        validate_ohlcv(
            df,
            source_name="duplicate-date",
            raise_on_error=True,
            require_all_columns=True,
            require_sorted_dates=True,
            require_unique_dates=True,
        )


def test_unsorted_dates_are_rejected() -> None:
    """Production data must be chronological."""
    df = valid_ohlcv_frame()
    df = df.iloc[
        [2, 0, 1]
    ].reset_index(drop=True)

    with pytest.raises(ValueError):
        validate_ohlcv(
            df,
            source_name="unsorted-date",
            raise_on_error=True,
            require_all_columns=True,
            require_sorted_dates=True,
            require_unique_dates=True,
        )


# ---------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------

def test_probability_at_threshold_is_long() -> None:
    """The frozen rule uses >= threshold for LONG."""
    signal = generate_signal(
        probability_up=0.50,
        threshold=0.50,
    )

    validate_signal(signal)

    assert signal.prediction == 1
    assert signal.signal == "LONG"
    assert get_position_size(signal) == 1.0


def test_probability_below_threshold_is_cash() -> None:
    """Probability below threshold must produce CASH."""
    signal = generate_signal(
        probability_up=0.4999,
        threshold=0.50,
    )

    validate_signal(signal)

    assert signal.prediction == 0
    assert signal.signal == "CASH"
    assert get_position_size(signal) == 0.0


def test_probabilities_sum_to_one() -> None:
    """Generated UP/DOWN probabilities must sum to one."""
    signal = generate_signal(
        probability_up=0.731,
        threshold=0.50,
    )

    assert (
        abs(
            signal.probability_up
            + signal.probability_down
            - 1.0
        )
        < 1e-12
    )


def test_prediction_result_is_recomputed_from_probability() -> None:
    """Signal conversion must use probability and threshold as source of truth."""
    result = {
        "probability_up": 0.60,
        "probability_down": 0.40,
        "prediction": 0,
        "signal": "CASH",
        "threshold": 0.50,
        "model_type": "Random Forest",
        "feature_count": 26,
        "feature_date": "2026-08-13",
    }

    signal = signal_from_prediction(result)

    assert signal.prediction == 1
    assert signal.signal == "LONG"


# ---------------------------------------------------------------------
# API schema
# ---------------------------------------------------------------------

def production_prediction_payload() -> dict:
    """Return a representative valid production response."""
    return {
        "status": "success",
        "requested_at": "2026-08-13T17:53:24.597747+00:00",
        "prediction_date": "2026-08-13",
        "signal": "LONG",
        "prediction": 1,
        "position": 1.0,
        "probability_up": 0.5283159035900113,
        "probability_down": 0.47168409640998865,
        "threshold": 0.5,
        "confidence": 0.0566318071800227,
        "model": {
            "type": "Random Forest",
            "feature_count": 26,
        },
        "data": {
            "source": "Supabase sp500_daily",
            "rows_used": 253,
            "date_start": "2025-08-12",
            "date_end": "2026-08-13",
            "validation_passed": True,
        },
    }


def test_prediction_response_schema() -> None:
    """The production prediction payload must satisfy the API contract."""
    response = PredictionResponse(
        **production_prediction_payload()
    )

    assert response.status == "success"
    assert response.signal == "LONG"
    assert response.prediction == 1
    assert response.model.feature_count == 26
    assert response.data.validation_passed is True


# ---------------------------------------------------------------------
# FastAPI route registration
# ---------------------------------------------------------------------

def test_root_route_registered() -> None:
    """The root endpoint must exist."""
    paths = {
        route.path
        for route in app.routes
    }

    assert "/" in paths


def test_health_route_registered() -> None:
    """The health endpoint must exist."""
    paths = {
        route.path
        for route in app.routes
    }

    assert "/health" in paths


def test_prediction_route_registered() -> None:
    """The protected prediction endpoint must exist."""
    paths = {
        route.path
        for route in app.routes
    }

    assert "/api/v1/predict" in paths


# ---------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------

def test_security_allows_local_mode_when_key_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty API_SECRET_KEY means local-development authentication is off."""
    import backend.security as security

    monkeypatch.setattr(
        security,
        "API_SECRET_KEY",
        "",
    )

    require_api_key(
        api_key=None
    )


def test_security_rejects_invalid_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured authentication must reject an invalid key."""
    import backend.security as security

    monkeypatch.setattr(
        security,
        "API_SECRET_KEY",
        "expected-secret",
    )

    with pytest.raises(Exception):
        require_api_key(
            api_key="wrong-secret"
        )
