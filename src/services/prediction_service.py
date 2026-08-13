"""
Production prediction orchestration service.

This service combines the production components in the correct order:

    Supabase
        ↓
    OHLCV loader
        ↓
    Data-integrity validator
        ↓
    Feature engineering
        ↓
    Frozen model
        ↓
    Trading signal

The service contains no training or optimization logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import MIN_HISTORY_DAYS
from src.data.loader import SupabaseDataLoader
from src.data.validator import validate_ohlcv
from src.models.predictor import ModelPredictor, get_predictor
from src.strategy.signal import (
    TradingSignal,
    get_position_size,
    signal_from_prediction,
    validate_signal,
)


class PredictionService:
    """
    Orchestrate the complete production prediction workflow.

    A PredictionService owns a data loader and a frozen model predictor.
    Dependencies can be injected for testing.
    """

    def __init__(
        self,
        data_loader: SupabaseDataLoader | None = None,
        predictor: ModelPredictor | None = None,
    ) -> None:
        self.data_loader = (
            data_loader
            if data_loader is not None
            else SupabaseDataLoader()
        )

        self.predictor = (
            predictor
            if predictor is not None
            else get_predictor()
        )

    def _load_history(
        self,
        history_limit: int = MIN_HISTORY_DAYS,
    ) -> pd.DataFrame:
        """
        Load enough recent history for feature generation.

        The loader returns chronological data even though Supabase is
        queried in descending order for latest-data retrieval.
        """
        if history_limit < MIN_HISTORY_DAYS:
            raise ValueError(
                f"history_limit must be at least "
                f"{MIN_HISTORY_DAYS} observations."
            )

        df = self.data_loader.load_latest(
            limit=history_limit
        )

        if len(df) < MIN_HISTORY_DAYS:
            raise ValueError(
                "Insufficient historical data for production "
                f"prediction. Required at least {MIN_HISTORY_DAYS} "
                f"rows, received {len(df)}."
            )

        return df

    def _validate_data(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run the production OHLCV integrity gate."""
        return validate_ohlcv(
            df,
            source_name="Supabase sp500_daily",
            raise_on_error=True,
            require_all_columns=True,
            require_sorted_dates=True,
            require_unique_dates=True,
        )

    def predict(
        self,
        history_limit: int = MIN_HISTORY_DAYS,
    ) -> dict[str, Any]:
        """
        Generate the current production trading prediction.

        Returns a JSON-serializable dictionary containing:
        - prediction date
        - signal
        - model probabilities
        - frozen threshold
        - position
        - data validation summary
        - model metadata
        """
        requested_at = datetime.now(
            timezone.utc
        ).isoformat()

        # -------------------------------------------------------------
        # 1. Load market data
        # -------------------------------------------------------------
        df = self._load_history(
            history_limit=history_limit
        )

        # -------------------------------------------------------------
        # 2. Validate market data
        # -------------------------------------------------------------
        validation = self._validate_data(
            df
        )

        # -------------------------------------------------------------
        # 3. Generate frozen-model prediction
        # -------------------------------------------------------------
        model_result = self.predictor.predict_dataframe(
            df
        )

        # -------------------------------------------------------------
        # 4. Convert probability into trading signal
        # -------------------------------------------------------------
        signal = signal_from_prediction(
            model_result
        )

        validate_signal(
            signal
        )

        position = get_position_size(
            signal
        )

        # -------------------------------------------------------------
        # 5. Build production response
        # -------------------------------------------------------------
        return {
            "status": "success",
            "requested_at": requested_at,
            "prediction_date": model_result[
                "feature_date"
            ],
            "signal": signal.signal,
            "prediction": signal.prediction,
            "position": position,
            "probability_up": signal.probability_up,
            "probability_down": signal.probability_down,
            "threshold": signal.threshold,
            "confidence": signal.confidence,
            "model": {
                "type": signal.model_type,
                "feature_count": model_result[
                    "feature_count"
                ],
            },
            "data": {
                "source": validation[
                    "source"
                ],
                "rows_used": validation[
                    "rows"
                ],
                "date_start": validation[
                    "date_start"
                ],
                "date_end": validation[
                    "date_end"
                ],
                "validation_passed": validation[
                    "valid"
                ],
            },
        }

    def health_check(self) -> dict[str, Any]:
        """
        Check whether the core prediction dependencies are available.

        This does not generate a trading signal.
        """
        result: dict[str, Any] = {
            "status": "healthy",
            "supabase": "unknown",
            "model": "unknown",
        }

        try:
            rows = self.data_loader.count_rows()
            result["supabase"] = {
                "status": "connected",
                "table": self.data_loader.table,
                "rows": rows,
            }
        except Exception as exc:
            result["status"] = "unhealthy"
            result["supabase"] = {
                "status": "error",
                "error": str(exc),
            }

        try:
            info = self.predictor.get_model_info()
            result["model"] = {
                "status": "loaded",
                "type": info.get("model_type"),
                "feature_count": info.get(
                    "feature_count"
                ),
                "threshold": info.get(
                    "probability_threshold"
                ),
            }
        except Exception as exc:
            result["status"] = "unhealthy"
            result["model"] = {
                "status": "error",
                "error": str(exc),
            }

        return result


# ---------------------------------------------------------------------
# Application-level service instance
# ---------------------------------------------------------------------

_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Return the application-wide prediction service."""
    global _service

    if _service is None:
        _service = PredictionService()

    return _service


def predict_current_market(
    history_limit: int = MIN_HISTORY_DAYS,
) -> dict[str, Any]:
    """Convenience function for API and dashboard layers."""
    return get_prediction_service().predict(
        history_limit=history_limit
    )


def prediction_health_check() -> dict[str, Any]:
    """Convenience function for health endpoints."""
    return get_prediction_service().health_check()


__all__ = [
    "PredictionService",
    "get_prediction_service",
    "predict_current_market",
    "prediction_health_check",
]
