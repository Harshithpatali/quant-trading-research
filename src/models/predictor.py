"""
Production model inference service.

Loads the frozen S&P 500 direction model packaged by Notebook 14 and
generates predictions from the exact 26-feature production schema.

This module does NOT retrain, tune, or modify the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_CONFIG_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_THRESHOLD,
)
from src.features.engineering import (
    FEATURE_COLUMNS,
    get_latest_model_features,
    validate_feature_schema,
)


class ModelPredictor:
    """
    Load and run the frozen production model.

    The model is loaded once during initialization and reused for
    subsequent predictions.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
        config_path: Path = MODEL_CONFIG_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.config_path = Path(config_path)

        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self.config: dict[str, Any] = {}

        self._load_artifacts()
        self._validate_artifacts()

    # -----------------------------------------------------------------
    # Artifact loading
    # -----------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load model, metadata, and frozen configuration."""
        missing = [
            str(path)
            for path in [
                self.model_path,
                self.metadata_path,
                self.config_path,
            ]
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Required production model artifacts are missing:\n"
                + "\n".join(missing)
            )

        try:
            self.model = joblib.load(
                self.model_path
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the production model artifact."
            ) from exc

        try:
            self.metadata = json.loads(
                self.metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load production model metadata."
            ) from exc

        try:
            self.config = json.loads(
                self.config_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load frozen model configuration."
            ) from exc

    # -----------------------------------------------------------------
    # Artifact validation
    # -----------------------------------------------------------------

    def _validate_artifacts(self) -> None:
        """Ensure the packaged artifact matches the frozen schema."""
        metadata_features = self.metadata.get(
            "feature_columns"
        )

        if metadata_features is None:
            raise ValueError(
                "Production model metadata does not contain "
                "'feature_columns'."
            )

        if list(metadata_features) != FEATURE_COLUMNS:
            raise ValueError(
                "Production model feature schema does not match "
                "src.features.engineering.FEATURE_COLUMNS."
            )

        metadata_count = self.metadata.get(
            "feature_count"
        )

        if metadata_count != len(FEATURE_COLUMNS):
            raise ValueError(
                "Production model feature_count does not match "
                "the production feature schema."
            )

        configured_threshold = self.config.get(
            "probability_threshold"
        )

        if configured_threshold is None:
            raise ValueError(
                "Frozen configuration does not contain "
                "'probability_threshold'."
            )

        self.threshold = float(
            configured_threshold
        )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "Frozen probability threshold must be between 0 and 1."
            )

        # Compare with the environment value only as a safety check.
        # The frozen model configuration remains authoritative.
        if abs(
            self.threshold -
            float(MODEL_THRESHOLD)
        ) > 1e-12:
            raise ValueError(
                "MODEL_THRESHOLD in the environment does not match "
                "the frozen research configuration. "
                "Do not override the frozen threshold."
            )

        if not hasattr(
            self.model,
            "predict_proba"
        ):
            raise TypeError(
                "The production model does not expose predict_proba()."
            )

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    def predict_features(
        self,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Predict from an already-generated feature DataFrame.

        Exactly one row is expected for production inference.
        """
        validate_feature_schema(
            features,
            expected_columns=FEATURE_COLUMNS,
        )

        if len(features) != 1:
            raise ValueError(
                "Production prediction requires exactly one feature row."
            )

        X = features[
            FEATURE_COLUMNS
        ].copy()

        probability_up = float(
            self.model.predict_proba(X)[0, 1]
        )

        if not np.isfinite(
            probability_up
        ):
            raise ValueError(
                "Model returned a non-finite probability."
            )

        if not 0.0 <= probability_up <= 1.0:
            raise ValueError(
                "Model returned a probability outside [0, 1]."
            )

        prediction = int(
            probability_up >= self.threshold
        )

        signal = (
            "LONG"
            if prediction == 1
            else "CASH"
        )

        probability_down = (
            1.0 - probability_up
        )

        return {
            "probability_up": probability_up,
            "probability_down": probability_down,
            "prediction": prediction,
            "signal": signal,
            "threshold": self.threshold,
            "model_type": self.metadata.get(
                "model_type"
            ),
            "feature_count": len(FEATURE_COLUMNS),
        }

    def predict_dataframe(
        self,
        ohlcv: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate a prediction from canonical OHLCV history.

        The latest complete feature row is used.
        """
        feature_date, features = (
            get_latest_model_features(
                ohlcv
            )
        )

        result = self.predict_features(
            features
        )

        result["feature_date"] = (
            feature_date.strftime("%Y-%m-%d")
        )

        return result

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Return safe model metadata for API/dashboard display."""
        return {
            "model_type": self.metadata.get(
                "model_type"
            ),
            "task": self.metadata.get(
                "task"
            ),
            "feature_count": len(
                FEATURE_COLUMNS
            ),
            "feature_columns": FEATURE_COLUMNS.copy(),
            "probability_threshold": self.threshold,
            "signal_type": self.config.get(
                "signal_type"
            ),
            "transaction_cost_bps": self.config.get(
                "transaction_cost_bps"
            ),
            "slippage_bps": self.config.get(
                "slippage_bps"
            ),
            "training_rows": self.metadata.get(
                "training_rows"
            ),
            "training_start": self.metadata.get(
                "training_start"
            ),
            "training_end": self.metadata.get(
                "training_end"
            ),
            "research_verdict": self.metadata.get(
                "research_verdict"
            ),
        }


# ---------------------------------------------------------------------
# Singleton-style application model
# ---------------------------------------------------------------------

_predictor: ModelPredictor | None = None


def get_predictor() -> ModelPredictor:
    """
    Return the application-wide predictor instance.

    The model is loaded only once per Python process.
    """
    global _predictor

    if _predictor is None:
        _predictor = ModelPredictor()

    return _predictor


def predict_latest(
    ohlcv: pd.DataFrame,
) -> dict[str, Any]:
    """Convenience function for the latest production prediction."""
    return get_predictor().predict_dataframe(
        ohlcv
    )


__all__ = [
    "ModelPredictor",
    "get_predictor",
    "predict_latest",
]
