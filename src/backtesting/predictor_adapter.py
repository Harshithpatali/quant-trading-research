from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.engineering import (
    FEATURE_COLUMNS,
    build_feature_frame,
    validate_feature_schema,
)
from src.models.predictor import get_predictor


def generate_historical_predictions(
    ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate historical model probabilities using the frozen model.

    Parameters
    ----------
    ohlcv:
        Canonical OHLCV DataFrame containing Date, Open, High, Low,
        Close, Adj.Close and Volume.

    Returns
    -------
    pd.DataFrame
        Columns:
            Date
            probability_up
            probability_down
            prediction
            signal

    Notes
    -----
    The feature row for date t only uses information available through
    date t. The backtesting engine subsequently shifts the resulting
    position to t+1 before applying returns. This prevents same-day
    close-to-close look-ahead bias.
    """

    if ohlcv.empty:
        raise ValueError(
            "Cannot generate historical predictions from an empty "
            "OHLCV DataFrame."
        )

    features = build_feature_frame(
        ohlcv,
        include_target=False,
    )

    required = [
        "Date",
        *FEATURE_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            "Historical feature frame is missing required columns: "
            + ", ".join(missing)
        )

    valid = (
        features[
            required
        ]
        .dropna(
            subset=FEATURE_COLUMNS
        )
        .copy()
        .reset_index(drop=True)
    )

    if valid.empty:
        raise ValueError(
            "No complete historical model-feature rows are available."
        )

    model_features = valid[
        FEATURE_COLUMNS
    ].copy()

    validate_feature_schema(
        model_features,
        expected_columns=FEATURE_COLUMNS,
    )

    predictor = get_predictor()

    probabilities = predictor.model.predict_proba(
        model_features
    )[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Frozen model returned non-finite historical probabilities."
        )

    if (
        (probabilities < 0)
        | (probabilities > 1)
    ).any():
        raise ValueError(
            "Frozen model returned probabilities outside [0, 1]."
        )

    predictions = (
        probabilities
        >= predictor.threshold
    ).astype(int)

    result = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                valid["Date"]
            ).reset_index(drop=True),
            "probability_up": probabilities,
            "probability_down": (
                1.0 - probabilities
            ),
            "prediction": predictions,
            "signal": np.where(
                predictions == 1,
                "LONG",
                "CASH",
            ),
        }
    )

    if result["Date"].duplicated().any():
        raise ValueError(
            "Historical prediction output contains duplicate dates."
        )

    if not result["Date"].is_monotonic_increasing:
        raise ValueError(
            "Historical prediction output is not chronological."
        )

    return result


def prepare_backtest_input(
    ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine canonical OHLCV data with frozen-model probabilities.

    The returned DataFrame is ready for ``run_backtest``.
    """

    if "Date" not in ohlcv.columns:
        raise ValueError(
            "OHLCV data must contain a Date column."
        )

    market = ohlcv.copy()

    market["Date"] = pd.to_datetime(
        market["Date"],
        errors="coerce",
    )

    market = (
        market
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    predictions = generate_historical_predictions(
        market
    )

    result = market.merge(
        predictions,
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    if result.empty:
        raise ValueError(
            "No overlapping dates between OHLCV data and "
            "historical model predictions."
        )

    result = (
        result
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return result


__all__ = [
    "generate_historical_predictions",
    "prepare_backtest_input",
]
