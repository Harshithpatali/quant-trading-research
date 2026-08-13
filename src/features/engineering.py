"""
Production feature engineering for the S&P 500 direction model.

IMPORTANT
---------
This module intentionally mirrors the feature definitions used in the
research notebooks and Notebook 14 production packaging.

The model expects exactly the feature names and order defined in
FEATURE_COLUMNS.

The function `build_feature_frame()` does not change the raw OHLCV
input. It returns a new DataFrame.

For live prediction, the latest row is generated using only information
available through that row's date. The `next_day_return` and `target`
columns are included only when the next day's Close is actually
available; they are never required for live prediction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Canonical feature list
# ---------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",
    "return_126d",
    "return_252d",
    "volatility_5d",
    "volatility_21d",
    "volatility_63d",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_200",
    "sma_50_vs_sma_200",
    "range_pct",
    "intraday_return",
    "overnight_return",
    "volume_ratio_20",
    "volume_ratio_63",
    "return_1d_lag1",
    "return_1d_lag2",
    "return_1d_lag3",
    "return_1d_lag5",
    "return_1d_lag10",
    "volatility_21d_lag1",
    "range_pct_lag1",
    "volume_ratio_20_lag1",
]


RAW_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj.Close",
    "Volume",
]


def _validate_raw_columns(df: pd.DataFrame) -> None:
    """Ensure the loader's canonical OHLCV schema is present."""
    missing = [
        column
        for column in RAW_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Feature engineering requires the canonical OHLCV schema. "
            "Missing columns: "
            + ", ".join(missing)
        )


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean working copy without modifying the input."""
    _validate_raw_columns(df)

    data = df[RAW_COLUMNS].copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    if data["Date"].isna().any():
        raise ValueError(
            "Feature engineering received invalid Date values."
        )

    for column in RAW_COLUMNS[1:]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data["Date"].duplicated().any():
        raise ValueError(
            "Feature engineering requires unique Date values."
        )

    if not data["Date"].is_monotonic_increasing:
        data = (
            data.sort_values("Date")
            .reset_index(drop=True)
        )

    return data


def build_feature_frame(
    df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Generate the complete production feature frame.

    Parameters
    ----------
    df:
        Canonical OHLCV DataFrame from src.data.loader.

    include_target:
        If True, calculate next_day_return and target when the next-day
        Close is available. If False, those columns are omitted.

    Returns
    -------
    pandas.DataFrame
        OHLCV data plus the exact research features.

    Notes
    -----
    All rolling features are calculated chronologically.
    No future value is used in the feature columns themselves.
    """
    data = _prepare_ohlcv(df)

    # ---------------------------------------------------------------
    # Returns
    # ---------------------------------------------------------------

    data["return_1d"] = (
        data["Close"].pct_change()
    )

    data["return_5d"] = (
        data["Close"].pct_change(5)
    )

    data["return_21d"] = (
        data["Close"].pct_change(21)
    )

    data["return_63d"] = (
        data["Close"].pct_change(63)
    )

    data["return_126d"] = (
        data["Close"].pct_change(126)
    )

    data["return_252d"] = (
        data["Close"].pct_change(252)
    )

    # ---------------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------------

    data["volatility_5d"] = (
        data["return_1d"]
        .rolling(5)
        .std()
        * np.sqrt(252)
    )

    data["volatility_21d"] = (
        data["return_1d"]
        .rolling(21)
        .std()
        * np.sqrt(252)
    )

    data["volatility_63d"] = (
        data["return_1d"]
        .rolling(63)
        .std()
        * np.sqrt(252)
    )

    # ---------------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------------

    data["sma_20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["sma_50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    data["sma_200"] = (
        data["Close"]
        .rolling(200)
        .mean()
    )

    data["price_to_sma_20"] = (
        data["Close"] /
        data["sma_20"] -
        1
    )

    data["price_to_sma_50"] = (
        data["Close"] /
        data["sma_50"] -
        1
    )

    data["price_to_sma_200"] = (
        data["Close"] /
        data["sma_200"] -
        1
    )

    data["sma_50_vs_sma_200"] = (
        data["sma_50"] /
        data["sma_200"] -
        1
    )

    # ---------------------------------------------------------------
    # OHLC / volume features
    # ---------------------------------------------------------------

    data["range_pct"] = (
        (
            data["High"] -
            data["Low"]
        ) /
        data["Close"]
    )

    data["intraday_return"] = (
        data["Close"] /
        data["Open"] -
        1
    )

    data["overnight_return"] = (
        data["Open"] /
        data["Close"].shift(1) -
        1
    )

    data["volume_ratio_20"] = (
        data["Volume"] /
        data["Volume"].rolling(20).mean()
    )

    data["volume_ratio_63"] = (
        data["Volume"] /
        data["Volume"].rolling(63).mean()
    )

    # ---------------------------------------------------------------
    # Lagged features
    # ---------------------------------------------------------------

    data["return_1d_lag1"] = (
        data["return_1d"].shift(1)
    )

    data["return_1d_lag2"] = (
        data["return_1d"].shift(2)
    )

    data["return_1d_lag3"] = (
        data["return_1d"].shift(3)
    )

    data["return_1d_lag5"] = (
        data["return_1d"].shift(5)
    )

    data["return_1d_lag10"] = (
        data["return_1d"].shift(10)
    )

    data["volatility_21d_lag1"] = (
        data["volatility_21d"].shift(1)
    )

    data["range_pct_lag1"] = (
        data["range_pct"].shift(1)
    )

    data["volume_ratio_20_lag1"] = (
        data["volume_ratio_20"].shift(1)
    )

    # ---------------------------------------------------------------
    # Target columns
    # ---------------------------------------------------------------
    #
    # These are useful for historical research only.
    # They are not required for a live prediction.
    #

    if include_target:
        data["next_day_return"] = (
            data["Close"].shift(-1) /
            data["Close"] -
            1
        )

        data["target"] = (
            data["next_day_return"] > 0
        ).astype("int8")

    return data


def get_model_features(
    df: pd.DataFrame,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Return exactly the columns expected by the frozen model.

    Parameters
    ----------
    df:
        Canonical OHLCV DataFrame.

    drop_incomplete:
        If True, remove rows containing NaN model features.
        For live prediction this should normally remain True.
    """
    features = build_feature_frame(
        df,
        include_target=False,
    )

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing:
        raise RuntimeError(
            "Feature generation failed. Missing model features: "
            + ", ".join(missing)
        )

    model_features = features[
        FEATURE_COLUMNS
    ].copy()

    if drop_incomplete:
        model_features = (
            model_features
            .dropna()
            .reset_index(drop=True)
        )

    return model_features


def get_latest_model_features(
    df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """
    Generate the latest complete feature row.

    Returns
    -------
    tuple
        (feature_date, one-row DataFrame)

    The returned DataFrame contains exactly one row and exactly the
    frozen model's feature columns.
    """
    features = build_feature_frame(
        df,
        include_target=False,
    )

    valid = features.dropna(
        subset=FEATURE_COLUMNS
    )

    if valid.empty:
        raise ValueError(
            "Not enough historical OHLCV data to generate a complete "
            "production feature row."
        )

    latest = valid.iloc[-1]

    feature_date = pd.Timestamp(
        latest["Date"]
    )

    model_features = (
        latest[FEATURE_COLUMNS]
        .to_frame()
        .T
        .reset_index(drop=True)
    )

    return feature_date, model_features


def validate_feature_schema(
    features: pd.DataFrame,
    expected_columns: Sequence[str] = FEATURE_COLUMNS,
) -> None:
    """
    Strictly validate model-feature schema before prediction.
    """
    expected = list(expected_columns)
    actual = list(features.columns)

    if actual != expected:
        raise ValueError(
            "Model feature schema mismatch.\n"
            f"Expected: {expected}\n"
            f"Received: {actual}"
        )

    if features.empty:
        raise ValueError(
            "Model feature DataFrame is empty."
        )

    if not np.isfinite(
        features.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Model feature DataFrame contains NaN or infinite values."
        )


def calculate_required_history() -> int:
    """
    Return the minimum practical raw-history length.

    The largest rolling window is 252 observations. Additional lagged
    transformations require existing rows, so 253 observations gives
    the latest row a complete feature vector.
    """
    return 253


__all__ = [
    "FEATURE_COLUMNS",
    "RAW_COLUMNS",
    "build_feature_frame",
    "get_model_features",
    "get_latest_model_features",
    "validate_feature_schema",
    "calculate_required_history",
]
