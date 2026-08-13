"""
Production OHLCV data-integrity validator.

This validator applies the same core integrity principles used during
the research notebooks, but is designed to run safely before production
feature generation and prediction.

Canonical input schema:
    Date
    Open
    High
    Low
    Close
    Adj.Close
    Volume
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj.Close",
    "Volume",
]

PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj.Close",
]

REQUIRED_NUMERIC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

# A small tolerance avoids false positives caused by floating-point
# representation while still detecting genuine OHLC inconsistencies.
PRICE_TOLERANCE = 1e-10


@dataclass
class ValidationReport:
    """Structured result returned by the production validator."""

    source: str
    valid: bool
    rows: int
    columns: list[str]
    date_start: str | None
    date_end: str | None
    duplicate_dates: int
    missing_dates: int
    null_values: dict[str, int]
    non_numeric_values: dict[str, int]
    invalid_high_low: int
    invalid_open_low: int
    invalid_open_high: int
    invalid_close_low: int
    invalid_close_high: int
    non_positive_prices: int
    negative_volume: int
    unsorted_dates: bool
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into a JSON-friendly dictionary."""
        return asdict(self)


def _count_invalid_numeric_values(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, int]:
    """Count values that cannot be interpreted as numbers."""
    counts: dict[str, int] = {}

    for column in columns:
        if column not in df.columns:
            continue

        original = df[column]
        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid = (
            original.notna()
            & converted.isna()
        )

        counts[column] = int(invalid.sum())

    return counts


def validate_ohlcv(
    df: pd.DataFrame,
    source_name: str = "S&P 500 dataset",
    raise_on_error: bool = True,
    require_all_columns: bool = True,
    require_sorted_dates: bool = True,
    require_unique_dates: bool = True,
) -> dict[str, Any]:
    """
    Validate an OHLCV DataFrame before production use.

    Checks include:
    - required schema
    - date parsing
    - duplicate dates
    - chronological ordering
    - null values
    - numeric conversion
    - High >= Low
    - Open within High/Low range
    - Close within High/Low range
    - positive prices
    - non-negative volume

    Parameters
    ----------
    df:
        Canonical OHLCV DataFrame.

    source_name:
        Human-readable source identifier.

    raise_on_error:
        Raise ValueError when validation fails.

    require_all_columns:
        Require the complete canonical schema.

    require_sorted_dates:
        Treat unsorted dates as an integrity error.

    require_unique_dates:
        Treat duplicate dates as an integrity error.

    Returns
    -------
    dict
        Structured validation report.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    rows = len(df)
    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    issues: list[str] = []

    if require_all_columns and missing_columns:
        issues.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if rows == 0:
        issues.append(
            "Dataset contains zero rows."
        )

    # Date handling
    if "Date" in df.columns:
        parsed_dates = pd.to_datetime(
            df["Date"],
            errors="coerce",
        )

        invalid_dates = int(
            parsed_dates.isna().sum()
        )

        if invalid_dates:
            issues.append(
                f"Invalid Date values: {invalid_dates}"
            )

        duplicate_dates = int(
            parsed_dates.duplicated().sum()
        )

        if (
            require_unique_dates
            and duplicate_dates > 0
        ):
            issues.append(
                f"Duplicate Date values: {duplicate_dates}"
            )

        unsorted_dates = bool(
            not parsed_dates.is_monotonic_increasing
        )

        if (
            require_sorted_dates
            and unsorted_dates
        ):
            issues.append(
                "Date column is not sorted chronologically."
            )

        valid_dates = parsed_dates.dropna()

        date_start = (
            valid_dates.min().strftime("%Y-%m-%d")
            if len(valid_dates)
            else None
        )

        date_end = (
            valid_dates.max().strftime("%Y-%m-%d")
            if len(valid_dates)
            else None
        )

        # Detect gaps between consecutive observations.
        # This is informational rather than a strict failure because
        # weekends and market holidays naturally create gaps.
        if len(valid_dates) > 1:
            sorted_unique_dates = (
                valid_dates
                .drop_duplicates()
                .sort_values()
            )

            day_gaps = (
                sorted_unique_dates
                .diff()
                .dt.days
                .dropna()
            )

            # A gap larger than 7 calendar days can indicate a data
            # ingestion problem. It is reported, not automatically
            # rejected, because long market closures can occur.
            missing_dates = int(
                (day_gaps > 7).sum()
            )
        else:
            missing_dates = 0

    else:
        duplicate_dates = 0
        unsorted_dates = False
        date_start = None
        date_end = None
        missing_dates = 0

        if require_all_columns:
            issues.append(
                "Date column is missing."
            )

    # Null counts
    null_values = {
        column: int(df[column].isna().sum())
        for column in EXPECTED_COLUMNS
        if column in df.columns
    }

    for column, count in null_values.items():
        if count > 0:
            issues.append(
                f"Null values in {column}: {count}"
            )

    # Numeric checks
    numeric_columns = [
        column
        for column in PRICE_COLUMNS + ["Volume"]
        if column in df.columns
    ]

    non_numeric_values = _count_invalid_numeric_values(
        df,
        numeric_columns,
    )

    for column, count in non_numeric_values.items():
        if count > 0:
            issues.append(
                f"Non-numeric values in {column}: {count}"
            )

    # Work on numeric copies so the validator itself never mutates
    # the caller's DataFrame.
    numeric = df.copy()

    for column in numeric_columns:
        numeric[column] = pd.to_numeric(
            numeric[column],
            errors="coerce",
        )

    # OHLC relationship checks
    if all(
        column in numeric.columns
        for column in [
            "High",
            "Low",
        ]
    ):
        invalid_high_low = int(
            (
                numeric["High"]
                + PRICE_TOLERANCE
                < numeric["Low"]
            ).sum()
        )
    else:
        invalid_high_low = 0

    if all(
        column in numeric.columns
        for column in [
            "Open",
            "Low",
        ]
    ):
        invalid_open_low = int(
            (
                numeric["Open"]
                + PRICE_TOLERANCE
                < numeric["Low"]
            ).sum()
        )
    else:
        invalid_open_low = 0

    if all(
        column in numeric.columns
        for column in [
            "Open",
            "High",
        ]
    ):
        invalid_open_high = int(
            (
                numeric["Open"]
                - PRICE_TOLERANCE
                > numeric["High"]
            ).sum()
        )
    else:
        invalid_open_high = 0

    if all(
        column in numeric.columns
        for column in [
            "Close",
            "Low",
        ]
    ):
        invalid_close_low = int(
            (
                numeric["Close"]
                + PRICE_TOLERANCE
                < numeric["Low"]
            ).sum()
        )
    else:
        invalid_close_low = 0

    if all(
        column in numeric.columns
        for column in [
            "Close",
            "High",
        ]
    ):
        invalid_close_high = int(
            (
                numeric["Close"]
                - PRICE_TOLERANCE
                > numeric["High"]
            ).sum()
        )
    else:
        invalid_close_high = 0

    relationship_errors = {
        "invalid_high_low": invalid_high_low,
        "invalid_open_low": invalid_open_low,
        "invalid_open_high": invalid_open_high,
        "invalid_close_low": invalid_close_low,
        "invalid_close_high": invalid_close_high,
    }

    for name, count in relationship_errors.items():
        if count > 0:
            issues.append(
                f"{name}: {count}"
            )

    # Positive-price checks
    price_frame = numeric[
        [
            column
            for column in PRICE_COLUMNS
            if column in numeric.columns
        ]
    ]

    if not price_frame.empty:
        non_positive_prices = int(
            (price_frame <= 0)
            .any(axis=1)
            .sum()
        )
    else:
        non_positive_prices = 0

    if non_positive_prices:
        issues.append(
            "Non-positive price observations: "
            f"{non_positive_prices}"
        )

    # Volume should never be negative.
    if "Volume" in numeric.columns:
        negative_volume = int(
            (numeric["Volume"] < 0).sum()
        )
    else:
        negative_volume = 0

    if negative_volume:
        issues.append(
            f"Negative volume observations: {negative_volume}"
        )

    # Remove duplicate messages while preserving order.
    issues = list(dict.fromkeys(issues))

    valid = len(issues) == 0

    report = ValidationReport(
        source=source_name,
        valid=valid,
        rows=rows,
        columns=actual_columns,
        date_start=date_start,
        date_end=date_end,
        duplicate_dates=duplicate_dates,
        missing_dates=missing_dates,
        null_values=null_values,
        non_numeric_values=non_numeric_values,
        invalid_high_low=invalid_high_low,
        invalid_open_low=invalid_open_low,
        invalid_open_high=invalid_open_high,
        invalid_close_low=invalid_close_low,
        invalid_close_high=invalid_close_high,
        non_positive_prices=non_positive_prices,
        negative_volume=negative_volume,
        unsorted_dates=unsorted_dates,
        issues=issues,
    ).to_dict()

    if (
        raise_on_error
        and not valid
    ):
        raise ValueError(
            f"{source_name}: data-integrity validation failed: "
            f"{issues}"
        )

    return report


def print_validation_report(
    report: dict[str, Any],
) -> None:
    """Print a compact human-readable validation report."""

    print("=" * 72)
    print("OHLCV DATA-INTEGRITY VALIDATION")
    print("=" * 72)

    print(f"Source: {report['source']}")
    print(f"Valid: {report['valid']}")
    print(f"Rows: {report['rows']}")

    print(
        "Date range: "
        f"{report['date_start']} → "
        f"{report['date_end']}"
    )

    print(
        "Duplicate dates: "
        f"{report['duplicate_dates']}"
    )

    print(
        "Large calendar gaps (>7 days): "
        f"{report['missing_dates']}"
    )

    print(
        "Unsorted dates: "
        f"{report['unsorted_dates']}"
    )

    print(
        "Invalid High < Low: "
        f"{report['invalid_high_low']}"
    )

    print(
        "Invalid Open < Low: "
        f"{report['invalid_open_low']}"
    )

    print(
        "Invalid Open > High: "
        f"{report['invalid_open_high']}"
    )

    print(
        "Invalid Close < Low: "
        f"{report['invalid_close_low']}"
    )

    print(
        "Invalid Close > High: "
        f"{report['invalid_close_high']}"
    )

    print(
        "Non-positive prices: "
        f"{report['non_positive_prices']}"
    )

    print(
        "Negative volume: "
        f"{report['negative_volume']}"
    )

    if report["issues"]:
        print("\nIssues:")
        for issue in report["issues"]:
            print(f" - {issue}")
    else:
        print("\nNo data-integrity issues detected.")

    print("=" * 72)


__all__ = [
    "EXPECTED_COLUMNS",
    "ValidationReport",
    "validate_ohlcv",
    "print_validation_report",
]
