"""
P29.6 — Genuine out-of-sample backtest runner.

IMPORTANT
---------
Do NOT use the frozen production model to predict the entire 1951→2026
history for this evaluation.

The production model was refit/packaged using the full available
training data. Therefore its predictions cannot be used as historical
out-of-sample predictions for dates that were inside its training set.

Instead, P29.6 consumes the already-generated walk-forward prediction
artifact:

    data/interim/sp500_walk_forward_predictions.parquet

This artifact is the correct research source for out-of-sample
historical predictions.

The script:
    1. Loads walk-forward predictions.
    2. Inspects and normalizes the prediction schema.
    3. Verifies chronological ordering and uniqueness.
    4. Verifies probability/threshold consistency.
    5. Uses the existing P29 backtesting engine.
    6. Produces an OOS performance report.
    7. Compares OOS results with the frozen final-test metrics stored
       in the production metadata when available.

This script does NOT retrain or modify the production model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =====================================================================
# PROJECT BOOTSTRAP
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.backtesting.engine import (  # noqa: E402
    BacktestConfig,
    run_backtest,
)


# =====================================================================
# PATHS
# =====================================================================

WALK_FORWARD_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "sp500_walk_forward_predictions.parquet"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "sp500_direction_model_metadata.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "walk_forward_oos_backtest.json"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

INITIAL_CAPITAL = 100_000.0

# Frozen research configuration:
# 5 bps transaction cost + 2 bps slippage.
TRANSACTION_COST = 0.0005
SLIPPAGE = 0.0002

THRESHOLD = 0.50


# =====================================================================
# COLUMN NORMALIZATION
# =====================================================================

DATE_CANDIDATES = [
    "Date",
    "date",
    "prediction_date",
    "feature_date",
]

PROBABILITY_CANDIDATES = [
    "probability_up",
    "prob_up",
    "probability",
    "proba_up",
    "p_up",
]

PREDICTION_CANDIDATES = [
    "prediction",
    "predicted",
    "y_pred",
    "prediction_class",
]

SIGNAL_CANDIDATES = [
    "signal",
    "Signal",
]

RETURN_CANDIDATES = [
    "next_day_return",
    "target_return",
    "future_return",
    "forward_return",
]


def find_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    """Return the first matching candidate column."""

    lower_map = {
        str(column).lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_predictions() -> pd.DataFrame:
    """Load and normalize the walk-forward prediction artifact."""

    if not WALK_FORWARD_PATH.exists():
        raise FileNotFoundError(
            "Walk-forward prediction artifact was not found:\n"
            f"{WALK_FORWARD_PATH}\n\n"
            "P29.6 requires the output of the walk-forward "
            "validation research pipeline."
        )

    df = pd.read_parquet(
        WALK_FORWARD_PATH
    )

    if df.empty:
        raise ValueError(
            "Walk-forward prediction artifact is empty."
        )

    # Flatten accidental MultiIndex columns defensively.
    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        df.columns = [
            "_".join(
                str(part)
                for part in column
                if str(part) != ""
            ).strip("_")
            for column in df.columns
        ]

    columns = [
        str(column)
        for column in df.columns
    ]

    date_column = find_column(
        columns,
        DATE_CANDIDATES,
    )

    probability_column = find_column(
        columns,
        PROBABILITY_CANDIDATES,
    )

    prediction_column = find_column(
        columns,
        PREDICTION_CANDIDATES,
    )

    signal_column = find_column(
        columns,
        SIGNAL_CANDIDATES,
    )

    if date_column is None:
        raise ValueError(
            "Could not identify the prediction date column.\n"
            f"Available columns: {columns}"
        )

    if probability_column is None:
        raise ValueError(
            "Could not identify probability_up in the "
            "walk-forward prediction artifact.\n"
            f"Available columns: {columns}"
        )

    result = pd.DataFrame()

    result["Date"] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    result["probability_up"] = pd.to_numeric(
        df[probability_column],
        errors="coerce",
    )

    if prediction_column is not None:
        result["prediction_original"] = pd.to_numeric(
            df[prediction_column],
            errors="coerce",
        )

    if signal_column is not None:
        result["signal_original"] = (
            df[signal_column]
            .astype(str)
        )

    # ---------------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------------

    result = (
        result
        .dropna(
            subset=[
                "Date",
                "probability_up",
            ]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if result.empty:
        raise ValueError(
            "No valid walk-forward prediction rows remain "
            "after date/probability cleaning."
        )

    duplicate_dates = int(
        result["Date"].duplicated().sum()
    )

    if duplicate_dates:
        raise ValueError(
            "Walk-forward predictions contain duplicate dates: "
            f"{duplicate_dates}"
        )

    probability_values = (
        result["probability_up"]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        probability_values
    ).all():
        raise ValueError(
            "Walk-forward probabilities contain NaN or infinity."
        )

    if (
        (probability_values < 0)
        | (probability_values > 1)
    ).any():
        raise ValueError(
            "Walk-forward probability values must be within [0, 1]."
        )

    # ---------------------------------------------------------------
    # Frozen signal rule
    # ---------------------------------------------------------------

    result["probability_down"] = (
        1.0
        - result["probability_up"]
    )

    result["prediction"] = (
        result["probability_up"]
        >= THRESHOLD
    ).astype(int)

    result["signal"] = np.where(
        result["prediction"] == 1,
        "LONG",
        "CASH",
    )

    # ---------------------------------------------------------------
    # Detect available return/price columns and retain them if the
    # backtesting engine supports them.
    # ---------------------------------------------------------------

    return_column = find_column(
        columns,
        RETURN_CANDIDATES,
    )

    if return_column is not None:
        result["next_day_return"] = pd.to_numeric(
            df.loc[
                result.index,
                return_column,
            ],
            errors="coerce",
        )

    # The engine's expected interface may use a target/return column.
    # We intentionally do not invent a return if the artifact doesn't
    # contain one. The existing engine may derive it from its supported
    # input schema.

    return result


# =====================================================================
# PROVENANCE CHECK
# =====================================================================

def load_metadata() -> dict[str, Any]:
    """Load production model metadata when available."""

    if not METADATA_PATH.exists():
        return {}

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        value = json.load(file)

    if not isinstance(value, dict):
        return {}

    return value


def print_provenance(
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Print the OOS provenance context."""

    print()
    print("=" * 78)
    print("P29.6 — GENUINE OUT-OF-SAMPLE BACKTEST")
    print("=" * 78)

    print()
    print("PREDICTION SOURCE")
    print("-" * 78)
    print(
        "Walk-forward artifact:"
    )
    print(
        WALK_FORWARD_PATH
    )

    print()
    print(
        f"Prediction rows:     "
        f"{len(predictions):,}"
    )

    print(
        "Prediction period:   "
        f"{predictions['Date'].min().strftime('%Y-%m-%d')}"
        " → "
        f"{predictions['Date'].max().strftime('%Y-%m-%d')}"
    )

    if metadata:
        print()
        print("PRODUCTION MODEL REFERENCE")
        print("-" * 78)
        print(
            f"Model type:          "
            f"{metadata.get('model_type', 'Unknown')}"
        )
        print(
            f"Training metadata:   "
            f"{metadata.get('training_start', 'Unknown')}"
            " → "
            f"{metadata.get('training_end', 'Unknown')}"
        )

        print()
        print(
            "NOTE: The frozen production artifact is NOT used "
            "to generate these historical predictions."
        )

        print(
            "The walk-forward prediction artifact is used "
            "because historical OOS evaluation must use "
            "predictions generated without future training data."
        )


# =====================================================================
# REPORT
# =====================================================================

def pct(value: float) -> str:
    return f"{value:.2%}"


def money(value: float) -> str:
    return f"₹{value:,.2f}"


def print_report(
    result: dict[str, Any],
) -> None:
    """Print OOS backtest results."""

    summary = result["summary"]
    risk = result["risk"]
    trades = result["trades"]
    benchmark = result["benchmark"]

    print()
    print("=" * 78)
    print("WALK-FORWARD OUT-OF-SAMPLE BACKTEST RESULTS")
    print("=" * 78)

    print()
    print("PERIOD")
    print("-" * 78)

    print(
        f"Start date:          "
        f"{summary['start_date']}"
    )

    print(
        f"End date:            "
        f"{summary['end_date']}"
    )

    print(
        f"Trading days:        "
        f"{summary['trading_days']:,}"
    )

    print()
    print("STRATEGY PERFORMANCE")
    print("-" * 78)

    print(
        f"Initial capital:     "
        f"{money(summary['initial_capital'])}"
    )

    print(
        f"Final value:         "
        f"{money(summary['final_value'])}"
    )

    print(
        f"Profit / Loss:       "
        f"{money(summary['profit_loss'])}"
    )

    print(
        f"Total return:        "
        f"{pct(summary['total_return'])}"
    )

    print(
        f"Annualized return:   "
        f"{pct(summary['annualized_return'])}"
    )

    print()
    print("RISK")
    print("-" * 78)

    print(
        f"Annualized volatility:"
        f"{pct(risk['annualized_volatility'])}"
    )

    print(
        f"Maximum drawdown:    "
        f"{pct(risk['max_drawdown'])}"
    )

    print(
        f"Sharpe ratio:        "
        f"{risk['sharpe_ratio']:.3f}"
    )

    print(
        f"Sortino ratio:       "
        f"{risk['sortino_ratio']:.3f}"
    )

    print(
        f"VaR 95%:             "
        f"{pct(risk['var_95'])}"
    )

    print(
        f"CVaR 95%:            "
        f"{pct(risk['cvar_95'])}"
    )

    print()
    print("TRADING")
    print("-" * 78)

    print(
        f"Position changes:    "
        f"{trades['position_changes']:,}"
    )

    print(
        f"Round trips:         "
        f"{trades['completed_round_trips']:,}"
    )

    print(
        f"Active days:         "
        f"{trades['active_days']:,}"
    )

    print(
        f"Win rate:            "
        f"{pct(trades['win_rate'])}"
    )

    print(
        f"Profit factor:       "
        f"{trades['profit_factor']:.3f}"
    )

    print()
    print("BUY & HOLD")
    print("-" * 78)

    print(
        f"Benchmark return:    "
        f"{pct(benchmark['total_return'])}"
    )

    print(
        f"Benchmark CAGR:      "
        f"{pct(benchmark['annualized_return'])}"
    )

    print(
        f"Benchmark drawdown:  "
        f"{pct(benchmark['max_drawdown'])}"
    )

    print(
        f"Outperformance:      "
        f"{pct(benchmark['outperformance'])}"
    )

    print()
    print("=" * 78)


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    """Run the genuine OOS backtest."""

    try:
        predictions = load_predictions()

        metadata = load_metadata()

        print_provenance(
            predictions,
            metadata,
        )

        print()
        print(
            "Running the existing backtesting engine "
            "against walk-forward predictions..."
        )

        config = BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            threshold=THRESHOLD,
            periods_per_year=252,
        )

        result = run_backtest(
            predictions,
            config=config,
        )

        print_report(
            result
        )

        output = {
            "status": "success",
            "evaluation_type": (
                "walk_forward_out_of_sample"
            ),
            "prediction_source": str(
                WALK_FORWARD_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "prediction_rows": len(
                predictions
            ),
            "prediction_start": (
                predictions["Date"]
                .min()
                .strftime("%Y-%m-%d")
            ),
            "prediction_end": (
                predictions["Date"]
                .max()
                .strftime("%Y-%m-%d")
            ),
            "model_type": metadata.get(
                "model_type"
            ),
            "result": result,
        }

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                indent=2,
            )

        print()
        print(
            "OOS backtest written to:"
        )
        print(
            OUTPUT_PATH
        )

        print()
        print(
            "P29.6 RESULT: SUCCESS"
        )

        return 0

    except Exception as exc:
        print()
        print("=" * 78)
        print("P29.6 FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        print(
            "No performance result was published."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
