"""
P29.3 — Local historical backtest runner.

This script:
    1. Loads historical S&P 500 OHLCV data from Supabase.
    2. Generates historical probabilities with the frozen production model.
    3. Runs the P29 backtesting engine.
    4. Prints strategy, risk, trade, and benchmark statistics.

This is a local verification script. It does not retrain the model,
modify Supabase data, or expose secrets.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Project-root bootstrap
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


from src.backtesting.engine import (  # noqa: E402
    BacktestConfig,
    run_backtest,
)
from src.backtesting.predictor_adapter import (  # noqa: E402
    prepare_backtest_input,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "",
).strip()

SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "",
).strip()

SUPABASE_TABLE = os.getenv(
    "SUPABASE_TABLE",
    "sp500_daily",
).strip()

INITIAL_CAPITAL = float(
    os.getenv(
        "BACKTEST_INITIAL_CAPITAL",
        "100000",
    )
)

TRANSACTION_COST = float(
    os.getenv(
        "BACKTEST_TRANSACTION_COST",
        "0.0005",
    )
)

SLIPPAGE = float(
    os.getenv(
        "BACKTEST_SLIPPAGE",
        "0.0005",
    )
)

THRESHOLD = float(
    os.getenv(
        "BACKTEST_THRESHOLD",
        "0.50",
    )
)

PAGE_SIZE = 1000


# ---------------------------------------------------------------------
# Supabase loading
# ---------------------------------------------------------------------

def validate_configuration() -> None:
    """Validate required environment variables."""

    missing = []

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")

    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")

    if not SUPABASE_TABLE:
        missing.append("SUPABASE_TABLE")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


def load_all_market_data() -> pd.DataFrame:
    """
    Load all S&P 500 daily rows from Supabase using REST pagination.

    The query is read-only and ordered chronologically.
    """

    endpoint = (
        f"{SUPABASE_URL.rstrip('/')}"
        f"/rest/v1/{SUPABASE_TABLE}"
    )

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": (
            f"Bearer {SUPABASE_KEY}"
        ),
    }

    rows: list[dict[str, Any]] = []

    offset = 0

    while True:
        params = {
            "select": (
                "date,open,high,low,close,adj_close,volume"
            ),
            "order": "date.asc",
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=60,
        )

        if not response.ok:
            raise RuntimeError(
                "Supabase historical-data request failed "
                f"with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        page = response.json()

        if not isinstance(page, list):
            raise RuntimeError(
                "Supabase returned an unexpected response."
            )

        rows.extend(page)

        print(
            f"Loaded {len(rows):,} market rows...",
            flush=True,
        )

        if len(page) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    if not rows:
        raise RuntimeError(
            "Supabase returned zero market-data rows."
        )

    df = pd.DataFrame(rows)

    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj.Close",
            "volume": "Volume",
        }
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj.Close",
        "Volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna(
            subset=[
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def pct(value: float) -> str:
    """Format a decimal as a percentage."""

    return f"{value:.2%}"


def money(value: float) -> str:
    """Format a monetary value."""

    return f"₹{value:,.2f}"


def print_report(
    result: dict[str, Any],
) -> None:
    """Print a readable production backtest report."""

    summary = result["summary"]
    risk = result["risk"]
    trades = result["trades"]
    benchmark = result["benchmark"]

    print()
    print("=" * 72)
    print(
        "S&P 500 QUANT STRATEGY — HISTORICAL BACKTEST"
    )
    print("=" * 72)

    print()
    print("PERIOD")
    print("-" * 72)
    print(
        f"Start date:          {summary['start_date']}"
    )
    print(
        f"End date:            {summary['end_date']}"
    )
    print(
        f"Trading days:        {summary['trading_days']:,}"
    )

    print()
    print("STRATEGY PERFORMANCE")
    print("-" * 72)
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
    print("-" * 72)
    print(
        f"Annualized volatility:{pct(risk['annualized_volatility'])}"
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
    print("TRADING STATISTICS")
    print("-" * 72)
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
        f"Winning days:        "
        f"{trades['winning_days']:,}"
    )
    print(
        f"Losing days:         "
        f"{trades['losing_days']:,}"
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
    print("BUY & HOLD BENCHMARK")
    print("-" * 72)
    print(
        f"Benchmark:           "
        f"{benchmark['name']}"
    )
    print(
        f"Final value:         "
        f"{money(benchmark['final_value'])}"
    )
    print(
        f"Total return:        "
        f"{pct(benchmark['total_return'])}"
    )
    print(
        f"Annualized return:   "
        f"{pct(benchmark['annualized_return'])}"
    )
    print(
        f"Volatility:          "
        f"{pct(benchmark['annualized_volatility'])}"
    )
    print(
        f"Maximum drawdown:    "
        f"{pct(benchmark['max_drawdown'])}"
    )
    print(
        f"Strategy outperformance:"
        f" {pct(benchmark['outperformance'])}"
    )

    print()
    print("=" * 72)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    """Run the local P29.3 backtest verification."""

    try:
        validate_configuration()

        print("=" * 72)
        print("P29.3 — LOADING HISTORICAL S&P 500 DATA")
        print("=" * 72)

        market = load_all_market_data()

        print()
        print(
            f"Market rows loaded: {len(market):,}"
        )
        print(
            "Market period: "
            f"{market['Date'].min().strftime('%Y-%m-%d')}"
            " → "
            f"{market['Date'].max().strftime('%Y-%m-%d')}"
        )

        print()
        print(
            "Generating historical probabilities "
            "with the frozen Random Forest..."
        )

        backtest_input = prepare_backtest_input(
            market
        )

        print(
            "Historical model rows: "
            f"{len(backtest_input):,}"
        )

        if len(backtest_input) < 253:
            raise RuntimeError(
                "Insufficient complete feature history for "
                "a production-style backtest. "
                f"Only {len(backtest_input)} rows are available."
            )

        config = BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            threshold=THRESHOLD,
            periods_per_year=252,
        )

        print()
        print(
            "Running strategy backtest..."
        )

        result = run_backtest(
            backtest_input,
            config=config,
        )

        print_report(result)

        output_dir = (
            PROJECT_ROOT
            / "logs"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_dir
            / "latest_backtest.json"
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
            )

        print()
        print(
            "Backtest JSON written to:"
        )
        print(output_file)

        return 0

    except Exception as exc:
        print()
        print(
            "P29.3 BACKTEST FAILED"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
