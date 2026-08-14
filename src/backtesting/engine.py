"""
Historical backtesting engine for the S&P 500 Quant Trading system.

P29 responsibilities:
    - Convert historical model predictions into positions.
    - Apply next-session execution to avoid look-ahead bias.
    - Apply transaction costs and slippage.
    - Build an equity curve.
    - Calculate return and risk statistics.
    - Calculate a buy-and-hold benchmark.
    - Produce JSON-serializable backtest results.

This module does NOT train a model.

The production Random Forest remains frozen and is supplied by the
existing prediction/model layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_TRANSACTION_COST = 0.0005
DEFAULT_SLIPPAGE = 0.0005


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a historical strategy backtest."""

    initial_capital: float = DEFAULT_INITIAL_CAPITAL

    transaction_cost: float = DEFAULT_TRANSACTION_COST

    slippage: float = DEFAULT_SLIPPAGE

    threshold: float = 0.50

    periods_per_year: int = 252


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_config(config: BacktestConfig) -> None:
    """Validate backtest configuration."""

    if config.initial_capital <= 0:
        raise ValueError(
            "initial_capital must be greater than zero."
        )

    if not 0 <= config.transaction_cost < 1:
        raise ValueError(
            "transaction_cost must be between 0 and 1."
        )

    if not 0 <= config.slippage < 1:
        raise ValueError(
            "slippage must be between 0 and 1."
        )

    if not 0 < config.threshold < 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if config.periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero."
        )


def _validate_input_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize the historical backtest input.

    Required columns:
        Date
        Close
        probability_up
    """

    required_columns = {
        "Date",
        "Close",
        "probability_up",
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Backtest input is missing required columns: "
            f"{missing}"
        )

    result = df.copy()

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    result["Close"] = pd.to_numeric(
        result["Close"],
        errors="coerce",
    )

    result["probability_up"] = pd.to_numeric(
        result["probability_up"],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "Date",
            "Close",
            "probability_up",
        ]
    )

    if result.empty:
        raise ValueError(
            "Backtest input contains no valid observations."
        )

    result = (
        result
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if (result["Close"] <= 0).any():
        raise ValueError(
            "Close prices must be greater than zero."
        )

    if (
        (result["probability_up"] < 0)
        | (result["probability_up"] > 1)
    ).any():
        raise ValueError(
            "probability_up must be between 0 and 1."
        )

    return result


# ---------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------

def _build_positions(
    probability_up: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Convert model probabilities into LONG/CASH positions.

    The position generated on day t is executed on day t+1.

    This prevents the model from using the same day's closing price
    to generate a signal and simultaneously receive that closing
    price as the strategy execution price.
    """

    return (
        probability_up
        >= threshold
    ).astype(float)


# ---------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------

def _simulate_strategy(
    df: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Simulate the strategy portfolio.

    Signals are shifted by one trading session before returns are
    applied. Position changes incur transaction costs and slippage.
    """

    result = df.copy()

    result["market_return"] = (
        result["Close"]
        .pct_change()
        .fillna(0.0)
    )

    result["signal_position"] = _build_positions(
        result["probability_up"],
        config.threshold,
    )

    # Execute tomorrow using today's signal.
    result["position"] = (
        result["signal_position"]
        .shift(1)
        .fillna(0.0)
    )

    result["position_change"] = (
        result["position"]
        .diff()
        .abs()
        .fillna(
            result["position"].abs()
        )
    )

    # Trading friction is charged whenever exposure changes.
    friction = (
        config.transaction_cost
        + config.slippage
    )

    result["strategy_return_before_cost"] = (
        result["position"]
        * result["market_return"]
    )

    result["trading_cost"] = (
        result["position_change"]
        * friction
    )

    result["strategy_return"] = (
        result["strategy_return_before_cost"]
        - result["trading_cost"]
    )

    result["strategy_growth"] = (
        1.0
        + result["strategy_return"]
    )

    result["benchmark_growth"] = (
        1.0
        + result["market_return"]
    )

    result["strategy_equity"] = (
        config.initial_capital
        * result["strategy_growth"].cumprod()
    )

    result["benchmark_equity"] = (
        config.initial_capital
        * result["benchmark_growth"].cumprod()
    )

    return result


# ---------------------------------------------------------------------
# Risk calculations
# ---------------------------------------------------------------------

def _maximum_drawdown(
    returns: pd.Series,
) -> float:
    """Calculate maximum drawdown from a return series."""

    wealth = (
        1.0
        + returns
    ).cumprod()

    running_peak = wealth.cummax()

    drawdown = (
        wealth / running_peak
    ) - 1.0

    return float(
        drawdown.min()
    )


def _annualized_return(
    total_return: float,
    number_of_periods: int,
    periods_per_year: int,
) -> float:
    """Calculate CAGR."""

    if number_of_periods <= 0:
        return 0.0

    years = (
        number_of_periods
        / periods_per_year
    )

    if years <= 0:
        return 0.0

    if total_return <= -1:
        return -1.0

    return float(
        (1.0 + total_return)
        ** (1.0 / years)
        - 1.0
    )


def _sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int,
) -> float:
    """Calculate annualized Sharpe ratio."""

    volatility = returns.std(
        ddof=1
    )

    if volatility == 0 or np.isnan(volatility):
        return 0.0

    return float(
        (
            returns.mean()
            / volatility
        )
        * np.sqrt(periods_per_year)
    )


def _sortino_ratio(
    returns: pd.Series,
    periods_per_year: int,
) -> float:
    """Calculate annualized Sortino ratio."""

    downside = returns[
        returns < 0
    ]

    if downside.empty:
        return 0.0

    downside_deviation = downside.std(
        ddof=1
    )

    if (
        downside_deviation == 0
        or np.isnan(downside_deviation)
    ):
        return 0.0

    return float(
        (
            returns.mean()
            / downside_deviation
        )
        * np.sqrt(periods_per_year)
    )


def _value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical Value at Risk."""

    return float(
        returns.quantile(
            1.0 - confidence
        )
    )


def _conditional_value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical Conditional VaR / Expected Shortfall."""

    var = _value_at_risk(
        returns,
        confidence,
    )

    tail = returns[
        returns <= var
    ]

    if tail.empty:
        return var

    return float(
        tail.mean()
    )


# ---------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------

def _trade_statistics(
    backtest: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate trade-level statistics."""

    entries = (
        backtest["position_change"]
        > 0
    )

    number_of_position_changes = int(
        entries.sum()
    )

    # A completed trade is an entry followed by an exit.
    completed_round_trips = int(
        (
            (
                backtest["position"].shift(1)
                > backtest["position"]
            )
            & (
                backtest["position"].shift(1)
                > 0
            )
        ).sum()
    )

    winning_days = int(
        (
            backtest["strategy_return"]
            > 0
        ).sum()
    )

    losing_days = int(
        (
            backtest["strategy_return"]
            < 0
        ).sum()
    )

    gross_profit = float(
        backtest.loc[
            backtest["strategy_return"] > 0,
            "strategy_return",
        ].sum()
    )

    gross_loss = float(
        -backtest.loc[
            backtest["strategy_return"] < 0,
            "strategy_return",
        ].sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = 0.0

    active_days = int(
        (
            backtest["position"]
            > 0
        ).sum()
    )

    win_rate = (
        winning_days
        / (
            winning_days
            + losing_days
        )
        if (
            winning_days
            + losing_days
        ) > 0
        else 0.0
    )

    return {
        "position_changes": (
            number_of_position_changes
        ),
        "completed_round_trips": (
            completed_round_trips
        ),
        "active_days": active_days,
        "winning_days": winning_days,
        "losing_days": losing_days,
        "win_rate": float(win_rate),
        "profit_factor": float(
            profit_factor
        ),
    }


# ---------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    """
    Run a complete historical strategy backtest.

    Parameters
    ----------
    df:
        DataFrame containing Date, Close and probability_up.

    config:
        Backtest configuration.

    Returns
    -------
    dict
        JSON-serializable backtest result containing:
            summary
            risk
            trades
            benchmark
            equity_curve
            drawdown_curve
    """

    if config is None:
        config = BacktestConfig()

    _validate_config(config)

    data = _validate_input_frame(df)

    backtest = _simulate_strategy(
        data,
        config,
    )

    strategy_returns = (
        backtest["strategy_return"]
    )

    benchmark_returns = (
        backtest["market_return"]
    )

    strategy_total_return = float(
        (
            backtest["strategy_equity"].iloc[-1]
            / config.initial_capital
        )
        - 1.0
    )

    benchmark_total_return = float(
        (
            backtest["benchmark_equity"].iloc[-1]
            / config.initial_capital
        )
        - 1.0
    )

    strategy_cagr = _annualized_return(
        strategy_total_return,
        len(backtest),
        config.periods_per_year,
    )

    benchmark_cagr = _annualized_return(
        benchmark_total_return,
        len(backtest),
        config.periods_per_year,
    )

    strategy_volatility = float(
        strategy_returns.std(
            ddof=1
        )
        * np.sqrt(
            config.periods_per_year
        )
    )

    benchmark_volatility = float(
        benchmark_returns.std(
            ddof=1
        )
        * np.sqrt(
            config.periods_per_year
        )
    )

    strategy_max_drawdown = _maximum_drawdown(
        strategy_returns
    )

    benchmark_max_drawdown = _maximum_drawdown(
        benchmark_returns
    )

    strategy_sharpe = _sharpe_ratio(
        strategy_returns,
        config.periods_per_year,
    )

    strategy_sortino = _sortino_ratio(
        strategy_returns,
        config.periods_per_year,
    )

    strategy_var_95 = _value_at_risk(
        strategy_returns
    )

    strategy_cvar_95 = (
        _conditional_value_at_risk(
            strategy_returns
        )
    )

    trades = _trade_statistics(
        backtest
    )

    equity_curve = [
        {
            "date": row.Date.strftime(
                "%Y-%m-%d"
            ),
            "strategy": round(
                float(row.strategy_equity),
                2,
            ),
            "benchmark": round(
                float(row.benchmark_equity),
                2,
            ),
        }
        for row in backtest.itertuples()
    ]

    strategy_equity = (
        backtest["strategy_equity"]
    )

    strategy_peak = (
        strategy_equity.cummax()
    )

    drawdown = (
        strategy_equity
        / strategy_peak
    ) - 1.0

    drawdown_curve = [
        {
            "date": row.Date.strftime(
                "%Y-%m-%d"
            ),
            "drawdown": round(
                float(row.drawdown),
                6,
            ),
        }
        for row in (
            pd.DataFrame(
                {
                    "Date": backtest["Date"],
                    "drawdown": drawdown,
                }
            ).itertuples()
        )
    ]

    final_value = float(
        backtest["strategy_equity"].iloc[-1]
    )

    benchmark_final_value = float(
        backtest["benchmark_equity"].iloc[-1]
    )

    return {
        "summary": {
            "start_date": data["Date"].iloc[0].strftime(
                "%Y-%m-%d"
            ),
            "end_date": data["Date"].iloc[-1].strftime(
                "%Y-%m-%d"
            ),
            "trading_days": int(
                len(data)
            ),
            "initial_capital": float(
                config.initial_capital
            ),
            "final_value": final_value,
            "profit_loss": float(
                final_value
                - config.initial_capital
            ),
            "total_return": strategy_total_return,
            "annualized_return": strategy_cagr,
        },
        "risk": {
            "annualized_volatility": strategy_volatility,
            "max_drawdown": strategy_max_drawdown,
            "sharpe_ratio": strategy_sharpe,
            "sortino_ratio": strategy_sortino,
            "var_95": strategy_var_95,
            "cvar_95": strategy_cvar_95,
        },
        "trades": trades,
        "benchmark": {
            "name": "S&P 500 Buy & Hold",
            "final_value": benchmark_final_value,
            "total_return": benchmark_total_return,
            "annualized_return": benchmark_cagr,
            "annualized_volatility": benchmark_volatility,
            "max_drawdown": benchmark_max_drawdown,
            "outperformance": (
                strategy_total_return
                - benchmark_total_return
            ),
        },
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "configuration": {
            "threshold": float(
                config.threshold
            ),
            "transaction_cost": float(
                config.transaction_cost
            ),
            "slippage": float(
                config.slippage
            ),
            "periods_per_year": int(
                config.periods_per_year
            ),
        },
    }


__all__ = [
    "BacktestConfig",
    "run_backtest",
]