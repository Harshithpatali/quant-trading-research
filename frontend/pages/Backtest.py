"""
Advanced Quant Backtest UI

Displays the already-generated research backtest report.

Data source:
    reports/generated/sp500_strategy_backtest_report.json

This page does not retrain models, run a backtest, or call an API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "generated"
    / "sp500_strategy_backtest_report.json"
)


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Quant Backtest",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        .hero {
            padding: 1.5rem 1.75rem;
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 18px;
            margin-bottom: 1.25rem;
            background: linear-gradient(
                135deg,
                rgba(30,30,30,.95),
                rgba(55,55,55,.82)
            );
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 750;
            margin-bottom: .25rem;
        }

        .hero-subtitle {
            opacity: .75;
            font-size: .95rem;
        }

        .metric-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 14px;
            min-height: 115px;
        }

        .metric-label {
            font-size: .78rem;
            opacity: .65;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        .metric-value {
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: .35rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 1.4rem;
            margin-bottom: .7rem;
        }

        .research-note {
            padding: 1rem 1.1rem;
            border-left: 4px solid #888;
            border-radius: 8px;
            background: rgba(128,128,128,.08);
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def money(value: float) -> str:
    return f"₹{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.2%}"


def metric_card(
    label: str,
    value: str,
) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Load report
# ---------------------------------------------------------------------

if not REPORT_PATH.exists():
    st.error(
        "Backtest report not found."
    )
    st.code(
        str(REPORT_PATH)
    )
    st.stop()

try:
    with REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)
except Exception as exc:
    st.error(
        f"Could not read backtest report: {exc}"
    )
    st.stop()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

evaluation = report.get(
    "evaluation_range",
    {},
)

start_date = evaluation.get(
    "start",
    "N/A",
)

end_date = evaluation.get(
    "end",
    "N/A",
)

best_strategy = report.get(
    "best_model_strategy",
    "N/A",
)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">
            📊 S&P 500 Quant Backtest
        </div>
        <div class="hero-subtitle">
            Historical strategy research ·
            {start_date} → {end_date}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Cost assumptions
# ---------------------------------------------------------------------

costs = report.get(
    "cost_assumptions",
    {},
)

c1, c2, c3 = st.columns(3)

with c1:
    metric_card(
        "Transaction Cost",
        f"{costs.get('transaction_cost_bps', 0):.1f} bps",
    )

with c2:
    metric_card(
        "Slippage",
        f"{costs.get('slippage_bps', 0):.1f} bps",
    )

with c3:
    metric_card(
        "Total Trading Cost",
        f"{costs.get('total_cost_bps', 0):.1f} bps",
    )


# ---------------------------------------------------------------------
# Main strategy comparison
# ---------------------------------------------------------------------

strategies = report.get(
    "strategies",
    [],
)

if not strategies:
    st.warning(
        "No strategy results found in the report."
    )
    st.stop()

strategy_df = pd.DataFrame(
    strategies
)

strategy_df["CAGR"] = strategy_df["CAGR"] * 100
strategy_df["Sharpe"] = strategy_df["Sharpe"].round(3)
strategy_df["Sortino"] = strategy_df["Sortino"].round(3)
strategy_df["Max Drawdown"] = (
    strategy_df["max_drawdown"] * 100
)
strategy_df["Win Rate"] = (
    strategy_df["win_rate"] * 100
)

display_df = strategy_df[
    [
        "strategy",
        "CAGR",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
        "Win Rate",
        "trades",
        "turnover",
    ]
].copy()

display_df.columns = [
    "Strategy",
    "CAGR %",
    "Sharpe",
    "Sortino",
    "Max Drawdown %",
    "Win Rate %",
    "Trades",
    "Turnover",
]

st.markdown(
    '<div class="section-title">Strategy Comparison</div>',
    unsafe_allow_html=True,
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "CAGR %": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "Max Drawdown %": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "Win Rate %": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "Sharpe": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "Sortino": st.column_config.NumberColumn(
            format="%.3f"
        ),
    },
)


# ---------------------------------------------------------------------
# Highlight selected/best strategy
# ---------------------------------------------------------------------

rf = next(
    (
        row
        for row in strategies
        if row.get("strategy")
        == best_strategy
    ),
    None,
)

buy_hold = next(
    (
        row
        for row in strategies
        if row.get("strategy")
        == "Buy & Hold"
    ),
    None,
)

st.markdown(
    '<div class="section-title">Key Performance</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

if rf:
    with k1:
        metric_card(
            "Selected Strategy",
            "Random Forest",
        )

    with k2:
        metric_card(
            "CAGR",
            pct(rf["CAGR"]),
        )

    with k3:
        metric_card(
            "Sharpe",
            f"{rf['Sharpe']:.3f}",
        )

    with k4:
        metric_card(
            "Max Drawdown",
            pct(rf["max_drawdown"]),
        )


# ---------------------------------------------------------------------
# Strategy vs benchmark chart
# ---------------------------------------------------------------------

chart_df = strategy_df[
    strategy_df["strategy"].isin(
        [
            "Buy & Hold",
            "Random Forest Long/Cash",
            "HistGradientBoosting Long/Cash",
            "Logistic Regression Long/Cash",
            "XGBoost Long/Cash",
        ]
    )
].copy()

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=chart_df["strategy"],
        y=chart_df["CAGR"],
        name="CAGR %",
        hovertemplate=(
            "%{x}<br>"
            "CAGR: %{y:.2f}%"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="Historical CAGR Comparison",
    xaxis_title="Strategy",
    yaxis_title="CAGR (%)",
    height=430,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=100,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ---------------------------------------------------------------------
# Risk comparison
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Risk Profile</div>',
    unsafe_allow_html=True,
)

risk_df = strategy_df[
    [
        "strategy",
        "CAGR",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
    ]
].copy()

risk_df.columns = [
    "Strategy",
    "CAGR",
    "Sharpe",
    "Sortino",
    "Max Drawdown",
]

fig_risk = go.Figure()

fig_risk.add_trace(
    go.Scatter(
        x=risk_df["Max Drawdown"],
        y=risk_df["CAGR"],
        mode="markers+text",
        text=risk_df["Strategy"],
        textposition="top center",
        marker=dict(
            size=12,
        ),
        hovertemplate=(
            "%{text}<br>"
            "CAGR: %{y:.2f}%<br>"
            "Max Drawdown: %{x:.2f}%"
            "<extra></extra>"
        ),
    )
)

fig_risk.update_layout(
    title="Return vs Maximum Drawdown",
    xaxis_title="Maximum Drawdown (%)",
    yaxis_title="CAGR (%)",
    height=470,
)

st.plotly_chart(
    fig_risk,
    use_container_width=True,
)


# ---------------------------------------------------------------------
# High-confidence strategies
# ---------------------------------------------------------------------

high_confidence = report.get(
    "high_confidence_strategies",
    [],
)

if high_confidence:
    st.markdown(
        '<div class="section-title">High-Confidence Signals</div>',
        unsafe_allow_html=True,
    )

    hc_df = pd.DataFrame(
        high_confidence
    )

    hc_display = hc_df[
        [
            "strategy",
            "CAGR",
            "Sharpe",
            "Sortino",
            "max_drawdown",
            "active_days",
            "trades",
        ]
    ].copy()

    hc_display["CAGR"] *= 100
    hc_display["max_drawdown"] *= 100

    hc_display.columns = [
        "Strategy",
        "CAGR %",
        "Sharpe",
        "Sortino",
        "Max Drawdown %",
        "Active Days",
        "Trades",
    ]

    st.dataframe(
        hc_display,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------

cost_sensitivity = report.get(
    "cost_sensitivity",
    [],
)

if cost_sensitivity:
    st.markdown(
        '<div class="section-title">Transaction-Cost Sensitivity</div>',
        unsafe_allow_html=True,
    )

    cost_df = pd.DataFrame(
        cost_sensitivity
    )

    cost_df["CAGR"] *= 100
    cost_df["max_drawdown"] *= 100

    fig_cost = go.Figure()

    for model in cost_df["model"].unique():
        model_df = cost_df[
            cost_df["model"] == model
        ].sort_values("cost_bps")

        fig_cost.add_trace(
            go.Scatter(
                x=model_df["cost_bps"],
                y=model_df["CAGR"],
                mode="lines+markers",
                name=model,
                hovertemplate=(
                    "%{fullData.name}<br>"
                    "Cost: %{x} bps<br>"
                    "CAGR: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    fig_cost.update_layout(
        title="CAGR Sensitivity to Trading Costs",
        xaxis_title="Trading Cost (bps)",
        yaxis_title="CAGR (%)",
        height=450,
    )

    st.plotly_chart(
        fig_cost,
        use_container_width=True,
    )

    cost_display = cost_df[
        [
            "model",
            "cost_bps",
            "CAGR",
            "Sharpe",
            "max_drawdown",
            "trades",
        ]
    ].copy()

    cost_display.columns = [
        "Model",
        "Cost (bps)",
        "CAGR %",
        "Sharpe",
        "Max Drawdown %",
        "Trades",
    ]

    st.dataframe(
        cost_display,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------
# Research disclaimer
# ---------------------------------------------------------------------

methodological_note = report.get(
    "methodological_note",
    (
        "Historical research results are not guarantees "
        "of future performance."
    ),
)

st.markdown(
    f"""
    <div class="research-note">
        <strong>Research note</strong><br>
        {methodological_note}
    </div>
    """,
    unsafe_allow_html=True,
)
