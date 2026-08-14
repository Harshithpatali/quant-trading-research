from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

API_BASE_URL = "https://quant-trading-research.onrender.com"
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="S&P 500 Quant Terminal",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

pio.templates.default = "plotly_dark"


# ---------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------
# A trading-terminal system, not a dashboard theme: warm graphite (never
# pure black), tabular monospace for every number, amber as the single
# signature accent (the color every real market terminal reaches for),
# and muted — not neon — up/down colors so the eye isn't fatigued by
# an all-day screen.

BG_VOID = "#0a0b0d"
BG_PANEL = "#121317"
BG_PANEL_RAISED = "#191b20"
BORDER = "#262932"
BORDER_SOFT = "#1a1c22"
TEXT_PRIMARY = "#e9e6de"
TEXT_SECONDARY = "#8b8f99"
TEXT_MUTED = "#565b66"
AMBER = "#d4a72c"
AMBER_DIM = "#8a7128"
UP = "#5fb787"
UP_DIM = "#1c3226"
DOWN = "#c96a5e"
DOWN_DIM = "#33201d"

MONO = "'IBM Plex Mono', 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
SANS = "'IBM Plex Sans', 'Inter', system-ui, sans-serif"


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    .stApp {{
        background-color: {BG_VOID};
        color: {TEXT_PRIMARY};
        font-family: {SANS};
    }}

    #MainMenu, footer {{ visibility: hidden; }}

    h1, h2, h3, h4 {{
        font-family: {SANS} !important;
        letter-spacing: -0.01em;
        color: {TEXT_PRIMARY} !important;
    }}

    /* everything numeric reads as terminal data */
    [data-testid="stMetricValue"] {{
        font-family: {MONO} !important;
        color: {TEXT_PRIMARY} !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        font-variant-numeric: tabular-nums;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-family: {MONO} !important;
        font-size: 0.68rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }}
    [data-testid="stMetric"] {{
        background: {BG_PANEL};
        border: 1px solid {BORDER_SOFT};
        border-top: 2px solid {AMBER_DIM};
        border-radius: 3px;
        padding: 14px 16px 12px 16px;
    }}

    [data-testid="stSidebar"] {{
        background: {BG_VOID};
        border-right: 1px solid {BORDER_SOFT};
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT_SECONDARY} !important; }}
    [data-testid="stSidebar"] strong {{
        color: {TEXT_PRIMARY} !important;
        font-family: {MONO} !important;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
    }}

    hr {{ border-color: {BORDER_SOFT} !important; margin: 1.1rem 0 !important; }}

    .stAlert {{ border-radius: 3px; border: 1px solid {BORDER_SOFT} !important; font-family: {MONO}; font-size: 0.82rem; }}

    .streamlit-expanderHeader {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER_SOFT};
        border-radius: 3px;
        color: {TEXT_SECONDARY} !important;
        font-family: {MONO};
        font-size: 0.82rem;
    }}

    .stButton > button {{
        background: {BG_PANEL_RAISED};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 3px;
        font-family: {MONO};
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        transition: border-color 0.15s ease, color 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {AMBER_DIM};
        color: {AMBER};
    }}

    /* ticker eyebrow */
    .eyebrow {{
        font-family: {MONO};
        color: {TEXT_MUTED};
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }}

    /* signal banner */
    .signal-banner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.15rem 1.5rem;
        border-radius: 3px;
        margin-bottom: 1.4rem;
        background: {BG_PANEL};
        border: 1px solid {BORDER_SOFT};
        border-left: 3px solid var(--sig-color);
    }}
    .signal-symbol {{
        font-family: {MONO};
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: var(--sig-color);
    }}
    .signal-sub {{
        font-family: {MONO};
        font-size: 0.74rem;
        color: {TEXT_MUTED};
        text-align: right;
        line-height: 1.7;
    }}

    .panel {{
        background: {BG_PANEL};
        border: 1px solid {BORDER_SOFT};
        border-radius: 3px;
        padding: 1rem 1.2rem;
        height: 100%;
    }}
    .panel h4 {{
        margin: 0 0 0.7rem 0;
        font-family: {MONO} !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {TEXT_MUTED} !important;
        font-weight: 600 !important;
        border-bottom: 1px solid {BORDER_SOFT};
        padding-bottom: 0.6rem;
    }}
    .row {{
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid {BORDER_SOFT};
        font-family: {MONO};
        font-size: 0.82rem;
    }}
    .row:last-child {{ border-bottom: none; }}
    .row-label {{ color: {TEXT_SECONDARY}; }}
    .row-value {{ color: {TEXT_PRIMARY}; font-weight: 500; font-variant-numeric: tabular-nums; }}

    .lineage-index {{
        font-family: {MONO};
        color: {AMBER_DIM};
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }}
    .lineage-title {{
        font-family: {MONO};
        font-size: 1.0rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
        margin: 0.3rem 0 0.15rem 0;
    }}
    .lineage-sub {{
        font-family: {MONO};
        font-size: 0.74rem;
        color: {TEXT_MUTED};
    }}

    .footer-text {{
        color: {TEXT_MUTED};
        font-family: {MONO};
        font-size: 0.72rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------

def _get_json(endpoint: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Call the public FastAPI production endpoint."""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return False, None, f"Unable to connect to the FastAPI backend. Details: {exc}"

    try:
        payload = response.json()
    except ValueError:
        return False, None, f"Backend returned HTTP {response.status_code} with a non-JSON response."

    if not response.ok:
        message = payload.get("message", payload.get("detail", "Backend request failed."))
        if isinstance(message, dict):
            message = message.get("message", "Backend request failed.")
        return False, payload, str(message)

    return True, payload, None


@st.cache_data(ttl=60, show_spinner=False)
def get_prediction() -> dict[str, Any]:
    success, payload, error = _get_json("/api/v1/predict")
    if not success or payload is None:
        raise RuntimeError(error or "Prediction request failed.")
    return payload


@st.cache_data(ttl=30, show_spinner=False)
def get_health() -> dict[str, Any]:
    success, payload, error = _get_json("/health")
    if not success or payload is None:
        raise RuntimeError(error or "Health request failed.")
    return payload


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:20px; padding:6px 0 10px 0;">
        <div>
            <div class="eyebrow">QUANT RESEARCH TERMINAL — S&amp;P 500</div>
            <div style="font-family:{MONO}; color:{TEXT_PRIMARY}; font-size:1.9rem; font-weight:700; letter-spacing:-0.01em; line-height:1.1; margin-top:6px;">
                MARKET&nbsp;INTELLIGENCE
            </div>
            <div style="color:{TEXT_SECONDARY}; font-size:0.86rem; margin-top:8px; font-family:{SANS};">
                Production inference · frozen Random Forest · live Supabase data
            </div>
        </div>
        <div style="text-align:right; font-family:{MONO}; font-size:0.74rem; line-height:1.8;">
            <div style="color:{UP}; font-weight:700;">●&nbsp; LIVE BACKEND</div>
            <div style="color:{TEXT_MUTED};">{now_utc}</div>
        </div>
    </div>
    <div style="
        padding:9px 14px; border:1px solid {BORDER_SOFT}; background:{BG_PANEL};
        border-radius:3px; display:flex; justify-content:space-between;
        color:{TEXT_MUTED}; font-size:0.74rem; font-family:{MONO};
    ">
        <span>ENDPOINT</span>
        <span style="color:{TEXT_SECONDARY};">{API_BASE_URL}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="eyebrow" style="margin-bottom:10px;">Terminal Controls</div>', unsafe_allow_html=True)

    if st.button("↻  REFRESH MARKET STATE", use_container_width=True):
        get_prediction.clear()
        get_health.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**EXECUTION MODE**")
    st.caption("Inference only · no orders are executed")
    st.markdown("**DATA SOURCE**")
    st.caption("Supabase · `sp500_daily`")
    st.markdown("**MODEL**")
    st.caption("Random Forest · 26 features")
    st.markdown("---")
    st.markdown(
        '<p class="footer-text">CACHE&nbsp;·&nbsp;prediction 60s<br>CACHE&nbsp;·&nbsp;health 30s</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Backend health
# ---------------------------------------------------------------------

try:
    health = get_health()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

health_status = health.get("status", "unknown")
supabase = health.get("supabase", {})
health_model = health.get("model", {})

h1, h2, h3 = st.columns(3)

with h1:
    if health_status == "healthy":
        st.success("●  SYSTEM · HEALTHY")
    else:
        st.error("●  SYSTEM · UNHEALTHY")

with h2:
    if supabase.get("status") == "connected":
        st.success(f"●  SUPABASE · {supabase.get('rows', 0):,} ROWS")
    else:
        st.error("●  SUPABASE · UNAVAILABLE")

with h3:
    if health_model.get("status") == "loaded":
        st.success(f"●  MODEL · {health_model.get('feature_count', 0)} FEATURES")
    else:
        st.error("●  MODEL · UNAVAILABLE")

st.divider()


# ---------------------------------------------------------------------
# Current prediction
# ---------------------------------------------------------------------

try:
    prediction = get_prediction()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

signal = prediction.get("signal", "UNKNOWN")
probability_up = float(prediction.get("probability_up", 0.0))
probability_down = float(prediction.get("probability_down", 0.0))
threshold = float(prediction.get("threshold", 0.5))
confidence = float(prediction.get("confidence", 0.0))
prediction_date = prediction.get("prediction_date", "Unknown")

model_info = prediction.get("model", {})
data_info = prediction.get("data", {})

if signal == "LONG":
    sig_color, sig_icon, sig_sub = UP, "▲", "Bullish model classification"
elif signal == "CASH":
    sig_color, sig_icon, sig_sub = AMBER, "■", "No long exposure"
else:
    sig_color, sig_icon, sig_sub = DOWN, "●", "Review model output"

st.markdown(
    f"""
    <div class="signal-banner" style="--sig-color:{sig_color};">
        <div>
            <div class="eyebrow" style="margin-bottom:6px;">MODEL SIGNAL</div>
            <div class="signal-symbol">{sig_icon}&nbsp; {signal}</div>
        </div>
        <div class="signal-sub">
            <div>{sig_sub}</div>
            <div>PREDICTION DATE · {prediction_date}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Probability Up", f"{probability_up:.2%}")
with col2:
    st.metric("Probability Down", f"{probability_down:.2%}")
with col3:
    st.metric("Model Threshold", f"{threshold:.2%}")
with col4:
    st.metric("Signal Confidence", f"{confidence:.2%}")
with col5:
    st.metric("Position", "100%" if signal == "LONG" else "0%")

st.markdown("")

chart_col, intelligence_col = st.columns([1.7, 1], gap="large")

with chart_col:
    st.markdown('<div class="eyebrow" style="margin-bottom:10px;">Signal Probability Surface</div>', unsafe_allow_html=True)

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=["UP", "DOWN"],
            y=[probability_up, probability_down],
            text=[f"{probability_up:.2%}", f"{probability_down:.2%}"],
            textposition="outside",
            textfont=dict(family=MONO, size=13, color=TEXT_PRIMARY),
            marker=dict(color=[UP, DOWN], line=dict(width=0)),
            width=0.42,
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )
    figure.add_hline(
        y=threshold,
        line_dash="dot",
        line_color=AMBER,
        line_width=1.3,
        annotation_text=f"THRESHOLD {threshold:.0%}",
        annotation_position="top right",
        annotation_font_color=AMBER,
        annotation_font_size=11,
        annotation_font_family=MONO,
    )
    figure.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=30, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=BORDER_SOFT, zeroline=False, color=TEXT_SECONDARY),
        xaxis=dict(color=TEXT_SECONDARY, tickfont=dict(size=13, family=MONO)),
        font=dict(family=MONO, color=TEXT_PRIMARY),
        showlegend=False,
        bargap=0.38,
    )
    st.plotly_chart(figure, use_container_width=True)

with intelligence_col:
    st.markdown('<div class="eyebrow" style="margin-bottom:10px;">Model Intelligence</div>', unsafe_allow_html=True)

    rows_used = data_info.get("rows_used", 0)
    validation = "PASS" if data_info.get("validation_passed") else "FAIL"
    validation_color = UP if validation == "PASS" else DOWN

    st.markdown(
        f"""
        <div class="panel" style="min-height:310px;">
            <h4>Inference Context</h4>
            <div class="row"><span class="row-label">Algorithm</span><span class="row-value">{model_info.get('type', 'Unknown')}</span></div>
            <div class="row"><span class="row-label">Features</span><span class="row-value">{model_info.get('feature_count', 'Unknown')}</span></div>
            <div class="row"><span class="row-label">Rows in window</span><span class="row-value">{rows_used:,}</span></div>
            <div class="row"><span class="row-label">Threshold</span><span class="row-value">{threshold:.2%}</span></div>
            <div class="row"><span class="row-label">Data validation</span><span class="row-value" style="color:{validation_color};">{validation}</span></div>
            <div class="row"><span class="row-label">Inference date</span><span class="row-value">{prediction_date}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ---------------------------------------------------------------------
# Production data lineage
# ---------------------------------------------------------------------

st.markdown('<div class="eyebrow" style="margin-bottom:10px;">Production Data Lineage</div>', unsafe_allow_html=True)

lineage1, lineage2, lineage3, lineage4 = st.columns(4)

with lineage1:
    st.markdown(
        f"""
        <div class="panel">
            <div class="lineage-index">01 · SOURCE</div>
            <div class="lineage-title">Supabase</div>
            <div class="lineage-sub">sp500_daily</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with lineage2:
    st.markdown(
        f"""
        <div class="panel">
            <div class="lineage-index">02 · WINDOW</div>
            <div class="lineage-title">{rows_used:,} rows</div>
            <div class="lineage-sub">{data_info.get('date_start', 'Unknown')} → {data_info.get('date_end', 'Unknown')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with lineage3:
    st.markdown(
        f"""
        <div class="panel">
            <div class="lineage-index">03 · MODEL</div>
            <div class="lineage-title">Random Forest</div>
            <div class="lineage-sub">Frozen production artifact</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with lineage4:
    st.markdown(
        f"""
        <div class="panel">
            <div class="lineage-index">04 · OUTPUT</div>
            <div class="lineage-title" style="color:{sig_color};">{signal}</div>
            <div class="lineage-sub">Position = {'1.0' if signal == 'LONG' else '0.0'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

with st.expander("View raw production API response", expanded=False):
    st.json(prediction)

st.divider()

st.caption(
    f"Backend: {API_BASE_URL} · "
    f"Prediction requested: {prediction.get('requested_at', 'Unknown')} · "
    "Inference dashboard only — no trades are executed."
)