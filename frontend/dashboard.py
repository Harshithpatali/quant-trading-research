"""
Streamlit dashboard for the S&P 500 Quant Trading production system.

The dashboard consumes the FastAPI backend rather than loading the model
or Supabase credentials directly.

Run:
    streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_API_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

API_BASE_URL = DEFAULT_API_URL.rstrip("/")
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="S&P 500 Quant Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional dark quant theme for Plotly
pio.templates.default = "plotly_dark"


# ---------------------------------------------------------------------
# Custom CSS – institutional quant aesthetic
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Global */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* Typography */
    h1, h2, h3, h4 {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        letter-spacing: -0.02em;
        color: #f0f2f6 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1f2e 0%, #141824 100%);
        border: 1px solid #2a3142;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    [data-testid="stMetricLabel"] {
        color: #8b95a8 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        color: #f0f2f6 !important;
        font-size: 1.55rem !important;
        font-weight: 600 !important;
        font-variant-numeric: tabular-nums;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0b0e14;
        border-right: 1px solid #1f2533;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #c9d1d9 !important;
    }

    /* Dividers */
    hr {
        border-color: #1f2533 !important;
        margin: 1.2rem 0 !important;
    }

    /* Success / Warning / Error boxes */
    .stAlert {
        border-radius: 8px;
        border: none;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #141824;
        border-radius: 8px;
        color: #8b95a8 !important;
        font-size: 0.9rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #162d4a 100%);
        color: #e0e6ed;
        border: 1px solid #2a4a6f;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #254a75 0%, #1c3a5c 100%);
        border-color: #3a6a9f;
        color: #ffffff;
    }

    /* Custom signal banner */
    .signal-banner {
        padding: 1.1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid transparent;
    }
    .signal-long {
        background: linear-gradient(90deg, #0d3320 0%, #0f3d28 100%);
        color: #3dd68c;
        border-color: #1a5c3a;
    }
    .signal-cash {
        background: linear-gradient(90deg, #3d2e0a 0%, #4a380c 100%);
        color: #f0c14b;
        border-color: #6b5210;
    }
    .signal-other {
        background: linear-gradient(90deg, #3d1515 0%, #4a1a1a 100%);
        color: #f07178;
        border-color: #6b2a2a;
    }

    /* Section headers */
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7385;
        margin-bottom: 0.75rem;
    }

    /* Detail cards */
    .detail-card {
        background: #141824;
        border: 1px solid #1f2533;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        height: 100%;
    }
    .detail-card h4 {
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        color: #c9d1d9 !important;
    }
    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid #1a1f2e;
        font-size: 0.9rem;
    }
    .detail-label {
        color: #8b95a8;
    }
    .detail-value {
        color: #e0e6ed;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
    }

    /* Caption / footer */
    .footer-text {
        color: #5c6578;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------

def _get_json(
    endpoint: str,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Call a backend endpoint and return a safe result tuple."""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return False, None, (
            "Unable to connect to the FastAPI backend. "
            f"Details: {exc}"
        )

    try:
        payload = response.json()
    except ValueError:
        return False, None, (
            f"Backend returned HTTP {response.status_code} "
            "with a non-JSON response."
        )

    if not response.ok:
        message = payload.get(
            "message",
            payload.get("detail", "Backend request failed."),
        )
        if isinstance(message, dict):
            message = message.get("message", "Backend request failed.")
        return False, payload, str(message)

    return True, payload, None


@st.cache_data(ttl=60, show_spinner=False)
def get_prediction() -> dict[str, Any]:
    """Fetch and cache the current prediction for 60 seconds."""
    success, payload, error = _get_json("/api/v1/predict")
    if not success or payload is None:
        raise RuntimeError(error or "Prediction request failed.")
    return payload


@st.cache_data(ttl=30, show_spinner=False)
def get_health() -> dict[str, Any]:
    """Fetch backend health information."""
    success, payload, error = _get_json("/health")
    if not success or payload is None:
        raise RuntimeError(error or "Health request failed.")
    return payload


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("S&P 500 Quant Trading")
st.caption(
    "Production inference dashboard · Frozen research model · Real-time signal"
)
st.divider()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("### System")
    st.markdown(f"**API Endpoint**  \n`{API_BASE_URL}`")
    st.markdown("")

    if st.button("↻  Refresh Prediction", use_container_width=True):
        get_prediction.clear()
        get_health.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<p class="footer-text">Data is cached for 30–60 s.<br />'
        "No trades are executed from this interface.</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------

try:
    health = get_health()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

health_status = health.get("status", "unknown")

if health_status == "healthy":
    st.success("Production system healthy")
else:
    st.warning("Production system reports an unhealthy dependency")

health_col1, health_col2 = st.columns(2)

with health_col1:
    supabase = health.get("supabase", {})
    st.markdown('<div class="section-header">Supabase</div>', unsafe_allow_html=True)
    if supabase.get("status") == "connected":
        st.success("Connected")
        st.metric("Rows", f"{supabase.get('rows', 0):,}")
    else:
        st.error(supabase.get("error", "Supabase unavailable."))

with health_col2:
    model = health.get("model", {})
    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)
    if model.get("status") == "loaded":
        st.success("Loaded")
        st.metric("Features", model.get("feature_count", 0))
    else:
        st.error(model.get("error", "Model unavailable."))

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


# ---------------------------------------------------------------------
# Signal banner
# ---------------------------------------------------------------------

if signal == "LONG":
    banner_class = "signal-long"
elif signal == "CASH":
    banner_class = "signal-cash"
else:
    banner_class = "signal-other"

st.markdown(
    f'<div class="signal-banner {banner_class}">'
    f"CURRENT SIGNAL &nbsp;·&nbsp; {signal}"
    f"</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Main metrics
# ---------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Probability UP", f"{probability_up:.2%}")
with col2:
    st.metric("Probability DOWN", f"{probability_down:.2%}")
with col3:
    st.metric("Threshold", f"{threshold:.2%}")
with col4:
    st.metric("Confidence", f"{confidence:.2%}")
with col5:
    st.metric("Prediction Date", prediction_date)

st.divider()


# ---------------------------------------------------------------------
# Probability chart
# ---------------------------------------------------------------------

st.markdown('<div class="section-header">Model Probability</div>', unsafe_allow_html=True)

figure = go.Figure()

# Bars
figure.add_trace(
    go.Bar(
        x=["UP", "DOWN"],
        y=[probability_up, probability_down],
        text=[f"{probability_up:.2%}", f"{probability_down:.2%}"],
        textposition="auto",
        marker=dict(
            color=["#3dd68c", "#f07178"],
            line=dict(width=0),
        ),
        width=0.45,
        hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    )
)

# Threshold line
figure.add_hline(
    y=threshold,
    line_dash="dash",
    line_color="#f0c14b",
    line_width=1.5,
    annotation_text=f"Threshold = {threshold:.2%}",
    annotation_position="top right",
    annotation_font_color="#f0c14b",
    annotation_font_size=12,
)

figure.update_layout(
    height=380,
    margin=dict(l=20, r=20, t=30, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(
        title="Probability",
        range=[0, 1],
        tickformat=".0%",
        gridcolor="#1f2533",
        zerolinecolor="#1f2533",
        color="#8b95a8",
    ),
    xaxis=dict(
        title="",
        color="#8b95a8",
        tickfont=dict(size=13),
    ),
    font=dict(family="Inter, system-ui, sans-serif", color="#e0e6ed"),
    showlegend=False,
    bargap=0.35,
)

st.plotly_chart(figure, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------
# Model / data details
# ---------------------------------------------------------------------

st.markdown('<div class="section-header">Production Details</div>', unsafe_allow_html=True)

details_col1, details_col2 = st.columns(2)

model_info = prediction.get("model", {})
data_info = prediction.get("data", {})

with details_col1:
    st.markdown(
        f"""
        <div class="detail-card">
            <h4>Model</h4>
            <div class="detail-row">
                <span class="detail-label">Type</span>
                <span class="detail-value">{model_info.get('type', 'Unknown')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Feature count</span>
                <span class="detail-value">{model_info.get('feature_count', 'Unknown')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with details_col2:
    validation_html = (
        '<span style="color:#3dd68c;">Passed</span>'
        if data_info.get("validation_passed")
        else '<span style="color:#f07178;">Failed</span>'
    )
    st.markdown(
        f"""
        <div class="detail-card">
            <h4>Data</h4>
            <div class="detail-row">
                <span class="detail-label">Source</span>
                <span class="detail-value">{data_info.get('source', 'Unknown')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Rows used</span>
                <span class="detail-value">{data_info.get('rows_used', 0):,}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Date range</span>
                <span class="detail-value">{data_info.get('date_start', 'Unknown')} → {data_info.get('date_end', 'Unknown')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Validation</span>
                <span class="detail-value">{validation_html}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")  # spacing


# ---------------------------------------------------------------------
# Raw API response
# ---------------------------------------------------------------------

with st.expander("View raw API response", expanded=False):
    st.json(prediction)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

requested_at = prediction.get("requested_at")
if requested_at:
    st.caption(f"Prediction requested at: {requested_at}")

st.caption(
    "This dashboard displays model output and does not execute trades."
)