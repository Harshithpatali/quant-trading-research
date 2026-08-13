from __future__ import annotations

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
    """Call the public FastAPI production endpoint."""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )
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
            message = message.get(
                "message",
                "Backend request failed.",
            )
        return False, payload, str(message)

    return True, payload, None


@st.cache_data(ttl=60, show_spinner=False)
def get_prediction() -> dict[str, Any]:
    """Fetch and cache the current production prediction."""
    success, payload, error = _get_json("/api/v1/predict")
    if not success or payload is None:
        raise RuntimeError(
            error or "Prediction request failed."
        )
    return payload


@st.cache_data(ttl=30, show_spinner=False)
def get_health() -> dict[str, Any]:
    """Fetch backend health information."""
    success, payload, error = _get_json("/health")
    if not success or payload is None:
        raise RuntimeError(
            error or "Health request failed."
        )
    return payload


# Header
# ---------------------------------------------------------------------

st.markdown(
    """
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        gap:20px;
        padding:8px 0 4px 0;
    ">
        <div>
            <div style="
                color:#657089;
                font-size:0.72rem;
                font-weight:700;
                letter-spacing:0.14em;
                text-transform:uppercase;
                margin-bottom:6px;
            ">QUANT RESEARCH TERMINAL · S&P 500</div>
            <div style="
                color:#f5f7fb;
                font-size:2.15rem;
                font-weight:750;
                letter-spacing:-0.045em;
                line-height:1.05;
            ">Market Intelligence</div>
            <div style="
                color:#7f8ba0;
                font-size:0.92rem;
                margin-top:8px;
            ">Production inference · frozen Random Forest · live Supabase data</div>
        </div>
        <div style="
            text-align:right;
            color:#667289;
            font-size:0.75rem;
            line-height:1.6;
        ">
            <div style="color:#3dd68c;font-weight:700;">● LIVE BACKEND</div>
            <div>FASTAPI / RENDER</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="
        margin-top:18px;
        padding:9px 14px;
        border:1px solid #1f2a3a;
        background:#0d121b;
        border-radius:8px;
        display:flex;
        justify-content:space-between;
        color:#758197;
        font-size:0.76rem;
        font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    ">
        <span>BACKEND</span>
        <span style="color:#aab5c8;">{API_BASE_URL}</span>
        
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

with st.sidebar:
    st.markdown(
        """
        <div style="
            font-size:0.72rem;
            color:#667289;
            font-weight:700;
            letter-spacing:0.13em;
            text-transform:uppercase;
            margin-bottom:8px;
        ">Terminal Controls</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "↻  Refresh Market State",
        use_container_width=True,
    ):
        get_prediction.clear()
        get_health.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Execution mode**")
    st.caption("Inference only · no orders are executed")
    st.markdown("**Data source**")
    st.caption("Supabase · `sp500_daily`")
    st.markdown("**Model**")
    st.caption("Random Forest · 26 features")
    st.markdown("---")
    st.markdown(
        '<p class="footer-text">'
        "Cached inference: 60 seconds<br>"
        "Health check: 30 seconds<br>"
       
        "</p>",
        unsafe_allow_html=True,
    )


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
        st.success("● SYSTEM · HEALTHY")
    else:
        st.error("● SYSTEM · UNHEALTHY")

with h2:
    if supabase.get("status") == "connected":
        st.success(
            f"● SUPABASE · {supabase.get('rows', 0):,} ROWS"
        )
    else:
        st.error("● SUPABASE · UNAVAILABLE")

with h3:
    if health_model.get("status") == "loaded":
        st.success(
            f"● MODEL · {health_model.get('feature_count', 0)} FEATURES"
        )
    else:
        st.error("● MODEL · UNAVAILABLE")

st.divider()


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
    banner_class = "signal-long"
    signal_icon = "▲"
    signal_subtitle = "Bullish model classification"
elif signal == "CASH":
    banner_class = "signal-cash"
    signal_icon = "■"
    signal_subtitle = "No long exposure"
else:
    banner_class = "signal-other"
    signal_icon = "●"
    signal_subtitle = "Review model output"

st.markdown(
    f"""
    <div class="signal-banner {banner_class}" style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        text-align:left;
        padding:1.15rem 1.45rem;
    ">
        <div>
            <div style="
                font-size:0.70rem;
                opacity:0.72;
                letter-spacing:0.13em;
                margin-bottom:5px;
            ">MODEL SIGNAL</div>
            <div style="font-size:1.65rem;">
                {signal_icon}&nbsp; {signal}
            </div>
        </div>
        <div style="
            font-size:0.80rem;
            opacity:0.78;
            text-align:right;
        ">
            <div>{signal_subtitle}</div>
            <div style="margin-top:4px;">Prediction date · {prediction_date}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Probability UP", f"{probability_up:.2%}")
with col2:
    st.metric("Probability DOWN", f"{probability_down:.2%}")
with col3:
    st.metric("Model Threshold", f"{threshold:.2%}")
with col4:
    st.metric("Signal Confidence", f"{confidence:.2%}")
with col5:
    st.metric(
        "Position",
        "100%" if signal == "LONG" else "0%",
    )

st.markdown("")

chart_col, intelligence_col = st.columns(
    [1.7, 1],
    gap="large",
)

with chart_col:
    st.markdown(
        '<div class="section-header">Signal Probability Surface</div>',
        unsafe_allow_html=True,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=["UP", "DOWN"],
            y=[probability_up, probability_down],
            text=[
                f"{probability_up:.2%}",
                f"{probability_down:.2%}",
            ],
            textposition="auto",
            marker=dict(
                color=["#3dd68c", "#f07178"],
                line=dict(width=0),
            ),
            width=0.42,
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )

    figure.add_hline(
        y=threshold,
        line_dash="dot",
        line_color="#f0c14b",
        line_width=1.4,
        annotation_text=f"threshold {threshold:.0%}",
        annotation_position="top right",
        annotation_font_color="#f0c14b",
        annotation_font_size=11,
    )

    figure.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=28, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            title="",
            range=[0, 1],
            tickformat=".0%",
            gridcolor="#1f2533",
            zeroline=False,
            color="#7f8ba0",
        ),
        xaxis=dict(
            title="",
            color="#7f8ba0",
            tickfont=dict(size=13),
        ),
        font=dict(
            family="Inter, system-ui, sans-serif",
            color="#e0e6ed",
        ),
        showlegend=False,
        bargap=0.38,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

with intelligence_col:
    st.markdown(
        '<div class="section-header">Model Intelligence</div>',
        unsafe_allow_html=True,
    )

    rows_used = data_info.get("rows_used", 0)
    validation = (
        "PASS"
        if data_info.get("validation_passed")
        else "FAIL"
    )

    st.markdown(
        f"""
        <div class="detail-card" style="min-height:310px;">
            <h4>Inference Context</h4>
            <div class="detail-row">
                <span class="detail-label">Algorithm</span>
                <span class="detail-value">{model_info.get('type', 'Unknown')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Features</span>
                <span class="detail-value">{model_info.get('feature_count', 'Unknown')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Rows in window</span>
                <span class="detail-value">{rows_used:,}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Threshold</span>
                <span class="detail-value">{threshold:.2%}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Data validation</span>
                <span class="detail-value">{validation}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Inference date</span>
                <span class="detail-value">{prediction_date}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# Production data lineage
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-header">Production Data Lineage</div>',
    unsafe_allow_html=True,
)

lineage1, lineage2, lineage3, lineage4 = st.columns(4)

with lineage1:
    st.markdown(
        """
        <div class="detail-card">
            <div class="section-header">01 · SOURCE</div>
            <div style="font-size:1.0rem;font-weight:600;">Supabase</div>
            <div class="footer-text">sp500_daily</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with lineage2:
    st.markdown(
        f"""
        <div class="detail-card">
            <div class="section-header">02 · WINDOW</div>
            <div style="font-size:1.0rem;font-weight:600;">{rows_used:,} rows</div>
            <div class="footer-text">
                {data_info.get('date_start', 'Unknown')}
                → {data_info.get('date_end', 'Unknown')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with lineage3:
    st.markdown(
        """
        <div class="detail-card">
            <div class="section-header">03 · MODEL</div>
            <div style="font-size:1.0rem;font-weight:600;">Random Forest</div>
            <div class="footer-text">Frozen production artifact</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with lineage4:
    st.markdown(
        f"""
        <div class="detail-card">
            <div class="section-header">04 · OUTPUT</div>
            <div style="font-size:1.0rem;font-weight:600;">{signal}</div>
            <div class="footer-text">
                Position = {'1.0' if signal == 'LONG' else '0.0'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

with st.expander(
    "View raw production API response",
    expanded=False,
):
    st.json(prediction)

st.divider()

st.caption(
    f"Backend: {API_BASE_URL} · "
    f"Prediction requested: {prediction.get('requested_at', 'Unknown')} · "
    "Inference dashboard only — no trades are executed."
)
