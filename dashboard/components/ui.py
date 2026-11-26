"""Shared visual theme and navigation for the AML dashboard."""

from __future__ import annotations

from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.config import DATASET_PATH
from src.prediction import model_status

PAGE_GROUPS = (
    (
        "WORKSPACE",
        (
            ("Executive overview", "app.py", ":material/dashboard:"),
            (
                "Single transaction",
                "pages/1_Single_Transaction_Prediction.py",
                ":material/receipt_long:",
            ),
        ),
    ),
    (
        "MONITORING",
        (
            (
                "Transaction analysis",
                "pages/1_Transaction_Analysis.py",
                ":material/search:",
            ),
            (
                "Alert investigation",
                "pages/5_Alert_Investigation.py",
                ":material/notification_important:",
            ),
        ),
    ),
    (
        "MODELS",
        (
            (
                "Performance",
                "pages/2_Model_Performance.py",
                ":material/monitoring:",
            ),
            (
                "Metric charts",
                "pages/3_Metric_Charts.py",
                ":material/insert_chart:",
            ),
            (
                "Model information",
                "pages/6_Model_Information.py",
                ":material/account_tree:",
            ),
            (
                "Train / retrain model",
                "pages/7_Model_Training.py",
                ":material/model_training:",
            ),
        ),
    ),
    (
        "DATA",
        (
            (
                "Data exploration",
                "pages/4_Data_Exploration.py",
                ":material/database:",
            ),
        ),
    ),
)

COLORWAY = ["#42D6C5", "#6DA9FF", "#F5B85B", "#F27E8A", "#A78BFA"]

THEME_CSS = """
<style>
:root {
  color-scheme: dark;
  --aml-bg: #07111f;
  --aml-panel: #0d1a2a;
  --aml-panel-raised: #112238;
  --aml-border: #20344b;
  --aml-text: #e8f0f8;
  --aml-muted: #9aacc0;
  --aml-teal: #42d6c5;
  --aml-blue: #6da9ff;
  --aml-green: #45d6a0;
  --aml-amber: #f5b85b;
  --aml-red: #f27e8a;
}
.stApp {
  background: linear-gradient(180deg, #07111f 0%, #091625 100%);
  color: var(--aml-text);
}
[data-testid="stHeader"] { background: rgba(7, 17, 31, 0.88); }
[data-testid="stSidebar"] {
  background: #091525;
  border-right: 1px solid var(--aml-border);
}
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
.block-container {
  max-width: 1500px;
  padding-top: 1.7rem;
  padding-bottom: 3rem;
}
h1, h2, h3 { color: var(--aml-text); letter-spacing: -0.025em; }
h1 { font-size: 2rem; font-weight: 680; }
h2 { font-size: 1.35rem; font-weight: 620; }
h3 { font-size: 1.05rem; font-weight: 600; }
p, li, label { color: #c3d0de; }
.aml-brand {
  padding: 1rem 0.9rem 1.2rem;
  margin: 0 0 1rem;
  border: 1px solid #20344b;
  border-radius: 14px;
  background: linear-gradient(145deg, #10243a, #0c1929);
}
.aml-brand-mark {
  color: var(--aml-teal);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.16em;
}
.aml-brand-name {
  color: #f2f7fc;
  font-size: 1.05rem;
  font-weight: 680;
  margin-top: 0.35rem;
}
.aml-brand-subtitle {
  color: var(--aml-muted);
  font-size: 0.75rem;
  margin-top: 0.22rem;
}
.aml-nav-label {
  color: #7389a0;
  font-size: 0.64rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  margin: 1rem 0 0.25rem 0.45rem;
}
[data-testid="stPageLink"] { margin: 0.12rem 0; }
[data-testid="stPageLink"] a {
  border: 1px solid transparent;
  border-radius: 9px;
  color: #b8c7d7;
  min-height: 2.45rem;
  padding: 0.45rem 0.65rem;
  transition: background 120ms ease, border-color 120ms ease;
}
[data-testid="stPageLink"] a:hover {
  background: #12253a;
  border-color: #203b55;
  color: #f2f7fc;
}
[data-testid="stPageLink"] a[aria-current="page"] {
  background: #123142;
  border-color: #22566a;
  border-left: 3px solid var(--aml-teal);
  color: #f4fbff;
}
.aml-status-panel {
  border-top: 1px solid var(--aml-border);
  margin-top: 1.1rem;
  padding: 0.85rem 0.3rem 0;
}
.aml-status-title {
  color: #7389a0;
  font-size: 0.64rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  margin-bottom: 0.45rem;
}
.aml-status-line {
  align-items: center;
  color: #b8c7d7;
  display: flex;
  font-size: 0.77rem;
  justify-content: space-between;
  padding: 0.25rem 0;
}
.aml-status-value { color: #e3edf6; }
.aml-dot {
  border-radius: 50%;
  display: inline-block;
  height: 0.48rem;
  margin-right: 0.42rem;
  width: 0.48rem;
}
.aml-dot-ready { background: var(--aml-green); }
.aml-dot-warn { background: var(--aml-amber); }
.aml-page-header {
  border-bottom: 1px solid var(--aml-border);
  margin-bottom: 1.25rem;
  padding: 0.1rem 0 1.15rem;
}
.aml-eyebrow {
  color: var(--aml-teal);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.15em;
  margin-bottom: 0.4rem;
  text-transform: uppercase;
}
.aml-page-title {
  color: var(--aml-text);
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1.2;
  margin: 0;
}
.aml-page-subtitle {
  color: var(--aml-muted);
  font-size: 0.95rem;
  margin: 0.45rem 0 0;
  max-width: 920px;
}
.aml-section-heading {
  border-left: 3px solid var(--aml-teal);
  color: var(--aml-text);
  font-size: 1.05rem;
  font-weight: 650;
  margin: 1.2rem 0 0.7rem;
  padding-left: 0.65rem;
}
[data-testid="stMetric"] {
  background: linear-gradient(145deg, #102137, #0d1a2a);
  border: 1px solid var(--aml-border);
  border-radius: 13px;
  padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--aml-muted); }
[data-testid="stMetricValue"] { color: #f2f7fc; font-weight: 680; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--aml-border);
  border-radius: 11px;
  overflow: hidden;
}
[data-testid="stForm"], [data-testid="stExpander"] {
  background: rgba(13, 26, 42, 0.82);
  border: 1px solid var(--aml-border);
  border-radius: 13px;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
  background: #0c1a2a;
  border-color: #29415b;
  border-radius: 8px;
  color: var(--aml-text);
}
.stButton > button, [data-testid="stDownloadButton"] > button {
  border: 1px solid #29415b;
  border-radius: 9px;
  min-height: 2.5rem;
  transition: border-color 120ms ease, transform 120ms ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
  border-color: var(--aml-teal);
  color: #f2f7fc;
  transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
  background: #167f82;
  border-color: #2aaba7;
  color: white;
}
[data-testid="stAlert"] { border-radius: 11px; }
hr { border-color: var(--aml-border); }
</style>
"""


def configure_page(page_title: str) -> None:
    """Set page defaults, apply the shared theme, and render navigation."""
    st.set_page_config(
        page_title=f"{page_title} | AML Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    render_sidebar()


def render_sidebar() -> None:
    st.sidebar.markdown(
        """
        <div class="aml-brand">
          <div class="aml-brand-mark">FINANCIAL CRIME INTELLIGENCE</div>
          <div class="aml-brand-name">AML / INSIGHT</div>
          <div class="aml-brand-subtitle">Transaction monitoring workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for group_name, pages in PAGE_GROUPS:
        st.sidebar.markdown(
            f'<div class="aml-nav-label">{escape(group_name)}</div>',
            unsafe_allow_html=True,
        )
        for label, page_path, icon in pages:
            st.sidebar.page_link(page_path, label=label, icon=icon)

    status = model_status()
    ready_models = sum(bool(item["model"]) for item in status.values())
    dataset_ready = DATASET_PATH is not None and DATASET_PATH.is_file()
    dataset_class = "aml-dot-ready" if dataset_ready else "aml-dot-warn"
    model_class = (
        "aml-dot-ready"
        if ready_models == len(status)
        else "aml-dot-warn"
    )
    dataset_text = "Connected" if dataset_ready else "Unavailable"
    st.sidebar.markdown(
        f"""
        <div class="aml-status-panel">
          <div class="aml-status-title">SYSTEM STATUS</div>
          <div class="aml-status-line">
            <span><i class="aml-dot {dataset_class}"></i>Dataset</span>
            <span class="aml-status-value">{dataset_text}</span>
          </div>
          <div class="aml-status-line">
            <span><i class="aml-dot {model_class}"></i>Models</span>
            <span class="aml-status-value">{ready_models}/{len(status)} ready</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "Decision support only. Alerts require human review."
    )


def page_header(title: str, subtitle: str, eyebrow: str = "AML INTELLIGENCE") -> None:
    st.markdown(
        f"""
        <div class="aml-page-header">
          <div class="aml-eyebrow">{escape(eyebrow)}</div>
          <h1 class="aml-page-title">{escape(title)}</h1>
          <p class="aml-page-subtitle">{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str) -> None:
    st.markdown(
        f'<div class="aml-section-heading">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def style_figure(
    figure: go.Figure,
    height: int | None = None,
) -> go.Figure:
    """Apply consistent Plotly colors and panel styling."""
    figure.update_layout(
        template="plotly_dark",
        colorway=COLORWAY,
        paper_bgcolor="#0d1a2a",
        plot_bgcolor="#0d1a2a",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": "#dce7f1"},
        margin={"l": 20, "r": 18, "t": 48, "b": 22},
        legend={
            "bgcolor": "rgba(13, 26, 42, 0.72)",
            "bordercolor": "#20344b",
            "borderwidth": 1,
        },
        hoverlabel={
            "bgcolor": "#13263c",
            "bordercolor": "#31506d",
            "font": {"color": "#f2f7fc"},
        },
    )
    if height is not None:
        figure.update_layout(height=height)
    figure.update_xaxes(
        gridcolor="#1b2d42",
        linecolor="#2a4057",
        zerolinecolor="#2a4057",
    )
    figure.update_yaxes(
        gridcolor="#1b2d42",
        linecolor="#2a4057",
        zerolinecolor="#2a4057",
    )
    return figure


def format_metric(value: Any, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number.is_integer():
        return f"{int(number):,}{suffix}"
    return f"{number:,.3f}{suffix}"
