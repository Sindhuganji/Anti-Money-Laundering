"""Explore full-dataset summaries and bounded visual samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR
from dashboard.components.ui import (
    configure_page,
    page_header,
    section_heading,
    style_figure,
)

configure_page("Data exploration")
page_header(
    "Data exploration",
    "Understand source coverage, label prevalence, transaction formats, "
    "currencies, and bounded-sample feature distributions.",
    "DATA",
)

summary_path = REPORTS_DIR / "eda_summary.json"
if not summary_path.is_file():
    st.info("Dataset summary is unavailable. Run `python main.py explore`.")
    st.stop()

try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Could not read dataset summary: {error}")
    st.stop()

columns = summary.get("columns", [])
target_counts = summary.get("target_counts", {})
if isinstance(target_counts, dict):
    legitimate = int(target_counts.get("0", target_counts.get(0, 0)))
    suspicious = int(target_counts.get("1", target_counts.get(1, 0)))
else:
    legitimate = suspicious = 0
rows = int(summary.get("rows", legitimate + suspicious))

st.caption(
    f"Source: **{Path(str(summary.get('dataset_path', 'unknown'))).name}** · "
    "Counts are full-dataset unless specifically identified as sampled."
)
metric_columns = st.columns(4)
metric_columns[0].metric("Transactions", f"{rows:,}")
metric_columns[1].metric("Source fields", f"{len(columns):,}")
metric_columns[2].metric("Labeled suspicious", f"{suspicious:,}")
metric_columns[3].metric(
    "Positive prevalence",
    f"{float(summary.get('positive_prevalence', 0)):.3%}",
)

section_heading("Label distribution")
left, right = st.columns(2)
class_frame = pd.DataFrame(
    {
        "Label": ["Legitimate (0)", "Labeled suspicious (1)"],
        "Transactions": [legitimate, suspicious],
    }
)
with left:
    figure = px.pie(
        class_frame,
        names="Label",
        values="Transactions",
        hole=0.68,
        title="Full-dataset label share",
        color="Label",
        color_discrete_map={
            "Legitimate (0)": "#6da9ff",
            "Labeled suspicious (1)": "#f27e8a",
        },
    )
    figure.update_traces(textinfo="label+percent")
    st.plotly_chart(
        style_figure(figure, height=350),
        use_container_width=True,
    )
with right:
    volume = summary.get("volume_by_month", {})
    laundering_volume = summary.get("laundering_volume_by_month", {})
    if isinstance(volume, dict) and volume:
        monthly = pd.DataFrame(
            {
                "Month": list(volume.keys()),
                "Transactions": list(volume.values()),
            }
        )
        monthly["Labeled suspicious"] = monthly["Month"].map(
            laundering_volume if isinstance(laundering_volume, dict) else {}
        ).fillna(0)
        monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
        monthly = monthly.dropna(subset=["Month"]).sort_values("Month")
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Transactions"],
                name="All transactions",
                mode="lines",
                line={"color": "#6da9ff", "width": 2},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Labeled suspicious"],
                name="Labeled suspicious",
                mode="lines",
                line={"color": "#f27e8a", "width": 2},
                yaxis="y2",
            )
        )
        figure.update_layout(
            title="Monthly volume · independent scales",
            yaxis={"title": "Transactions"},
            yaxis2={
                "title": "Labeled suspicious",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
        )
        st.plotly_chart(
            style_figure(figure, height=350),
            use_container_width=True,
        )
    else:
        st.info("Monthly counts are unavailable. Run `python main.py explore`.")

section_heading("Transaction mix")
currency_tabs = st.tabs(
    ["Payment format", "Receiving currency", "Payment currency"]
)
for tab, field in zip(
    currency_tabs,
    (
        "payment_format_counts",
        "receiving_currency_counts",
        "payment_currency_counts",
    ),
):
    with tab:
        values = summary.get(field, {})
        if isinstance(values, dict) and values:
            frame = pd.DataFrame(
                list(values.items()),
                columns=["Category", "Transactions"],
            ).sort_values("Transactions", ascending=False)
            figure = px.bar(
                frame,
                x="Category",
                y="Transactions",
                color="Transactions",
                color_continuous_scale=["#17445a", "#42d6c5"],
                title=f"{field.replace('_counts', '').replace('_', ' ').title()}",
            )
            st.plotly_chart(
                style_figure(figure, height=360),
                use_container_width=True,
            )
        else:
            st.info("This count is not present in the exploration report.")

section_heading("Feature distributions and relationships")
amount_distribution = summary.get("amount_received_log_distribution", {})
if isinstance(amount_distribution, dict) and amount_distribution:
    figure = go.Figure()
    colors = {"normal": "#6da9ff", "laundering": "#f27e8a"}
    for label, values in amount_distribution.items():
        if not isinstance(values, dict):
            continue
        left_edges = values.get("bin_left", [])
        right_edges = values.get("bin_right", [])
        shares = values.get("share", [])
        if len(left_edges) != len(right_edges) or len(left_edges) != len(shares):
            continue
        centers = [
            (float(left) + float(right)) / 2
            for left, right in zip(left_edges, right_edges)
        ]
        figure.add_trace(
            go.Scatter(
                x=centers,
                y=shares,
                name=label.title(),
                mode="lines",
                line={"color": colors.get(label, "#42d6c5"), "width": 2},
            )
        )
    figure.update_layout(
        title="Log amount received · within-class share",
        xaxis_title="log(1 + Amount Received)",
        yaxis_title="Share of sampled class",
    )
    st.plotly_chart(
        style_figure(figure, height=360),
        use_container_width=True,
    )
    st.caption(
        "This chart uses all labeled positives and a bounded random sample of "
        "negatives; it is not the full-population class distribution."
    )
else:
    st.info("Amount distribution is unavailable. Run `python main.py explore`.")

correlations = summary.get("numeric_correlations", {})
if isinstance(correlations, dict) and correlations:
    correlation_frame = pd.DataFrame(correlations, dtype=float)
    if not correlation_frame.empty:
        figure = px.imshow(
            correlation_frame,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
            title="Amount and target correlations · bounded plot sample",
        )
        st.plotly_chart(
            style_figure(figure, height=360),
            use_container_width=True,
        )

section_heading("Data quality")
missing = summary.get("missing_values_by_column", {})
if isinstance(missing, dict) and missing:
    st.dataframe(
        pd.DataFrame(
            list(missing.items()),
            columns=["Field", "Missing values"],
        ).sort_values("Missing values", ascending=False),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.success("The full-dataset report records no missing values.")

with st.expander("Dataset schema and methodology"):
    st.write("**Source fields**")
    st.code(", ".join(str(column) for column in columns))
    for note in summary.get("notes", []):
        st.markdown(f"- {note}")

st.download_button(
    "Download exploration summary",
    data=json.dumps(summary, indent=2),
    file_name="eda_summary.json",
    mime="application/json",
)
