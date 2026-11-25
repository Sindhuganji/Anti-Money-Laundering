"""Interactive monitoring and model metric visualizations."""

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

from src.config import METRICS_PATHS, PREDICTION_PATHS, REPORTS_DIR
from dashboard.components.ui import (
    configure_page,
    page_header,
    section_heading,
    style_figure,
)

configure_page("Metric charts")
page_header(
    "Metric charts",
    "Interactive views of transaction activity, source labels, model scores, "
    "and saved held-out evaluation metrics.",
    "ANALYTICS",
)

summary_path = REPORTS_DIR / "eda_summary.json"
summary: dict[str, object] = {}
if summary_path.is_file():
    try:
        loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded_summary, dict):
            summary = loaded_summary
    except (OSError, json.JSONDecodeError) as error:
        st.warning(f"Could not read dataset exploration summary: {error}")
else:
    st.info("Run `python main.py explore` to generate transaction summaries.")

tab_activity, tab_risk, tab_performance = st.tabs(
    ["Transaction activity", "Risk distribution", "Model metrics"]
)

with tab_activity:
    section_heading("Time-based activity")
    monthly_values = summary.get("volume_by_month", {})
    laundering_values = summary.get("laundering_volume_by_month", {})
    if isinstance(monthly_values, dict) and monthly_values:
        monthly = pd.DataFrame(
            {
                "Month": list(monthly_values.keys()),
                "Transactions": list(monthly_values.values()),
            }
        )
        monthly["Labeled suspicious"] = monthly["Month"].map(
            laundering_values if isinstance(laundering_values, dict) else {}
        ).fillna(0)
        monthly["Month"] = pd.to_datetime(monthly["Month"], errors="coerce")
        monthly = monthly.dropna(subset=["Month"]).sort_values("Month")
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Transactions"],
                mode="lines",
                name="All transactions",
                line={"color": "#6da9ff"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Labeled suspicious"],
                mode="lines",
                name="Source-labeled suspicious",
                yaxis="y2",
                line={"color": "#f27e8a"},
            )
        )
        figure.update_layout(
            title="Monthly transaction volume",
            xaxis_title="Month",
            yaxis={"title": "Transactions"},
            yaxis2={
                "title": "Source-labeled suspicious",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
        )
        st.plotly_chart(
            style_figure(figure, height=400),
            use_container_width=True,
        )
    else:
        st.info("Monthly volume is not available in the exploration report.")

    section_heading("Transaction mix")
    left, right = st.columns(2)
    for column, field, title in (
        (left, "payment_format_counts", "Payment format"),
        (right, "receiving_currency_counts", "Receiving currency"),
    ):
        values = summary.get(field, {})
        with column:
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
                    title=f"{title} distribution · full dataset",
                )
                st.plotly_chart(
                    style_figure(figure, height=350),
                    use_container_width=True,
                )
            else:
                st.info(f"{title} counts are unavailable.")

    distribution = summary.get("amount_received_log_distribution", {})
    if isinstance(distribution, dict) and distribution:
        section_heading("Transaction amount distribution")
        figure = go.Figure()
        for label, values in distribution.items():
            if not isinstance(values, dict):
                continue
            left_edges = values.get("bin_left", [])
            right_edges = values.get("bin_right", [])
            shares = values.get("share", [])
            if not (len(left_edges) == len(right_edges) == len(shares)):
                continue
            centers = [
                (float(low) + float(high)) / 2
                for low, high in zip(left_edges, right_edges)
            ]
            figure.add_trace(
                go.Scatter(
                    x=centers,
                    y=shares,
                    mode="lines",
                    name=label.title(),
                )
            )
        figure.update_layout(
            title="Log amount received · within-class share",
            xaxis_title="log(1 + Amount Received)",
            yaxis_title="Share within sampled class",
        )
        st.plotly_chart(
            style_figure(figure, height=380),
            use_container_width=True,
        )
        st.caption(
            "Amount distributions use all labeled positive rows and a bounded "
            "sample of negatives, not the population class balance."
        )

with tab_risk:
    section_heading("Model alert distribution")
    latest = st.session_state.get("latest_prediction_report")
    if isinstance(latest, pd.DataFrame):
        prediction_frame = latest.copy()
        source_name = "Current session upload"
    else:
        model_select = st.selectbox(
            "Benchmark model",
            list(PREDICTION_PATHS),
            format_func=lambda value: value.replace("_", " ").title(),
        )
        benchmark_path = PREDICTION_PATHS[model_select]
        source_name = f"{model_select.replace('_', ' ').title()} held-out benchmark"
        if not benchmark_path.is_file():
            prediction_frame = None
        else:
            try:
                prediction_frame = pd.read_csv(benchmark_path)
            except (OSError, pd.errors.ParserError, ValueError) as error:
                st.error(f"Could not read benchmark predictions: {error}")
                prediction_frame = None

    if isinstance(prediction_frame, pd.DataFrame):
        column = next(
            (
                key
                for key in ("predicted_class", "prediction")
                if key in prediction_frame.columns
            ),
            None,
        )
        if column is not None:
            counts = prediction_frame[column].value_counts().reindex(
                [0, 1],
                fill_value=0,
            )
            figure = px.pie(
                pd.DataFrame(
                    {
                        "Decision": ["Not flagged", "Flagged for review"],
                        "Transactions": [int(counts.loc[0]), int(counts.loc[1])],
                    }
                ),
                names="Decision",
                values="Transactions",
                hole=0.65,
                color="Decision",
                color_discrete_map={
                    "Not flagged": "#6da9ff",
                    "Flagged for review": "#f27e8a",
                },
                title="Model decision distribution",
            )
            figure.update_traces(textinfo="label+percent")
            st.plotly_chart(
                style_figure(figure, height=400),
                use_container_width=True,
            )
            st.caption(
                f"Source: **{source_name}**. Benchmark records are held-out "
                "evaluation examples, not current live transactions."
            )
            if "actual" in prediction_frame.columns:
                actual_counts = prediction_frame["actual"].value_counts()
                st.metric(
                    "Source-labeled positive examples",
                    f"{int(actual_counts.get(1, 0)):,}",
                )
        else:
            st.info("Prediction report has no predicted-class column.")
    else:
        st.info("No prediction reports are available.")

with tab_performance:
    section_heading("Saved model evaluation metrics")
    records: list[dict[str, object]] = []
    for name, path in METRICS_PATHS.items():
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            st.warning(f"Could not read {path.name}: {error}")
            continue
        if isinstance(payload, dict):
            records.append({"model": name, **payload})
    if records:
        frame = pd.DataFrame(records)
        metric_options = [
            item
            for item in (
                "average_precision",
                "precision",
                "recall",
                "f1_score",
                "f2_score",
                "roc_auc",
                "false_positive_rate",
                "total_alerts",
            )
            if item in frame.columns
        ]
        selected_metric = st.selectbox(
            "Evaluation metric",
            metric_options,
            format_func=lambda value: value.replace("_", " ").title(),
        )
        figure = px.bar(
            frame,
            x="model",
            y=selected_metric,
            color="model",
            title=f"{selected_metric.replace('_', ' ').title()} · held-out test",
            labels={"model": "Model"},
        )
        st.plotly_chart(
            style_figure(figure, height=380),
            use_container_width=True,
        )
        columns = [
            column
            for column in (
                "model",
                "precision",
                "recall",
                "f1_score",
                "f2_score",
                "roc_auc",
                "average_precision",
                "false_positive_rate",
                "false_negative_rate",
                "total_alerts",
            )
            if column in frame.columns
        ]
        st.dataframe(
            frame[columns],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No model metrics are available.")
