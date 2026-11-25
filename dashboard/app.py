"""Executive overview for the AML decision-support dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_PATH, METRICS_PATHS, PREDICTION_PATHS, REPORTS_DIR
from src.prediction import model_status
from dashboard.components.ui import (
    configure_page,
    page_header,
    section_heading,
    style_figure,
)

configure_page("Executive overview")
page_header(
    "Executive overview",
    "A measured view of dataset coverage, saved model performance, and "
    "transactions prioritized for human review.",
    "AML INTELLIGENCE",
)


@st.cache_data(show_spinner=False)
def _read_json(path: str, modified_time: float) -> dict[str, object]:
    del modified_time
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


@st.cache_data(show_spinner=False)
def _read_predictions(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


def _artifact_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return _read_json(str(path), path.stat().st_mtime)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return None


def _artifact_predictions(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return _read_predictions(str(path), path.stat().st_mtime)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return None


summary = _artifact_json(REPORTS_DIR / "eda_summary.json") or {}
if DATASET_PATH is not None and DATASET_PATH.is_file():
    dataset_label = DATASET_PATH.name
else:
    dataset_label = "Dataset unavailable"

counts = summary.get("target_counts", {})
if isinstance(counts, dict):
    legitimate = int(counts.get("0", counts.get(0, 0)))
    labeled_suspicious = int(counts.get("1", counts.get(1, 0)))
else:
    legitimate = labeled_suspicious = 0
transaction_count = int(summary.get("rows", legitimate + labeled_suspicious))

model_metrics: list[dict[str, object]] = []
for model_name, path in METRICS_PATHS.items():
    metrics = _artifact_json(path)
    if metrics is not None:
        model_metrics.append({"model": model_name, **metrics})

model_state = model_status()
ready_count = sum(bool(item["model"]) for item in model_state.values())
saved_models = len(model_state)

if not transaction_count:
    st.info("Dataset summary is unavailable. Run `python main.py explore`.")

kpi_columns = st.columns(4)
kpi_columns[0].metric("Dataset transactions", f"{transaction_count:,}")
kpi_columns[1].metric("Labeled suspicious", f"{labeled_suspicious:,}")
kpi_columns[2].metric(
    "Labeled prevalence",
    f"{labeled_suspicious / transaction_count:.3%}"
    if transaction_count
    else "—",
)
kpi_columns[3].metric("Models available", f"{ready_count} / {saved_models}")
st.caption(
    f"Configured dataset: **{dataset_label}** · "
    "Dataset labels are source annotations, not investigation outcomes."
)

if model_metrics:
    section_heading("Saved holdout model performance")
    metrics_frame = pd.DataFrame(model_metrics)
    display_columns = [
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
            "total_alerts",
        )
        if column in metrics_frame.columns
    ]
    left, right = st.columns([1.45, 1])
    with left:
        if "average_precision" in metrics_frame.columns:
            figure = px.bar(
                metrics_frame,
                x="model",
                y="average_precision",
                color="model",
                title="Average precision · held-out test split",
                labels={
                    "model": "Model",
                    "average_precision": "Average precision",
                },
            )
            st.plotly_chart(
                style_figure(figure, height=330),
                use_container_width=True,
            )
        st.dataframe(
            metrics_frame[display_columns],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        selected_metric = st.selectbox(
            "Compare metric",
            [
                column
                for column in ("recall", "precision", "f2_score", "roc_auc")
                if column in metrics_frame.columns
            ],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        figure = px.bar(
            metrics_frame,
            x="model",
            y=selected_metric,
            color="model",
            title=selected_metric.replace("_", " ").title(),
            labels={"model": "Model", selected_metric: "Score"},
        )
        st.plotly_chart(
            style_figure(figure, height=330),
            use_container_width=True,
        )
    st.caption(
        "Metrics use the saved held-out test split. Thresholds were selected "
        "using validation data. Average precision is emphasized because "
        "laundering labels are rare."
    )
else:
    st.info("Model evaluation reports are not available yet.")

volume = summary.get("volume_by_month", {})
laundering_volume = summary.get("laundering_volume_by_month", {})
if isinstance(volume, dict) and volume:
    section_heading("Transaction volume")
    monthly = pd.DataFrame(
        {
            "month": list(volume.keys()),
            "transactions": list(volume.values()),
        }
    )
    if isinstance(laundering_volume, dict):
        monthly["labeled suspicious"] = monthly["month"].map(
            laundering_volume
        ).fillna(0)
    monthly["month"] = pd.to_datetime(monthly["month"], errors="coerce")
    monthly = monthly.dropna(subset=["month"]).sort_values("month")
    left, right = st.columns(2)
    with left:
        figure = px.area(
            monthly,
            x="month",
            y="transactions",
            title="Monthly transaction volume",
            labels={"month": "Month", "transactions": "Transactions"},
        )
        st.plotly_chart(
            style_figure(figure, height=320),
            use_container_width=True,
        )
    with right:
        if "labeled suspicious" in monthly:
            figure = px.line(
                monthly,
                x="month",
                y="labeled suspicious",
                title="Monthly source-labeled suspicious transactions",
                labels={
                    "month": "Month",
                    "labeled suspicious": "Labeled transactions",
                },
            )
            st.plotly_chart(
                style_figure(figure, height=320),
                use_container_width=True,
            )
else:
    st.info(
        "Monthly volume is not in the saved exploration summary. "
        "Run `python main.py explore` to refresh it."
    )

section_heading("Alert review")
uploaded_report = st.session_state.get("latest_prediction_report")
uploaded_model = st.session_state.get("latest_prediction_model")
report_label = "Current session upload"
report: pd.DataFrame | None = None
if isinstance(uploaded_report, pd.DataFrame):
    report = uploaded_report.copy()
elif "random_forest" in PREDICTION_PATHS:
    report = _artifact_predictions(PREDICTION_PATHS["random_forest"])
    report_label = "Random Forest held-out benchmark"

if report is None or report.empty:
    st.info(
        "No prediction report is available. Upload transactions on the "
        "Transaction Analysis page or train a model."
    )
else:
    prediction_column = next(
        (
            name
            for name in ("predicted_class", "prediction")
            if name in report.columns
        ),
        None,
    )
    score_column = next(
        (
            name
            for name in (
                "laundering_probability",
                "probability",
                "anomaly_score",
                "reconstruction_error",
            )
            if name in report.columns
        ),
        None,
    )
    if prediction_column is None or score_column is None:
        st.info("The available prediction report has no supported alert score.")
    else:
        alerts = report.loc[report[prediction_column] == 1].copy()
        alerts = alerts.sort_values(score_column, ascending=False).head(10)
        name = (
            str(uploaded_model).replace("_", " ").title()
            if uploaded_model
            else "Random Forest"
        )
        first, second = st.columns(2)
        first.metric("Alerts in report", f"{int((report[prediction_column] == 1).sum()):,}")
        second.metric("Score used for ranking", score_column.replace("_", " ").title())
        st.caption(
            f"Source: **{report_label}** · Model: **{name}**. "
            "Saved prediction files are benchmark holdout records, not live "
            "or recent transactions."
        )
        st.dataframe(alerts, hide_index=True, use_container_width=True)
        if "actual" in report.columns and not isinstance(
            uploaded_report, pd.DataFrame
        ):
            true_positives = int(
                (
                    (report[prediction_column] == 1)
                    & (report["actual"] == 1)
                ).sum()
            )
            st.caption(
                f"Benchmark reference only: {true_positives:,} flagged rows "
                "carry a positive dataset label. This is not an investigative "
                "finding."
            )

status_frame = pd.DataFrame(model_state).T
section_heading("System readiness")
readiness_columns = st.columns(len(status_frame.columns))
for column, (name, values) in zip(readiness_columns, status_frame.items()):
    column.metric(name.replace("_", " ").title(), f"{int(values.sum())}/{len(values)}")
