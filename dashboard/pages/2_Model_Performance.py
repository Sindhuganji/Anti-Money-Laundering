"""Interactive held-out performance comparison and diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_PATHS, MODEL_PATHS, PREDICTION_PATHS
from dashboard.components.ui import (
    configure_page,
    format_metric,
    page_header,
    section_heading,
    style_figure,
)

configure_page("Model performance")
page_header(
    "Model performance",
    "Compare saved models on the same held-out split and inspect ranking "
    "quality, threshold trade-offs, and error profiles.",
    "MODEL QUALITY",
)

SCORE_COLUMNS = {
    "random_forest": "probability",
    "isolation_forest": "anomaly_score",
    "autoencoder": "reconstruction_error",
}


def _read_metrics(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return None
    return payload if isinstance(payload, dict) else None


def _read_predictions(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return None


models: dict[str, dict[str, object]] = {}
for key, path in METRICS_PATHS.items():
    metrics = _read_metrics(path)
    if metrics is not None:
        models[key] = metrics

if not models:
    st.warning("No saved model metrics are available. Train a model first.")
    st.stop()

display_name = lambda value: value.replace("_", " ").title()
section_heading("Model scorecards · held-out test data")
cards = st.columns(len(models))
for card, (model_name, metrics) in zip(cards, models.items()):
    with card.container(border=True):
        st.markdown(f"#### {display_name(model_name)}")
        st.metric(
            "Average precision",
            format_metric(metrics.get("average_precision")),
        )
        st.metric("Recall", format_metric(metrics.get("recall")))
        st.metric("Precision", format_metric(metrics.get("precision")))
        st.caption(
            f"Alerts: {format_metric(metrics.get('total_alerts'))} · "
            f"Threshold: {format_metric(metrics.get('threshold'))}"
        )

section_heading("Comparative performance")
comparison = pd.DataFrame(
    [{"model": name, **metrics} for name, metrics in models.items()]
)
metric_options = [
    column
    for column in (
        "average_precision",
        "precision",
        "recall",
        "f1_score",
        "f2_score",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
    )
    if column in comparison.columns
]
selected_metrics = st.multiselect(
    "Metrics",
    metric_options,
    default=[
        metric
        for metric in ("average_precision", "precision", "recall")
        if metric in metric_options
    ],
    format_func=lambda value: value.replace("_", " ").title(),
)
if selected_metrics:
    metric_frame = comparison.melt(
        id_vars="model",
        value_vars=selected_metrics,
        var_name="Metric",
        value_name="Score",
    )
    figure = px.bar(
        metric_frame,
        x="model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Saved held-out test metrics",
        labels={"model": "Model"},
    )
    st.plotly_chart(
        style_figure(figure, height=430),
        use_container_width=True,
    )

display_columns = [
    column
    for column in (
        "model",
        "threshold",
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
    if column in comparison.columns
]
st.dataframe(
    comparison[display_columns],
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "These are final held-out test metrics. Use them for reporting, not for "
    "selecting model architecture or thresholds."
)

section_heading("Confusion matrix and ranking curves")
available_names = list(models)
selected_model = st.selectbox(
    "Inspect model",
    available_names,
    format_func=display_name,
)
metrics = models[selected_model]
matrix = [
    [metrics.get("true_negatives", 0), metrics.get("false_positives", 0)],
    [metrics.get("false_negatives", 0), metrics.get("true_positives", 0)],
]
figure = go.Figure(
    data=go.Heatmap(
        z=matrix,
        x=["Predicted not flagged", "Predicted flagged"],
        y=["Actually not labeled", "Actually labeled"],
        colorscale=[[0, "#10243a"], [1, "#167f82"]],
        text=[[f"{int(value):,}" for value in row] for row in matrix],
        texttemplate="%{text}",
        showscale=False,
        hovertemplate="%{y}<br>%{x}: %{z:,}<extra></extra>",
    )
)
figure.update_layout(title="Saved test confusion matrix")
left, right = st.columns(2)
with left:
    st.plotly_chart(
        style_figure(figure, height=360),
        use_container_width=True,
    )

prediction_frame = _read_predictions(PREDICTION_PATHS[selected_model])
with right:
    if (
        prediction_frame is not None
        and {"actual", SCORE_COLUMNS[selected_model]}.issubset(
            prediction_frame.columns
        )
        and prediction_frame["actual"].nunique() == 2
    ):
        labels = prediction_frame["actual"].to_numpy(dtype=int)
        scores = prediction_frame[SCORE_COLUMNS[selected_model]].to_numpy(
            dtype=float
        )
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=false_positive_rate,
                y=true_positive_rate,
                name="ROC",
                mode="lines",
                line={"color": "#42d6c5", "width": 2},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                name="Precision–Recall",
                mode="lines",
                line={"color": "#6da9ff", "width": 2},
                xaxis="x2",
                yaxis="y2",
            )
        )
        figure.update_layout(
            title="Score ranking diagnostics · test only",
            grid={"rows": 1, "columns": 2, "pattern": "independent"},
            xaxis={"title": "False-positive rate"},
            yaxis={"title": "True-positive rate"},
            xaxis2={"title": "Recall"},
            yaxis2={"title": "Precision", "anchor": "x2"},
            showlegend=False,
        )
        st.plotly_chart(
            style_figure(figure, height=360),
            use_container_width=True,
        )
    else:
        st.info(
            "Prediction scores for both held-out classes are needed to draw "
            "ranking curves."
        )

if selected_model == "autoencoder":
    section_heading("Autoencoder calibration and score profile")
    threshold_path = METRICS_PATHS["autoencoder"].with_name(
        "autoencoder_thresholds.csv"
    )
    if threshold_path.is_file():
        try:
            thresholds = pd.read_csv(threshold_path)
        except (OSError, pd.errors.ParserError, ValueError) as error:
            st.warning(f"Could not read threshold comparison: {error}")
        else:
            if not thresholds.empty:
                plot_frame = thresholds.copy()
                plot_frame["normal_validation_false_positive_target"] = (
                    1 - plot_frame["normal_validation_quantile"]
                )
                figure = px.line(
                    plot_frame,
                    x="false_positive_rate",
                    y="recall",
                    color="normal_validation_quantile",
                    markers=True,
                    hover_data=["threshold", "alerts", "precision"],
                    title="Validation trade-off · normal-only threshold calibration",
                    labels={
                        "false_positive_rate": "Validation false-positive rate",
                        "recall": "Validation recall",
                        "normal_validation_quantile": "Normal quantile",
                    },
                )
                st.plotly_chart(
                    style_figure(figure, height=380),
                    use_container_width=True,
                )
                st.dataframe(
                    thresholds,
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "These comparisons use validation scores and labels. "
                    "The saved threshold is not selected using the test set."
                )
    distribution = metrics.get("test_score_distribution", {})
    if isinstance(distribution, dict) and distribution:
        rows = []
        for class_name, values in distribution.items():
            if isinstance(values, dict):
                rows.append({"class": class_name, **values})
        if rows:
            distribution_frame = pd.DataFrame(rows)
            percentile_columns = [
                column
                for column in (
                    "median",
                    "p90",
                    "p95",
                    "p99",
                    "p99_5",
                    "maximum",
                )
                if column in distribution_frame.columns
            ]
            figure = px.line(
                distribution_frame.melt(
                    id_vars="class",
                    value_vars=percentile_columns,
                    var_name="Statistic",
                    value_name="Reconstruction error",
                ),
                x="Statistic",
                y="Reconstruction error",
                color="class",
                markers=True,
                title="Autoencoder test score distribution summary",
            )
            st.plotly_chart(
                style_figure(figure, height=360),
                use_container_width=True,
            )
            st.dataframe(
                distribution_frame,
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Descriptive held-out distributions only. The test data has "
                "not been used to adjust the Autoencoder."
            )

if selected_model == "random_forest":
    section_heading("Random Forest feature importance")
    model_path = MODEL_PATHS["random_forest"]
    if model_path.is_file():
        try:
            pipeline = joblib.load(model_path)
            classifier = pipeline.named_steps["classifier"]
            preprocessor = pipeline.named_steps["preprocessor"]
            names = preprocessor.get_feature_names_out()
            importance = np.asarray(classifier.feature_importances_)
            if len(names) == len(importance):
                importance_frame = pd.DataFrame(
                    {"feature": names, "importance": importance}
                ).nlargest(15, "importance")
                figure = px.bar(
                    importance_frame.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 15 Random Forest impurity importances",
                )
                st.plotly_chart(
                    style_figure(figure, height=480),
                    use_container_width=True,
                )
                st.caption(
                    "Impurity importance is descriptive and is not a causal "
                    "explanation of an individual alert."
                )
        except (OSError, ValueError, KeyError, AttributeError) as error:
            st.info(f"Feature importance is unavailable: {error}")
    else:
        st.info("The Random Forest model artifact is not available.")
