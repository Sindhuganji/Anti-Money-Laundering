"""Model provenance, threshold strategy, and limitations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_PATH, METRICS_PATHS
from src.prediction import model_status
from dashboard.components.ui import (
    configure_page,
    format_metric,
    page_header,
    section_heading,
)

configure_page("Model information")
page_header(
    "Model information",
    "Model provenance, training configuration, threshold policy, evaluation "
    "scope, and responsible-use limitations.",
    "GOVERNANCE",
)

descriptions = {
    "random_forest": {
        "title": "Random Forest classifier",
        "method": (
            "Supervised tree ensemble trained with class weighting. "
            "Laundering probability is compared to a validation-selected "
            "threshold."
        ),
        "direction": "Higher predicted laundering probability indicates higher risk.",
    },
    "isolation_forest": {
        "title": "Isolation Forest",
        "method": (
            "Unsupervised detector fitted on legitimate training rows. "
            "Its saved anomaly score is the negated decision function."
        ),
        "direction": "Higher negated decision scores indicate more anomalous behavior.",
    },
    "autoencoder": {
        "title": "Autoencoder",
        "method": (
            "Neural network fitted on legitimate training rows. The inference "
            "pipeline uses saved preprocessing, feature order, residual scales, "
            "and validation calibration."
        ),
        "direction": "Higher feature-normalized reconstruction errors indicate more anomalous behavior.",
    },
}
limitations = [
    "An alert is a prioritization signal and must not be presented as a "
    "confirmed laundering finding.",
    "The Autoencoder currently has weak held-out class separation and has "
    "not detected the positive test examples at the saved operating threshold.",
    "Model metrics describe a deterministic sample of labeled IBM HI-Small "
    "data and may not generalize to live financial activity.",
    "Thresholds are selected on validation data. The held-out test split is "
    "reserved for final reporting, not model or threshold selection.",
    "The available data and models do not replace analyst review, "
    "institution-specific controls, or regulatory reporting workflows.",
    "Training dates are not recorded in the saved metadata and are therefore "
    "not inferred here.",
]

status = model_status()
dataset_name = DATASET_PATH.name if DATASET_PATH is not None else "Unavailable"
st.caption(
    f"Configured dataset: **{dataset_name}** · "
    "The models exclude account identifiers from predictive features."
)

for model_name, description in descriptions.items():
    metrics_path = METRICS_PATHS[model_name]
    state = status[model_name]
    try:
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else {}
        )
    except (OSError, json.JSONDecodeError) as error:
        metrics = {}
        st.warning(f"Could not read {model_name} metrics: {error}")
    if not isinstance(metrics, dict):
        metrics = {}

    with st.container(border=True):
        section_heading(description["title"])
        st.write(description["method"])
        st.caption(description["direction"])

        columns = st.columns(4)
        columns[0].metric(
            "Model artifact",
            "Available" if state["model"] else "Missing",
        )
        columns[1].metric(
            "Metrics artifact",
            "Available" if state["metrics"] else "Missing",
        )
        columns[2].metric(
            "Validation threshold",
            format_metric(metrics.get("threshold")),
        )
        columns[3].metric(
            "Threshold strategy",
            str(metrics.get("threshold_strategy", "Not recorded"))
            .replace("_", " ")
            .title(),
        )

        provenance_columns = st.columns(4)
        for column, key, label in zip(
            provenance_columns,
            ("training_rows", "validation_rows", "test_rows", "random_state"),
            ("Training rows", "Validation rows", "Test rows", "Random seed"),
        ):
            column.metric(label, format_metric(metrics.get(key)))

        configuration = {
            key: metrics[key]
            for key in (
                "threshold_quantile",
                "target_validation_false_positive_rate",
                "validation_normal_false_positive_rate",
                "score_direction",
                "score_method",
                "reconstruction_loss",
                "latent_dimension",
                "hidden_dimension",
                "epochs_requested",
                "epochs_trained",
                "batch_size",
                "contamination",
            )
            if key in metrics
        }
        if configuration:
            with st.expander("Training and scoring configuration"):
                st.json(configuration)

        test_metrics = [
            key
            for key in (
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
            if key in metrics
        ]
        if test_metrics:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Metric": key.replace("_", " ").title(),
                            "Held-out test result": metrics[key],
                        }
                        for key in test_metrics
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

section_heading("Limitations and responsible use")
for limitation in limitations:
    st.markdown(f"- {limitation}")
