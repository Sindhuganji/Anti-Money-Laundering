"""Explicit, schema-routed model training workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.ui import configure_page, page_header, section_heading
from dashboard.transaction_workflow import train_selected_model
from src.prediction import get_model_metadata
from src.model_registry import resolve_model_profile

configure_page("Train Model")
page_header(
    "Train / retrain model",
    "Training is an explicit action. Uploading a file or analyzing transactions "
    "never trains or changes a saved model.",
    "MODEL OPERATIONS",
)

model_labels = {
    "Random Forest": "random_forest",
    "Isolation Forest": "isolation_forest",
    "Autoencoder": "autoencoder",
}

section_heading("Training configuration")
selected_model = st.selectbox("Algorithm", list(model_labels))
model_name = model_labels[selected_model]

section_heading("Labeled training data")
training_file = st.file_uploader(
    "Upload a labeled training CSV",
    type=["csv"],
    help=(
        "The transaction schema determines the isolated artifact destination. "
        "Include Is_Laundering (or Is Laundering) as the target."
    ),
)
training_data: pd.DataFrame | None = None
training_data_error: str | None = None
if training_file is not None:
    try:
        training_data = pd.read_csv(training_file)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        st.error(f"Could not read the training CSV: {error}")
    else:
        if training_data.empty:
            st.error("The training CSV contains no transaction rows.")
            training_data = None
        else:
            try:
                resolve_model_profile(training_data)
            except ValueError as error:
                training_data_error = str(error)
                st.error(f"Training CSV schema is not supported: {error}")
            else:
                st.caption(
                    f"Loaded {len(training_data):,} rows and "
                    f"{len(training_data.columns):,} columns. The schema "
                    "will select isolated artifacts automatically. No model "
                    "training has started."
                )

try:
    metadata = get_model_metadata(
        model_name,
        transactions=training_data if training_data_error is None else None,
    )
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not read the selected model metadata: {error}")
else:
    status_columns = st.columns(3)
    status_columns[0].metric("Model status", str(metadata["status"]))
    status_columns[1].metric("Model version", str(metadata["version"]))
    threshold = metadata["threshold"]
    status_columns[2].metric(
        "Saved threshold",
        "—" if threshold is None else f"{float(threshold):.8g}",
    )

if st.button(
    "Train / Retrain selected model",
    type="primary",
    disabled=training_data is None or training_data_error is not None,
):
    if training_data is not None:
        try:
            with st.spinner(f"Training {selected_model}…"):
                result = train_selected_model(
                    training_data,
                    model_name=model_name,
                )
        except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
            st.error(f"Training failed: {error}")
        else:
            metadata = get_model_metadata(
                model_name,
                transactions=training_data,
            )
            st.success(
                f"Training completed. Saved {selected_model} model "
                f"{metadata['version']}."
            )
            st.metric(
                "Validation-selected threshold",
                f"{float(result['threshold']):.8g}",
            )
            st.caption(
                "Training and threshold selection used the model's training "
                "and validation splits; the test split remains held out."
            )
