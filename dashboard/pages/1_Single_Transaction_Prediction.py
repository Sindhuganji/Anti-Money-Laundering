"""Score one transaction using a selected saved AML model."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import (
    ModelNotTrainedError,
    build_single_transaction_frame,
    get_model_metadata,
    get_model_input_schema,
    predict_transaction_report,
)
from dashboard.components.ui import configure_page, page_header, section_heading

configure_page("Single Transaction Prediction")
page_header(
    "Single transaction assessment",
    "Score one transaction using the selected saved model, preprocessing "
    "pipeline, and validation threshold.",
    "TRANSACTION REVIEW",
)

MODEL_NAMES = {
    "Random Forest": "random_forest",
    "Isolation Forest": "isolation_forest",
    "Autoencoder": "autoencoder",
}
LABELS = {
    "Timestamp": "Transaction timestamp",
    "From Bank": "Origin bank (From Bank)",
    "To Bank": "Destination bank (To Bank)",
    "Amount Received": "Amount received",
    "Amount Paid": "Amount paid",
    "Receiving Currency": "Receiving currency",
    "Payment Currency": "Payment currency",
    "Payment Format": "Transaction type (Payment Format)",
    "Sender_Bank": "Sender bank",
    "Receiver_Bank": "Receiver bank",
    "Transaction_Type": "Transaction type",
    "Amount": "Transaction amount",
    "Currency": "Currency",
    "Payment_Format": "Payment format",
    "Cross_Border": "Cross-border",
    "Day_Of_Week": "Day of week",
}
WIDGET_KEYS = {
    *(f"single_input_{column}" for column in (
        "Timestamp",
        "From Bank",
        "To Bank",
        "Amount Received",
        "Amount Paid",
        "Receiving Currency",
        "Payment Currency",
        "Payment Format",
        "Sender_Bank",
        "Receiver_Bank",
        "Transaction_Type",
        "Amount",
        "Currency",
        "Payment_Format",
        "Cross_Border",
        "Hour",
        "Day_Of_Week",
        "Account",
        "Account.1",
    )),
}


def _reset_form() -> None:
    for key in WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("single_transaction_result", None)
    st.session_state.pop("single_transaction_model", None)


@st.cache_resource(show_spinner="Loading saved preprocessing settings...")
def _get_schema(model_name: str):
    return get_model_input_schema(model_name)

selected_label = st.selectbox("Select model", list(MODEL_NAMES))
model_name = MODEL_NAMES[selected_label]

try:
    model_metadata = get_model_metadata(model_name)
    schema = _get_schema(model_name)
except (ModelNotTrainedError, OSError, ValueError, KeyError) as error:
    st.error(f"Model inputs are unavailable: {error}")
    st.stop()

metadata_columns = st.columns(3)
metadata_columns[0].metric("Model status", str(model_metadata["status"]))
metadata_columns[1].metric("Model version", str(model_metadata["version"]))
threshold = model_metadata["threshold"]
metadata_columns[2].metric(
    "Decision threshold",
    "—" if threshold is None else f"{float(threshold):.8g}",
)
st.info(
    "Input fields are derived from the selected model's saved preprocessing "
    "pipeline. Identifier columns are excluded from model features."
)

with st.container(border=True):
    with st.form("single_transaction_prediction"):
        st.markdown("#### Transaction details")
        values: dict[str, object] = {}
        left, right = st.columns(2)

        for index, column in enumerate(schema.required_columns):
            target_column = left if index % 2 == 0 else right
            key = f"single_input_{column}"
            label = LABELS.get(column, column)
            with target_column:
                if column in schema.categorical_options:
                    options = (
                        "-- Select --",
                        *schema.categorical_options[column],
                    )
                    selection = st.selectbox(
                        label,
                        options,
                        key=key,
                    )
                    values[column] = (
                        None if selection == "-- Select --" else selection
                    )
                else:
                    values[column] = st.text_input(
                        label,
                        key=key,
                        help=(
                            "Required by the saved model's preprocessing "
                            "pipeline."
                        ),
                    )

        st.markdown("#### Account context")
        account_left, account_right = st.columns(2)
        with account_left:
            values["Account"] = st.text_input(
                "Origin account (context only; not scored)",
                key="single_input_Account",
            )
        with account_right:
            values["Account.1"] = st.text_input(
                "Destination account (context only; not scored)",
                key="single_input_Account.1",
            )

        submitted = st.form_submit_button(
            "Assess transaction",
            type="primary",
            use_container_width=True,
        )

st.button("Clear / reset form", on_click=_reset_form)

if submitted:
    st.session_state.pop("single_transaction_result", None)
    try:
        transaction = build_single_transaction_frame(values, schema)
        result = predict_transaction_report(transaction, model_name)
    except (ModelNotTrainedError, OSError, ValueError, TypeError, KeyError) as error:
        st.error(f"Could not predict this transaction: {error}")
    else:
        st.session_state["single_transaction_result"] = result
        st.session_state["single_transaction_model"] = model_name

saved_result = st.session_state.get("single_transaction_result")
saved_model = st.session_state.get("single_transaction_model")
if (
    isinstance(saved_result, pd.DataFrame)
    and not saved_result.empty
    and saved_model == model_name
):
    prediction = saved_result.iloc[0]
    flagged = int(prediction["predicted_class"]) == 1
    with st.container(border=True):
        section_heading("Assessment result")
        if flagged:
            st.error(
                "HIGH RISK — Flagged for further investigation. "
                "This is not a confirmed money-laundering determination."
            )
        else:
            st.success(
                "LOW RISK — Not flagged by the selected model. "
                "This does not guarantee that the transaction is legitimate."
            )

        first, second, third = st.columns(3)
        first.metric("Prediction", "Flagged" if flagged else "Not flagged")
        second.metric(
            "Selected model",
            str(saved_model).replace("_", " ").title(),
        )
        third.metric("Risk level", str(prediction["suspiciousness_level"]))

        score_column = {
            "random_forest": "laundering_probability",
            "isolation_forest": "anomaly_score",
            "autoencoder": "reconstruction_error",
        }[str(saved_model)]
        score_columns = st.columns(2)
        score_columns[0].metric(
            "Validation threshold",
            f"{float(prediction['threshold_used']):.8g}",
        )
        if score_column == "laundering_probability":
            score_columns[1].metric(
                "Laundering probability",
                f"{float(prediction[score_column]):.2%}",
            )
        else:
            score_method = str(prediction.get("score_method", ""))
            score_label = {
                "feature_normalized_mae": (
                    "Feature-normalized reconstruction error"
                ),
                "mae": "Mean absolute reconstruction error",
                "mse": "Mean squared reconstruction error",
            }.get(
                score_method,
                "Anomaly score"
                if score_column != "reconstruction_error"
                else "Reconstruction error",
            )
            score_columns[1].metric(
                score_label,
                f"{float(prediction[score_column]):.8g}",
            )

        if flagged:
            st.caption(
                "The model score met or exceeded its saved threshold. "
                "No causal explanation or investigative outcome is inferred."
            )
        else:
            st.caption(
                "The model score is below its decision threshold. "
                "A non-alert is not proof of legitimacy."
            )

        warning = str(prediction.get("input_quality_warning", ""))
        if warning:
            st.warning(warning)
