"""Batch transaction scoring and interactive review."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import (
    ModelCompatibilityError,
    ModelNotTrainedError,
    get_model_metadata,
)
from src.evaluation import evaluate_labeled_report
from dashboard.transaction_workflow import score_uploaded_transactions
from dashboard.components.ui import (
    configure_page,
    page_header,
    section_heading,
    style_figure,
)

configure_page("Transaction analysis")
page_header(
    "Transaction analysis",
    "Score a transaction file, prioritize model alerts, and inspect records "
    "with clear separation between scores and confirmed findings.",
    "MONITORING",
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help=(
        "Upload IBM HI-Small or synthetic AML columns. The matching saved "
        "model artifacts are selected automatically from the column schema."
    ),
)
if uploaded_file is None:
    st.info("Upload a CSV to start a batch assessment.")
    st.stop()

try:
    transactions = pd.read_csv(uploaded_file)
except (
    OSError,
    UnicodeDecodeError,
    pd.errors.ParserError,
    ValueError,
) as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()
upload_fingerprint = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

if transactions.empty:
    st.error("The uploaded CSV contains no transaction rows.")
    st.stop()

section_heading("Assessment configuration")
first, second = st.columns([1, 2])
with first:
    model_name = st.selectbox(
        "Select model",
        options=["random_forest", "isolation_forest", "autoencoder"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
with second:
    st.metric("Uploaded transactions", f"{len(transactions):,}")

try:
    model_metadata = get_model_metadata(model_name, transactions=transactions)
except (ModelCompatibilityError, ModelNotTrainedError, OSError, ValueError, KeyError) as error:
    model_metadata = None
    st.error(f"Saved model metadata is unavailable: {error}")

metadata_columns = st.columns(3)
metadata_columns[0].metric(
    "Model status",
    "Not trained" if model_metadata is None else model_metadata["status"],
)
metadata_columns[1].metric(
    "Model version",
    "Unavailable" if model_metadata is None else model_metadata["version"],
)
threshold = None if model_metadata is None else model_metadata["threshold"]
metadata_columns[2].metric(
    "Decision threshold",
    "Unavailable" if threshold is None else f"{float(threshold):.8g}",
)

if model_metadata is None or model_metadata["status"] != "Ready":
    st.warning(
        "No saved model is available for this algorithm and CSV schema. "
        "No fallback model will be used. Train the matching model explicitly "
        "from the Train / retrain model page."
    )
elif model_metadata["threshold"] is not None:
    st.caption(
        f"Using the saved **{model_name.replace('_', ' ').title()}** model. "
        "Its threshold was selected using validation data."
    )

if st.button(
    "Analyze transactions",
    type="primary",
    disabled=model_metadata is None or model_metadata["status"] != "Ready",
):
    try:
        results = score_uploaded_transactions(
            transactions,
            model_name=model_name,
        )
    except (
        FileNotFoundError,
        ModelNotTrainedError,
        ModelCompatibilityError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        st.error(f"Prediction unavailable: {error}")
    else:
        st.session_state["latest_prediction_report"] = results
        st.session_state["latest_prediction_model"] = model_name
        st.session_state["latest_prediction_upload"] = upload_fingerprint

latest = st.session_state.get("latest_prediction_report")
latest_model = st.session_state.get("latest_prediction_model")
latest_upload = st.session_state.get("latest_prediction_upload")
if (
    model_metadata is None
    or model_metadata["status"] != "Ready"
    or not isinstance(latest, pd.DataFrame)
    or latest_model != model_name
    or latest_upload != upload_fingerprint
):
    with st.expander("Preview uploaded records"):
        st.dataframe(
            transactions.head(50),
            hide_index=True,
            use_container_width=True,
        )
    st.stop()

results = latest.copy()
prediction_column = "predicted_class"
score_column = next(
    (
        column
        for column in (
            "laundering_probability",
            "probability",
            "anomaly_score",
            "reconstruction_error",
        )
        if column in results.columns
    ),
    None,
)
if prediction_column not in results.columns or score_column is None:
    st.error("The model report is missing its prediction or score columns.")
    st.stop()

alerts = results.loc[results[prediction_column] == 1]
kpi_columns = st.columns(4)
kpi_columns[0].metric("Total transactions", f"{len(transactions):,}")
kpi_columns[1].metric("Scored transactions", f"{len(results):,}")
kpi_columns[2].metric("Flagged transactions", f"{len(alerts):,}")
kpi_columns[3].metric(
    "Alert rate",
    f"{len(alerts) / len(results):.2%}" if len(results) else "—",
)
st.caption(
    f"Score: **{score_column.replace('_', ' ').title()}** · "
    f"Threshold: **{float(model_metadata['threshold']):.8g}** · "
    f"Algorithm: **{model_name.replace('_', ' ').title()}** · "
    f"Version: **{model_metadata['version']}**"
)

if "ground_truth_label" in results:
    try:
        evaluation = evaluate_labeled_report(results, score_column)
    except ValueError as error:
        st.warning(f"Ground-truth evaluation is unavailable: {error}")
    else:
        section_heading("Evaluation against uploaded ground-truth labels")
        metric_columns = st.columns(6)
        for column, key, label in zip(
            metric_columns,
            (
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "average_precision",
                "false_positive_rate",
            ),
            ("Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR"),
        ):
            value = evaluation.get(key)
            column.metric(
                label,
                "—" if value is None else f"{float(value):.4f}",
            )
        st.dataframe(
            pd.DataFrame(
                evaluation["confusion_matrix"],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            ),
            use_container_width=True,
        )
        st.caption(
            "Predictions and model scores were generated before labels were "
            "read for evaluation. Labels are not prediction inputs."
        )

section_heading("Filter and review results")
left, middle, right = st.columns([1.2, 1, 1])
with left:
    search_term = st.text_input("Search visible fields")
with middle:
    risk_filter = st.selectbox(
        "Risk filter",
        ["All records", "Flagged only", "Not flagged"],
    )
with right:
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)

filtered = results.copy()
if risk_filter == "Flagged only":
    filtered = filtered.loc[filtered[prediction_column] == 1]
elif risk_filter == "Not flagged":
    filtered = filtered.loc[filtered[prediction_column] == 0]
if search_term:
    searchable = filtered.astype("string")
    matches = searchable.apply(
        lambda column: column.str.contains(
            search_term,
            case=False,
            regex=False,
            na=False,
        )
    ).any(axis=1)
    filtered = filtered.loc[matches]

sort_options = [score_column]
if "sample_row_index" in filtered.columns:
    sort_options.append("sample_row_index")
sort_column = st.selectbox(
    "Sort by",
    sort_options,
    format_func=lambda value: value.replace("_", " ").title(),
)
sort_descending = sort_column == score_column
filtered = filtered.sort_values(
    sort_column,
    ascending=not sort_descending,
    kind="stable",
)
page_count = max(1, (len(filtered) + page_size - 1) // page_size)
page_number = st.number_input(
    "Page",
    min_value=1,
    max_value=page_count,
    value=1,
    step=1,
)
start = (int(page_number) - 1) * page_size
page = filtered.iloc[start : start + page_size]
st.caption(
    f"Showing {len(page):,} of {len(filtered):,} filtered rows "
    f"(page {int(page_number)} of {page_count})."
)

if score_column in page.columns:
    styled = page.style.map(
        lambda value: "color: #f27e8a; font-weight: 600"
        if value == 1
        else "color: #45d6a0",
        subset=[prediction_column],
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)
else:
    st.dataframe(page, hide_index=True, use_container_width=True)

download_columns = st.columns([1, 1, 3])
with download_columns[0]:
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{model_name}_filtered_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

if not filtered.empty and "sample_row_index" in filtered.columns:
    options = filtered["sample_row_index"].astype(str).tolist()
    with download_columns[1]:
        selected_row = st.selectbox("Inspect row", options)
    row_index = filtered["sample_row_index"].astype(str) == selected_row
    detail = filtered.loc[row_index].iloc[0].to_dict()
    with st.expander(f"Transaction details · row {selected_row}", expanded=True):
        st.json(detail)
elif not filtered.empty:
    with st.expander("Selected transaction detail"):
        row_number = st.number_input(
            "Filtered row number",
            min_value=1,
            max_value=len(filtered),
            value=1,
            step=1,
        )
        st.json(filtered.iloc[int(row_number) - 1].to_dict())

section_heading("Score distribution")
figure = px.histogram(
    results,
    x=score_column,
    color=prediction_column,
    nbins=50,
    marginal="box",
    title=f"{score_column.replace('_', ' ').title()} by model decision",
    labels={score_column: "Model score", prediction_column: "Predicted class"},
)
st.plotly_chart(
    style_figure(figure, height=390),
    use_container_width=True,
)
st.caption(
    "Flagged means the model score crossed its saved decision threshold. "
    "It is not a confirmed money-laundering determination."
)
