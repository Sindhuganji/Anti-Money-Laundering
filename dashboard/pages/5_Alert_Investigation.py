"""Review model alerts with explicitly session-scoped analyst annotations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTION_PATHS
from dashboard.components.ui import (
    configure_page,
    page_header,
    section_heading,
)

configure_page("Alert investigation")
page_header(
    "Alert investigation",
    "Review model-ranked transactions, capture temporary analyst notes, "
    "and export a worklist without implying investigative outcomes.",
    "CASE REVIEW",
)

available = {
    name.replace("_", " ").title(): path
    for name, path in PREDICTION_PATHS.items()
    if path.is_file()
}
latest = st.session_state.get("latest_prediction_report")
if isinstance(latest, pd.DataFrame):
    available["Current session upload"] = None

if not available:
    st.info("No prediction reports are available. Train or run a model first.")
    st.stop()

selected = st.selectbox("Prediction source", list(available))
if selected == "Current session upload":
    predictions = latest.copy()
    model_used = str(
        st.session_state.get("latest_prediction_model", "uploaded")
    )
    source_label = "Current session upload"
else:
    try:
        predictions = pd.read_csv(available[selected])
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read this prediction report: {error}")
        st.stop()
    model_used = selected.lower().replace(" ", "_")
    source_label = f"{selected} held-out benchmark"

prediction_column = next(
    (
        column
        for column in ("predicted_class", "prediction")
        if column in predictions.columns
    ),
    None,
)
score_column = next(
    (
        column
        for column in (
            "laundering_probability",
            "probability",
            "anomaly_score",
            "reconstruction_error",
        )
        if column in predictions.columns
    ),
    None,
)
if prediction_column is None or score_column is None:
    st.error("Prediction report must include a decision and ranking score.")
    st.stop()

st.caption(
    f"Report source: **{source_label}** · "
    "Saved test reports are benchmarks, not live transactions."
)
alerts_only = st.checkbox("Show flagged transactions only", value=True)
minimum_score = st.number_input(
    "Minimum model score",
    value=float(predictions[score_column].min()),
)
filtered = predictions.copy()
if alerts_only:
    filtered = filtered.loc[filtered[prediction_column] == 1]
filtered = filtered.loc[filtered[score_column] >= minimum_score]
filtered = filtered.sort_values(
    score_column,
    ascending=False,
    kind="stable",
)

alert_total = int((predictions[prediction_column] == 1).sum())
first, second, third = st.columns(3)
first.metric("Report records", f"{len(predictions):,}")
second.metric("Flagged for review", f"{alert_total:,}")
third.metric("Worklist records", f"{len(filtered):,}")

section_heading("Ranked worklist")
if filtered.empty:
    st.info("No records match the current worklist filters.")
    st.stop()

row_id_column = next(
    (
        column
        for column in (
            "transaction_id",
            "input_row_index",
            "sample_row_index",
            "transaction_index",
        )
        if column in filtered.columns
    ),
    None,
)
if row_id_column is None:
    filtered = filtered.reset_index(names="report_row")
    row_id_column = "report_row"
ids = filtered[row_id_column].astype(str)
selected_id = st.selectbox("Select a record", ids.tolist())
selected_row = filtered.loc[ids == selected_id].iloc[0]

source_key = source_label.lower().replace(" ", "_")
status_key = f"aml_investigation_{source_key}_{model_used}_{selected_id}"
annotations = st.session_state.setdefault("aml_investigation_annotations", {})
annotation = annotations.get(status_key, {})

with st.container(border=True):
    st.markdown(f"#### Record {selected_id}")
    detail_columns = st.columns(3)
    detail_columns[0].metric(
        "Model decision",
        "Flagged" if int(selected_row[prediction_column]) == 1 else "Not flagged",
    )
    detail_columns[1].metric(
        "Model score",
        f"{float(selected_row[score_column]):.8g}",
    )
    if "threshold_used" in selected_row:
        detail_columns[2].metric(
            "Decision threshold",
            f"{float(selected_row['threshold_used']):.8g}",
        )
    elif "actual" in selected_row:
        detail_columns[2].metric(
            "Dataset label",
            "Labeled suspicious" if int(selected_row["actual"]) == 1 else "Not labeled",
        )
    st.json(selected_row.to_dict())

section_heading("Session-only analyst notes")
st.caption(
    "Annotations remain in this Streamlit session and are not written to the "
    "project or a persistent case-management system."
)
with st.form("investigation_annotation"):
    status = st.selectbox(
        "Review status",
        ["Unreviewed", "In progress", "Escalated for review"],
        index=[
            "Unreviewed",
            "In progress",
            "Escalated for review",
        ].index(annotation.get("status", "Unreviewed")),
    )
    notes = st.text_area(
        "Analyst notes",
        value=annotation.get("notes", ""),
        max_chars=5000,
    )
    save_annotation = st.form_submit_button("Save session annotation")

if save_annotation:
    annotations[status_key] = {
        "status": status,
        "notes": notes,
    }
    st.session_state["aml_investigation_annotations"] = annotations
    st.success("Session annotation saved.")

export = filtered.copy()
export["investigation_status_session_only"] = [
    annotations.get(
        f"aml_investigation_{source_key}_{model_used}_{str(value)}",
        {},
    ).get("status", "Unreviewed")
    for value in export[row_id_column]
]
export["analyst_notes_session_only"] = [
    annotations.get(
        f"aml_investigation_{source_key}_{model_used}_{str(value)}",
        {},
    ).get("notes", "")
    for value in export[row_id_column]
]
st.download_button(
    "Download investigation worklist",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=f"{model_used}_investigation_worklist.csv",
    mime="text/csv",
)
