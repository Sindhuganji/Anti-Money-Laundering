"""Feature engineering derived exclusively from transaction columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN

TARGET_COLUMNS = {TARGET_COLUMN, "Is_Laundering"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add safe, reproducible transaction features."""
    engineered = df.drop(columns=TARGET_COLUMNS, errors="ignore").copy()

    synthetic_schema = bool(
        {"Sender_Bank", "Receiver_Bank", "Transaction_Type", "Currency"}
        & set(engineered.columns)
    )
    if not synthetic_schema:
        received = pd.to_numeric(
            engineered.get(
                "Amount Received",
                pd.Series(np.nan, index=df.index),
            ),
            errors="coerce",
        )
        paid = pd.to_numeric(
            engineered.get("Amount Paid", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )

        engineered["Amount_Received_Log"] = np.log1p(received.clip(lower=0))
        engineered["Amount_Paid_Log"] = np.log1p(paid.clip(lower=0))
        engineered["Amount_Ratio"] = np.divide(
            received,
            paid.clip(lower=1e-9),
            out=np.full(len(engineered), np.nan, dtype=float),
            where=np.asarray(paid) > 0,
        )
        engineered["Amount_Difference"] = paid - received
        engineered["Amount_Mismatch"] = (
            ~np.isclose(received.fillna(-1), paid.fillna(-2), rtol=1e-6)
        ).astype("int8")
    elif "Amount" in engineered:
        amount = pd.to_numeric(engineered["Amount"], errors="coerce")
        engineered["Amount_Log"] = np.log1p(amount.clip(lower=0))

    timestamp = pd.to_datetime(
        engineered.get("Timestamp", pd.Series(index=df.index, dtype=object)),
        errors="coerce",
    )
    if "Hour" not in engineered:
        engineered["Hour"] = timestamp.dt.hour
    engineered["Day"] = timestamp.dt.day
    engineered["Month"] = timestamp.dt.month
    engineered["DayOfWeek"] = timestamp.dt.dayofweek
    engineered["Is_Weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype("int8")

    if not synthetic_schema:
        receiving_currency = engineered.get(
            "Receiving Currency", pd.Series(index=df.index, dtype=object)
        )
        payment_currency = engineered.get(
            "Payment Currency", pd.Series(index=df.index, dtype=object)
        )
        engineered["Currency_Mismatch"] = (
            receiving_currency.fillna("__missing__")
            != payment_currency.fillna("__missing__")
        ).astype("int8")

    return engineered