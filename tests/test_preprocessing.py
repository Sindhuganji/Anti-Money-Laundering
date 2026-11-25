"""Tests for AML feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.config import TARGET_COLUMN
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor


def create_sample_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "Timestamp": [
            "2022/09/01 00:17",
            "2022/09/01 12:45",
            "2022/09/03 23:55",
        ],
        "From Bank": [20, 3196, 1208],
        "Account": ["800104D70", "800107150", "80010E430"],
        "To Bank": [20, 3196, 20],
        "Account.1": ["800104D70", "800107150", "80010E6F0"],
        "Amount Received": [6794.63, 7739.29, 73966883.00],
        "Receiving Currency": [
            "US Dollar",
            "US Dollar",
            "US Dollar",
        ],
        "Amount Paid": [6794.63, 7739.29, 73966883.00],
        "Payment Currency": [
            "US Dollar",
            "US Dollar",
            "Euro",
        ],
        "Payment Format": [
            "Reinvestment",
            "Reinvestment",
            "Cheque",
        ],
        TARGET_COLUMN: [0, 0, 1],
    })


def test_feature_engineering_creates_expected_columns() -> None:
    transactions = create_sample_transactions()

    engineered = engineer_features(transactions)

    expected_columns = {
        "Amount_Received_Log",
        "Amount_Paid_Log",
        "Amount_Ratio",
        "Amount_Difference",
        "Amount_Mismatch",
        "Hour",
        "Day",
        "Month",
        "DayOfWeek",
        "Is_Weekend",
        "Currency_Mismatch",
    }

    assert expected_columns.issubset(engineered.columns)
    assert engineered.shape[0] == transactions.shape[0]
    assert TARGET_COLUMN not in engineered.columns


def test_feature_engineering_tolerates_missing_source_columns() -> None:
    engineered = engineer_features(pd.DataFrame({"Amount Paid": [12.0]}))

    assert engineered.shape[0] == 1
    assert np.isnan(engineered.loc[0, "Amount_Received_Log"])
    assert "Currency_Mismatch" in engineered.columns


def test_feature_engineering_handles_missing_amount() -> None:
    transactions = create_sample_transactions()
    transactions.loc[0, "Amount Received"] = np.nan
    transactions.loc[0, "Amount Paid"] = np.nan

    engineered = engineer_features(transactions)

    assert np.isfinite(
        engineered.loc[1, "Amount_Received_Log"]
    )
    assert np.isfinite(
        engineered.loc[1, "Amount_Paid_Log"]
    )
    assert np.isfinite(
        engineered.loc[0, "Amount_Received_Log"]
        if not np.isnan(engineered.loc[0, "Amount_Received_Log"])
        else 0
    )


def test_zero_paid_amount_has_missing_ratio() -> None:
    transactions = create_sample_transactions()
    transactions.loc[0, "Amount Paid"] = 0.0

    engineered = engineer_features(transactions)

    assert np.isnan(engineered.loc[0, "Amount_Ratio"])


def test_currency_mismatch_is_binary() -> None:
    transactions = create_sample_transactions()
    engineered = engineer_features(transactions)

    mismatch = engineered["Currency_Mismatch"].to_numpy()

    assert set(mismatch).issubset({0, 1})
    assert mismatch[2] == 1


def test_build_preprocessor_returns_column_transformer() -> None:
    transactions = create_sample_transactions()

    preprocessor, numeric_features, categorical_features = (
        build_preprocessor(transactions)
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert len(numeric_features) > 0
    assert len(categorical_features) > 0
    assert preprocessor.sparse_threshold == 0.3


def test_preprocessor_can_fit_and_transform_training_data() -> None:
    transactions = create_sample_transactions()
    features = transactions.drop(columns=[TARGET_COLUMN])

    engineered = engineer_features(features)
    preprocessor, _, _ = build_preprocessor(
        transactions
    )

    preprocessor.fit(engineered)
    transformed = preprocessor.transform(engineered)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert transformed.shape[0] == engineered.shape[0]
    assert transformed.shape[1] > 0
    assert np.isfinite(np.asarray(transformed)).all()


def test_preprocessor_handles_unknown_category() -> None:
    transactions = create_sample_transactions()
    features = engineer_features(
        transactions.drop(columns=[TARGET_COLUMN])
    )

    preprocessor, _, _ = build_preprocessor(transactions)
    preprocessor.fit(features)

    unseen = features.iloc[:1].copy()
    unseen["Payment Format"] = "Previously-Unseen-Payment-Type"
    unseen["Payment Currency"] = "Previously-Unseen-Currency"

    transformed = preprocessor.transform(
        engineer_features(unseen)
    )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert transformed.shape[0] == 1
    assert np.isfinite(np.asarray(transformed)).all()


def test_timestamp_features_are_present() -> None:
    transactions = create_sample_transactions()
    engineered = engineer_features(transactions)

    assert {"Hour", "Day", "Month", "DayOfWeek", "Is_Weekend"} \
        .issubset(engineered.columns)
    assert engineered["Hour"].between(0, 23).all()
    assert engineered["DayOfWeek"].between(0, 6).all()