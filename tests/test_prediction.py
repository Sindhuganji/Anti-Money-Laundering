"""Tests for the unified AML prediction interface."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import prediction
from src.prediction import (
    ModelInputSchema,
    build_single_transaction_frame,
    predict_transactions,
)


def create_transaction() -> pd.DataFrame:
    return pd.DataFrame([{
        "Timestamp": "2022/09/01 00:17",
        "From Bank": 20,
        "Account": "800104D70",
        "To Bank": 20,
        "Account.1": "800104D70",
        "Amount Received": 6794.63,
        "Receiving Currency": "US Dollar",
        "Amount Paid": 6794.63,
        "Payment Currency": "US Dollar",
        "Payment Format": "Reinvestment",
    }])


def test_unknown_model_name_is_rejected() -> None:
    transaction = create_transaction()

    with pytest.raises(ValueError, match="Unknown model name"):
        predict_transactions(
            transaction,
            model_name="unknown_model",  # type: ignore[arg-type]
        )


def test_random_forest_requires_trained_artifact(tmp_path, monkeypatch) -> None:
    transaction = create_transaction()
    monkeypatch.setitem(
        prediction.MODEL_ARTIFACTS,
        "random_forest",
        (tmp_path / "random_forest.joblib", tmp_path / "metrics.json"),
    )

    with pytest.raises(FileNotFoundError, match="Missing model artifact"):
        predict_transactions(
            transaction,
            model_name="random_forest",
        )


def test_isolation_forest_requires_trained_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    transaction = create_transaction()
    monkeypatch.setitem(
        prediction.MODEL_ARTIFACTS,
        "isolation_forest",
        (tmp_path / "isolation_forest.joblib", tmp_path / "metrics.json"),
    )

    with pytest.raises(FileNotFoundError, match="Missing model artifact"):
        predict_transactions(
            transaction,
            model_name="isolation_forest",
        )


def test_autoencoder_requires_trained_artifact(tmp_path, monkeypatch) -> None:
    transaction = create_transaction()
    monkeypatch.setitem(
        prediction.MODEL_ARTIFACTS,
        "autoencoder",
        (tmp_path / "autoencoder.keras", tmp_path / "metrics.json"),
    )

    with pytest.raises(FileNotFoundError, match="Missing autoencoder model"):
        predict_transactions(
            transaction,
            model_name="autoencoder",
        )


def test_autoencoder_scoring_metadata_must_match_feature_order(
    tmp_path,
    monkeypatch,
) -> None:
    metrics_path = tmp_path / "autoencoder_metrics.json"
    model_path = tmp_path / "autoencoder.keras"
    monkeypatch.setitem(
        prediction.MODEL_ARTIFACTS,
        "autoencoder",
        (model_path, metrics_path),
    )

    class Preprocessor:
        def get_feature_names_out(self) -> np.ndarray:
            return np.array(["feature_a", "feature_b"])

    metrics_path.write_text(
        json.dumps({
            "score_method": "feature_normalized_mae",
            "score_feature_names": ["feature_a", "feature_b"],
            "score_feature_scales": [0.5, 2.0],
        }),
        encoding="utf-8",
    )
    method, scales = prediction._autoencoder_score_configuration(
        Preprocessor()
    )

    assert method == "feature_normalized_mae"
    np.testing.assert_allclose(scales, [0.5, 2.0])

    metrics_path.write_text(
        json.dumps({
            "score_method": "feature_normalized_mae",
            "score_feature_names": ["feature_b", "feature_a"],
            "score_feature_scales": [0.5, 2.0],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="feature order"):
        prediction._autoencoder_score_configuration(Preprocessor())


def test_single_transaction_input_is_validated_and_uses_dataset_columns() -> None:
    schema = ModelInputSchema(
        required_columns=(
            "Timestamp",
            "From Bank",
            "To Bank",
            "Amount Received",
            "Amount Paid",
            "Receiving Currency",
            "Payment Currency",
            "Payment Format",
        ),
        categorical_options={
            "Receiving Currency": ("US Dollar", "Euro"),
            "Payment Currency": ("US Dollar", "Euro"),
            "Payment Format": ("ACH", "Wire"),
        },
    )
    values = {
        "Timestamp": "2022/09/01 00:17",
        "From Bank": "20",
        "To Bank": "21",
        "Amount Received": "120.50",
        "Amount Paid": "120.50",
        "Receiving Currency": "US Dollar",
        "Payment Currency": "US Dollar",
        "Payment Format": "ACH",
        "Account": "origin-id",
        "Account.1": "destination-id",
    }

    transaction = build_single_transaction_frame(values, schema)

    assert len(transaction) == 1
    assert transaction.loc[0, "From Bank"] == 20
    assert transaction.loc[0, "Amount Received"] == 120.5
    assert {"Account", "Account.1"}.issubset(transaction.columns)
    assert "Is Laundering" not in transaction.columns

    with pytest.raises(ValueError, match="required transaction fields"):
        build_single_transaction_frame(
            {**values, "Amount Paid": ""},
            schema,
        )
    with pytest.raises(ValueError, match="valid date and time"):
        build_single_transaction_frame(
            {**values, "Timestamp": "not-a-date"},
            schema,
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        build_single_transaction_frame(
            {**values, "Amount Received": "-1"},
            schema,
        )


def test_autoencoder_flags_only_scores_above_its_threshold(monkeypatch) -> None:
    transactions = pd.concat(
        [create_transaction(), create_transaction()],
        ignore_index=True,
    )
    monkeypatch.setattr(
        prediction,
        "_score_transactions",
        lambda _transactions, _model_name: (
            pd.Series([0.5, 0.5001]).to_numpy(),
            0.5,
            "reconstruction_error",
            "mse",
        ),
    )

    labels, scores = prediction.predict_transactions(
        transactions,
        "autoencoder",
    )
    report = prediction.predict_transaction_report(
        transactions,
        "autoencoder",
    )

    assert scores.tolist() == [0.5, 0.5001]
    assert labels.tolist() == [0, 1]
    assert report.set_index("reconstruction_error")[
        "predicted_class"
    ].to_dict() == {0.5: 0, 0.5001: 1}