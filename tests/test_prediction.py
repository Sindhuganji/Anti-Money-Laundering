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
from src.model_registry import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILES,
    MODEL_REGISTRY,
    ModelArtifacts,
    resolve_model_profile,
)


def _set_registry_artifacts(
    monkeypatch,
    model_name: str,
    profile: str,
    model_path,
    metrics_path,
    predictions_path=None,
    preprocessor_path=None,
) -> None:
    current = MODEL_REGISTRY[model_name][profile]
    monkeypatch.setitem(
        MODEL_REGISTRY[model_name],
        profile,
        ModelArtifacts(
            model_path,
            metrics_path,
            predictions_path or current.predictions,
            preprocessor_path
            if preprocessor_path is not None
            else current.preprocessor,
        ),
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


def test_registry_routes_profiles_without_crossing_artifact_paths() -> None:
    assert resolve_model_profile() == DEFAULT_MODEL_PROFILE == "ibm"
    assert resolve_model_profile(create_transaction()) == "ibm"
    assert MODEL_PROFILES == ("ibm", "synthetic")
    for model_name in MODEL_REGISTRY:
        assert (
            MODEL_REGISTRY[model_name]["ibm"].model
            != MODEL_REGISTRY[model_name]["synthetic"].model
        )


def test_unknown_model_name_is_rejected() -> None:
    transaction = create_transaction()

    with pytest.raises(ValueError, match="Unknown model name"):
        predict_transactions(
            transaction,
            model_name="unknown_model",  # type: ignore[arg-type]
        )


def test_random_forest_requires_trained_artifact(tmp_path, monkeypatch) -> None:
    transaction = create_transaction()
    _set_registry_artifacts(
        monkeypatch,
        "random_forest",
        "ibm",
        tmp_path / "random_forest.joblib",
        tmp_path / "metrics.json",
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
    _set_registry_artifacts(
        monkeypatch,
        "isolation_forest",
        "ibm",
        tmp_path / "isolation_forest.joblib",
        tmp_path / "metrics.json",
    )

    with pytest.raises(FileNotFoundError, match="Missing model artifact"):
        predict_transactions(
            transaction,
            model_name="isolation_forest",
        )


def test_autoencoder_requires_trained_artifact(tmp_path, monkeypatch) -> None:
    transaction = create_transaction()
    _set_registry_artifacts(
        monkeypatch,
        "autoencoder",
        "ibm",
        tmp_path / "autoencoder.keras",
        tmp_path / "metrics.json",
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
    _set_registry_artifacts(
        monkeypatch,
        "autoencoder",
        "ibm",
        model_path,
        metrics_path,
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


def test_feature_alignment_preserves_trained_order_and_drops_extras() -> None:
    class Model:
        feature_names_in_ = np.array(["second", "first", "missing"])

    aligned = prediction._align_features(
        pd.DataFrame({"first": [1.0], "second": [2.0], "unused": [3.0]}),
        Model(),
    )

    assert aligned.columns.tolist() == ["second", "first", "missing"]
    assert aligned.iloc[0]["second"] == 2.0
    assert aligned.iloc[0]["first"] == 1.0
    assert np.isnan(aligned.iloc[0]["missing"])


@pytest.mark.parametrize(
    ("model_name", "scores", "threshold", "expected"),
    [
        ("random_forest", [0.49, 0.5], 0.5, [0, 1]),
        ("isolation_forest", [-0.1, 0.1], 0.0, [0, 1]),
        ("autoencoder", [0.5, 0.51], 0.5, [0, 1]),
    ],
)
def test_threshold_application_uses_model_score_contract(
    model_name,
    scores,
    threshold,
    expected,
) -> None:
    labels = prediction._classify_scores(
        np.asarray(scores),
        threshold,
        model_name,
    )

    assert labels.tolist() == expected


def test_isolation_forest_score_direction_is_higher_for_anomalies(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "isolation.joblib"
    metrics_path = tmp_path / "metrics.json"
    model_path.touch()
    metrics_path.write_text(json.dumps({"threshold": 0.0}), encoding="utf-8")
    _set_registry_artifacts(
        monkeypatch,
        "isolation_forest",
        "ibm",
        model_path,
        metrics_path,
    )

    class Detector:
        def decision_function(self, features):
            return np.array([0.3, -0.2])[: len(features)]

    monkeypatch.setattr(prediction.joblib, "load", lambda _path: Detector())
    monkeypatch.setattr(
        prediction,
        "get_model_input_schema",
        lambda *_args, **_kwargs: ModelInputSchema(
            required_columns=tuple(create_transaction().columns),
            categorical_options={},
        ),
    )
    transactions = pd.concat(
        [create_transaction(), create_transaction()],
        ignore_index=True,
    )

    labels, scores = predict_transactions(transactions, "isolation_forest")

    np.testing.assert_allclose(scores, [-0.3, 0.2])
    assert labels.tolist() == [0, 1]


def test_random_forest_uses_label_one_probability(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "random_forest.joblib"
    metrics_path = tmp_path / "metrics.json"
    model_path.touch()
    metrics_path.write_text(json.dumps({"threshold": 0.6}), encoding="utf-8")
    _set_registry_artifacts(
        monkeypatch,
        "random_forest",
        "ibm",
        model_path,
        metrics_path,
    )

    class Classifier:
        classes_ = np.array([1, 0])

        def predict_proba(self, features):
            return np.tile([0.7, 0.3], (len(features), 1))

    monkeypatch.setattr(prediction.joblib, "load", lambda _path: Classifier())
    monkeypatch.setattr(
        prediction,
        "get_model_input_schema",
        lambda *_args, **_kwargs: ModelInputSchema(
            required_columns=tuple(create_transaction().columns),
            categorical_options={},
        ),
    )

    labels, scores = predict_transactions(
        create_transaction(),
        "random_forest",
    )

    assert scores.tolist() == [0.7]
    assert labels.tolist() == [1]


def test_synthetic_input_never_falls_back_to_ibm_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    synthetic = pd.DataFrame([{
        "Timestamp": "2026-01-01 12:00:00",
        "Sender_ID": "sender-1",
        "Receiver_ID": "receiver-1",
        "Sender_Bank": "Bank A",
        "Receiver_Bank": "Bank B",
        "Transaction_Type": "Wire",
        "Amount": 100.0,
        "Currency": "USD",
        "Payment_Format": "SWIFT",
        "Cross_Border": True,
        "Hour": 12,
        "Day_Of_Week": 3,
        "Is_Laundering": 0,
    }])
    synthetic_model_path = tmp_path / "synthetic-rf.joblib"
    synthetic_metrics_path = tmp_path / "synthetic-metrics.json"
    _set_registry_artifacts(
        monkeypatch,
        "random_forest",
        "synthetic",
        synthetic_model_path,
        synthetic_metrics_path,
        predictions_path=tmp_path / "synthetic-predictions.csv",
        preprocessor_path=None,
    )
    monkeypatch.setattr(
        prediction.joblib,
        "load",
        lambda _path: pytest.fail("IBM artifact must not be loaded"),
    )

    with pytest.raises(FileNotFoundError, match="separately trained synthetic"):
        predict_transactions(synthetic, "random_forest")


def test_report_evaluates_label_after_generating_predictions(monkeypatch) -> None:
    synthetic = pd.DataFrame({
        "Timestamp": ["2026-01-01 12:00:00", "2026-01-02 13:00:00"],
        "Sender_Bank": ["Bank A", "Bank B"],
        "Receiver_Bank": ["Bank B", "Bank A"],
        "Transaction_Type": ["Wire", "ACH"],
        "Amount": [100.0, 200.0],
        "Currency": ["USD", "EUR"],
        "Payment_Format": ["SWIFT", "ACH"],
        "Cross_Border": [True, False],
        "Hour": [12, 13],
        "Day_Of_Week": [3, 4],
        "Is_Laundering": [0, 1],
    })
    monkeypatch.setattr(
        prediction,
        "_score_transactions",
        lambda _transactions, _model_name: (
            np.array([0.2, 0.9]),
            0.5,
            "laundering_probability",
            "random_forest_class_1_probability",
        ),
    )

    report = prediction.predict_transaction_report(
        synthetic,
        "random_forest",
    )
    from src.evaluation import evaluate_labeled_report

    metrics = evaluate_labeled_report(report, "laundering_probability")

    assert report["ground_truth_label"].tolist() == [1, 0]
    assert report["predicted_class"].tolist() == [1, 0]
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert metrics["alert_rate"] == 0.5
    assert metrics["average_precision"] == 1.0