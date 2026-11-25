"""Small synthetic smoke tests for training integrations.

Synthetic transactions here are isolated unit-test fixtures, not project data
or evidence of model performance.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src import (
    prediction,
    train_autoencoder,
    train_isolation_forest,
    train_random_forest,
)
from src.threshold_optimization import (
    select_normal_validation_threshold,
    threshold_table,
)


def _transactions() -> pd.DataFrame:
    rows = []
    for index in range(120):
        suspicious = int(index % 6 == 0)
        amount = float(index + 1)
        rows.append({
            "Timestamp": f"2022/09/{1 + index % 20:02} 12:{index % 60:02}",
            "From Bank": index % 7,
            "Account": f"from-{index}",
            "To Bank": (index + 2) % 7,
            "Account.1": f"to-{index}",
            "Amount Received": amount,
            "Receiving Currency": "US Dollar" if index % 2 else "Euro",
            "Amount Paid": amount * (1.0 + suspicious),
            "Payment Currency": "US Dollar",
            "Payment Format": ("ACH", "Wire", "Cheque")[index % 3],
            TARGET_COLUMN: suspicious,
        })
    return pd.DataFrame(rows)


def _single_transaction_values(schema) -> dict[str, object]:
    values: dict[str, object] = {
        "Timestamp": "2022/09/01 00:17",
        "From Bank": "20",
        "To Bank": "21",
        "Account": "origin-id",
        "Account.1": "destination-id",
        "Amount Received": "100.0",
        "Amount Paid": "100.0",
    }
    for column, categories in schema.categorical_options.items():
        values[column] = categories[0]
    return values


def test_normal_only_threshold_ignores_laundering_validation_scores() -> None:
    threshold = select_normal_validation_threshold(
        y_true=pd.Series([0, 0, 0, 1]).to_numpy(),
        anomaly_scores=pd.Series([0.1, 0.2, 0.3, 100.0]).to_numpy(),
        quantile=0.5,
    )

    assert threshold == pytest.approx(0.2)


def test_autoencoder_threshold_comparison_uses_normal_scores_only() -> None:
    labels = pd.Series([0, 0, 0, 0, 1]).to_numpy()
    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0]).to_numpy()

    table = train_autoencoder._validation_threshold_table(
        labels,
        scores,
        selected_quantile=0.75,
    )

    q99 = table.loc[
        table["normal_validation_quantile"] == 0.99
    ].iloc[0]
    q98 = table.loc[
        table["normal_validation_quantile"] == 0.98
    ].iloc[0]
    q75 = table.loc[
        table["normal_validation_quantile"] == 0.75
    ].iloc[0]
    assert q99["threshold"] == pytest.approx(
        np.quantile([1.0, 2.0, 3.0, 4.0], 0.99)
    )
    assert q98["threshold"] == pytest.approx(
        np.quantile([1.0, 2.0, 3.0, 4.0], 0.98)
    )
    assert q75["threshold"] == pytest.approx(3.25)


def test_threshold_table_supports_strict_anomaly_cutoffs() -> None:
    table = threshold_table(
        y_true=pd.Series([1]).to_numpy(),
        positive_scores=pd.Series([0.5]).to_numpy(),
        strict=True,
    )

    selected_score = table.loc[table["threshold"] == 0.5].iloc[0]
    assert selected_score["tp"] == 0


def test_random_forest_training_saves_validation_threshold_and_predicts(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "models" / "random_forest.joblib"
    metrics_path = tmp_path / "metrics" / "random_forest.json"
    prediction_path = tmp_path / "predictions" / "random_forest.csv"
    plots_path = tmp_path / "plots"

    monkeypatch.setattr(train_random_forest, "RANDOM_FOREST_MODEL_PATH", model_path)
    monkeypatch.setattr(train_random_forest, "RANDOM_FOREST_METRICS_PATH", metrics_path)
    monkeypatch.setattr(
        train_random_forest,
        "RANDOM_FOREST_PREDICTIONS_PATH",
        prediction_path,
    )
    monkeypatch.setattr(train_random_forest, "MODEL_DIR", model_path.parent)
    monkeypatch.setattr(train_random_forest, "METRICS_DIR", metrics_path.parent)
    monkeypatch.setattr(
        train_random_forest,
        "PREDICTIONS_DIR",
        prediction_path.parent,
    )
    monkeypatch.setattr(train_random_forest, "PLOTS_DIR", plots_path)
    monkeypatch.setattr(
        prediction,
        "MODEL_ARTIFACTS",
        {
            **prediction.MODEL_ARTIFACTS,
            "random_forest": (model_path, metrics_path),
        },
    )

    metrics = train_random_forest.train_random_forest(
        _transactions(),
        test_size=0.15,
        validation_size=0.15,
        random_state=17,
        estimators=5,
    )
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert model_path.is_file()
    assert prediction_path.is_file()
    assert metrics["threshold"] == saved["threshold"]
    assert metrics["validation_rows"] > 0
    assert metrics["test_rows"] > 0
    assert metrics["false_positive_rate"] >= 0

    incomplete_transaction = pd.DataFrame({"Amount Received": [50.0]})
    predicted, scores = prediction.predict_transactions(
        incomplete_transaction,
        "random_forest",
    )
    assert (predicted == (scores >= saved["threshold"])).all()
    report = prediction.predict_transaction_report(
        incomplete_transaction,
        "random_forest",
    )
    assert "alert_status" in report.columns
    assert "laundering_probability" in report.columns
    assert "Amount Paid" in report.iloc[0]["input_quality_warning"]
    assert report.iloc[0]["alert_status"] != "Confirmed money laundering"
    schema = prediction.get_model_input_schema("random_forest")
    assert "Account" not in schema.required_columns
    assert "Account.1" not in schema.required_columns
    assert TARGET_COLUMN not in schema.required_columns
    single = prediction.build_single_transaction_frame(
        _single_transaction_values(schema),
        schema,
    )
    single_prediction, single_score = prediction.predict_transactions(
        single,
        "random_forest",
    )
    single_report = prediction.predict_transaction_report(
        single,
        "random_forest",
    )
    assert int(single_report.iloc[0]["predicted_class"]) == int(
        single_prediction[0]
    )
    assert single_report.iloc[0]["laundering_probability"] == pytest.approx(
        single_score[0]
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        prediction.predict_transaction_report(
            incomplete_transaction,
            "random_forest",
            threshold=1.1,
        )


def _patch_model_paths(monkeypatch, module, model_name: str, tmp_path) -> None:
    extension = ".keras" if model_name == "autoencoder" else ".joblib"
    model_path = tmp_path / "models" / f"{model_name}{extension}"
    metrics_path = tmp_path / "metrics" / f"{model_name}_metrics.json"
    prediction_path = tmp_path / "predictions" / f"{model_name}.csv"
    plots_path = tmp_path / "plots"

    model_constant = {
        "random_forest": "RANDOM_FOREST_MODEL_PATH",
        "isolation_forest": "ISOLATION_FOREST_MODEL_PATH",
        "autoencoder": "AUTOENCODER_MODEL_PATH",
    }[model_name]
    metrics_constant = {
        "random_forest": "RANDOM_FOREST_METRICS_PATH",
        "isolation_forest": "ISOLATION_FOREST_METRICS_PATH",
        "autoencoder": "AUTOENCODER_METRICS_PATH",
    }[model_name]
    predictions_constant = {
        "random_forest": "RANDOM_FOREST_PREDICTIONS_PATH",
        "isolation_forest": "ISOLATION_FOREST_PREDICTIONS_PATH",
        "autoencoder": "AUTOENCODER_PREDICTIONS_PATH",
    }[model_name]
    monkeypatch.setattr(module, model_constant, model_path)
    monkeypatch.setattr(module, metrics_constant, metrics_path)
    monkeypatch.setattr(module, predictions_constant, prediction_path)
    monkeypatch.setattr(module, "MODEL_DIR", model_path.parent, raising=False)
    monkeypatch.setattr(module, "METRICS_DIR", metrics_path.parent)
    monkeypatch.setattr(module, "PREDICTIONS_DIR", prediction_path.parent)
    monkeypatch.setattr(module, "PLOTS_DIR", plots_path)
    return model_path, metrics_path, prediction_path


def test_isolation_forest_uses_saved_validation_threshold(
    tmp_path,
    monkeypatch,
) -> None:
    model_path, metrics_path, _ = _patch_model_paths(
        monkeypatch,
        train_isolation_forest,
        "isolation_forest",
        tmp_path,
    )
    monkeypatch.setitem(
        prediction.MODEL_ARTIFACTS,
        "isolation_forest",
        (model_path, metrics_path),
    )

    metrics = train_isolation_forest.train_isolation_forest(
        _transactions(),
        random_state=23,
        estimators=5,
    )
    scored, scores = prediction.predict_transactions(
        _transactions().drop(columns=[TARGET_COLUMN]).iloc[:8],
        "isolation_forest",
    )
    threshold = json.loads(metrics_path.read_text(encoding="utf-8"))["threshold"]
    assert model_path.is_file()
    assert metrics["score_direction"].startswith("higher")
    assert len(scored) == 8
    assert metrics["threshold"] == threshold
    assert (scored == (scores >= threshold)).all()
    schema = prediction.get_model_input_schema("isolation_forest")
    single = prediction.build_single_transaction_frame(
        _single_transaction_values(schema),
        schema,
    )
    single_prediction, single_score = prediction.predict_transactions(
        single,
        "isolation_forest",
    )
    single_report = prediction.predict_transaction_report(
        single,
        "isolation_forest",
    )
    assert int(single_report.iloc[0]["predicted_class"]) == int(
        single_prediction[0]
    )
    assert single_report.iloc[0]["anomaly_score"] == pytest.approx(
        single_score[0]
    )


def test_autoencoder_training_and_inference_use_fitted_preprocessor(
    tmp_path,
    monkeypatch,
) -> None:
    model_path, metrics_path, prediction_path = _patch_model_paths(
        monkeypatch,
        train_autoencoder,
        "autoencoder",
        tmp_path,
    )
    preprocessor_path = tmp_path / "models" / "autoencoder_preprocessor.joblib"
    monkeypatch.setattr(
        train_autoencoder,
        "AUTOENCODER_PREPROCESSOR_PATH",
        preprocessor_path,
    )
    monkeypatch.setattr(
        prediction,
        "AUTOENCODER_PREPROCESSOR_PATH",
        preprocessor_path,
    )
    monkeypatch.setitem(
        prediction.MODEL_ARTIFACTS,
        "autoencoder",
        (model_path, metrics_path),
    )

    metrics = train_autoencoder.train_autoencoder(
        _transactions(),
        random_state=29,
        epochs=2,
        batch_size=16,
    )
    labels, scores = prediction.predict_transactions(
        _transactions().drop(columns=[TARGET_COLUMN]).iloc[:8],
        "autoencoder",
    )

    assert model_path.is_file()
    assert preprocessor_path.is_file()
    assert len(labels) == len(scores) == 8
    assert metrics["score_direction"].startswith("higher")
    assert (labels == (scores > metrics["threshold"])).all()
    assert metrics["epochs_trained"] == 2
    assert metrics["threshold_strategy"] == "normal_validation_quantile"
    assert metrics["threshold_quantile"] == pytest.approx(0.999)
    assert metrics["target_validation_false_positive_rate"] == pytest.approx(
        0.001
    )
    assert metrics["score_method"] == "feature_normalized_mae"
    assert len(metrics["score_feature_names"]) == len(
        metrics["score_feature_scales"]
    )
    assert set(metrics["test_raw_mse_distribution"]) == {
        "normal",
        "laundering",
    }
    assert "raw_reconstruction_mse" in pd.read_csv(prediction_path).columns
    assert metrics["threshold"] == json.loads(
        metrics_path.read_text(encoding="utf-8")
    )["threshold"]
    schema = prediction.get_model_input_schema("autoencoder")
    single = prediction.build_single_transaction_frame(
        _single_transaction_values(schema),
        schema,
    )
    single_prediction, single_score = prediction.predict_transactions(
        single,
        "autoencoder",
    )
    single_report = prediction.predict_transaction_report(
        single,
        "autoencoder",
    )
    assert int(single_report.iloc[0]["predicted_class"]) == int(
        single_prediction[0]
    )
    assert single_prediction[0] == int(single_score[0] > metrics["threshold"])
    assert single_report.iloc[0]["reconstruction_error"] == pytest.approx(
        single_score[0]
    )
    assert single_report.iloc[0]["score_method"] == "feature_normalized_mae"
