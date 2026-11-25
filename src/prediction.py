"""Unified transaction prediction interface."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from src.config import (
    AUTOENCODER_METRICS_PATH,
    AUTOENCODER_MODEL_PATH,
    AUTOENCODER_PREPROCESSOR_PATH,
    ISOLATION_FOREST_METRICS_PATH,
    ISOLATION_FOREST_MODEL_PATH,
    RANDOM_FOREST_METRICS_PATH,
    RANDOM_FOREST_MODEL_PATH,
)
from src.autoencoder_scoring import reconstruction_scores
from src.feature_engineering import engineer_features

ModelName = Literal[
    "random_forest",
    "isolation_forest",
    "autoencoder",
]

MODEL_ARTIFACTS = {
    "random_forest": (RANDOM_FOREST_MODEL_PATH, RANDOM_FOREST_METRICS_PATH),
    "isolation_forest": (
        ISOLATION_FOREST_MODEL_PATH,
        ISOLATION_FOREST_METRICS_PATH,
    ),
    "autoencoder": (AUTOENCODER_MODEL_PATH, AUTOENCODER_METRICS_PATH),
}

TRANSACTION_INPUT_COLUMNS = {
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
}

TRANSACTION_INPUT_COLUMN_ORDER = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
)

ENGINEERED_FEATURE_SOURCES = {
    "Amount_Received_Log": ("Amount Received",),
    "Amount_Paid_Log": ("Amount Paid",),
    "Amount_Ratio": ("Amount Received", "Amount Paid"),
    "Amount_Difference": ("Amount Paid", "Amount Received"),
    "Amount_Mismatch": ("Amount Received", "Amount Paid"),
    "Hour": ("Timestamp",),
    "Day": ("Timestamp",),
    "Month": ("Timestamp",),
    "DayOfWeek": ("Timestamp",),
    "Is_Weekend": ("Timestamp",),
    "Currency_Mismatch": ("Receiving Currency", "Payment Currency"),
}
NUMERIC_INPUT_COLUMNS = {
    "From Bank",
    "To Bank",
    "Amount Received",
    "Amount Paid",
}
CATEGORICAL_INPUT_COLUMNS = {
    "Receiving Currency",
    "Payment Currency",
    "Payment Format",
}
CONTEXT_ONLY_COLUMNS = ("Account", "Account.1")


@dataclass(frozen=True)
class ModelInputSchema:
    """Raw transaction fields required by a saved model preprocessor."""

    required_columns: tuple[str, ...]
    categorical_options: dict[str, tuple[str, ...]]


class ModelNotTrainedError(FileNotFoundError):
    """Raised when a requested trained model artifact is missing."""


def get_model_input_schema(model_name: ModelName) -> ModelInputSchema:
    """Discover raw input fields and categories from a trained preprocessor."""
    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(f"Unknown model name: {model_name}")

    model_path, _ = MODEL_ARTIFACTS[model_name]
    if not model_path.is_file():
        raise ModelNotTrainedError(
            f"Missing model artifact for {model_name.replace('_', ' ')}: "
            f"{model_path}. Train it first."
        )

    if model_name == "autoencoder":
        if not AUTOENCODER_PREPROCESSOR_PATH.is_file():
            raise ModelNotTrainedError(
                "Missing autoencoder preprocessor artifact: "
                f"{AUTOENCODER_PREPROCESSOR_PATH}. Run: python main.py train-ae"
            )
        preprocessor_pipeline = joblib.load(AUTOENCODER_PREPROCESSOR_PATH)
        preprocessor = preprocessor_pipeline.named_steps["preprocessor"]
    else:
        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps["preprocessor"]

    selected_features: set[str] = set()
    categorical_columns: list[str] = []
    for name, _, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        selected_features.update(str(column) for column in columns)
        if name == "categorical":
            categorical_columns = [str(column) for column in columns]

    raw_columns: set[str] = set()
    for feature in selected_features:
        sources = ENGINEERED_FEATURE_SOURCES.get(feature, (feature,))
        unsupported = set(sources).difference(TRANSACTION_INPUT_COLUMNS)
        if unsupported:
            raise ValueError(
                f"Model feature {feature!r} maps to unsupported raw columns: "
                f"{sorted(unsupported)}"
            )
        raw_columns.update(sources)

    required_columns = tuple(
        column
        for column in TRANSACTION_INPUT_COLUMN_ORDER
        if column in raw_columns
    )
    if not required_columns:
        raise ValueError(
            f"No supported transaction inputs found for {model_name}."
        )

    categorical_options: dict[str, tuple[str, ...]] = {}
    if categorical_columns:
        categorical_transformer = preprocessor.named_transformers_[
            "categorical"
        ]
        encoder = categorical_transformer.named_steps["encoder"]
        categorical_options = {
            column: tuple(str(value) for value in values)
            for column, values in zip(
                categorical_columns,
                encoder.categories_,
                strict=True,
            )
        }

    return ModelInputSchema(
        required_columns=required_columns,
        categorical_options=categorical_options,
    )


def build_single_transaction_frame(
    values: Mapping[str, object],
    schema: ModelInputSchema,
) -> pd.DataFrame:
    """Validate form values and construct one raw IBM AML transaction row."""
    missing = [
        column
        for column in schema.required_columns
        if column not in values
        or values[column] is None
        or not str(values[column]).strip()
    ]
    if missing:
        raise ValueError(
            "Complete all required transaction fields: "
            + ", ".join(missing)
        )

    row: dict[str, object] = {}
    for column in schema.required_columns:
        value = values[column]
        if column == "Timestamp":
            timestamp = pd.to_datetime(value, errors="coerce")
            if pd.isna(timestamp):
                raise ValueError(
                    "Timestamp must be a valid date and time "
                    "(for example, 2022/09/01 00:17)."
                )
            row[column] = str(value).strip()
        elif column in NUMERIC_INPUT_COLUMNS:
            try:
                number = float(str(value).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column} must be a valid number.") from exc
            if not math.isfinite(number):
                raise ValueError(f"{column} must be finite.")
            if number < 0:
                raise ValueError(f"{column} cannot be negative.")
            if column in {"From Bank", "To Bank"} and not number.is_integer():
                raise ValueError(f"{column} must be a whole number.")
            row[column] = (
                int(number)
                if column in {"From Bank", "To Bank"}
                else number
            )
        elif column in CATEGORICAL_INPUT_COLUMNS:
            category = str(value).strip()
            if not category:
                raise ValueError(f"{column} cannot be empty.")
            row[column] = category
        else:
            row[column] = value

    for column in CONTEXT_ONLY_COLUMNS:
        value = values.get(column)
        row[column] = "" if value is None else str(value).strip()

    ordered_columns = [
        column
        for column in TRANSACTION_INPUT_COLUMN_ORDER
        if column in row
    ]
    return pd.DataFrame([row], columns=ordered_columns)


def model_status() -> dict[str, dict[str, object]]:
    return {
        model_name: {
            "model": model_path.is_file()
            and (
                model_name != "autoencoder"
                or AUTOENCODER_PREPROCESSOR_PATH.is_file()
            ),
            "metrics": metrics_path.is_file(),
        }
        for model_name, (model_path, metrics_path) in MODEL_ARTIFACTS.items()
    }


def get_model_threshold(model_name: ModelName) -> float:
    """Load the validation-selected threshold saved with the model metrics."""
    _, metrics_path = MODEL_ARTIFACTS[model_name]
    if not metrics_path.is_file():
        raise ModelNotTrainedError(
            f"Missing {model_name.replace('_', ' ')} metrics artifact: "
            f"{metrics_path}. Train the model first."
        )

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model metrics at {metrics_path}.") from exc

    threshold = payload.get("threshold")
    if threshold is None:
        raise ValueError(
            f"Model metrics at {metrics_path} do not include a threshold."
        )

    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("The saved prediction threshold is not finite.")
    if model_name == "random_forest" and not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "The saved Random Forest probability threshold must be in [0, 1]."
        )
    return threshold


def _align_features(features: pd.DataFrame, model: object) -> pd.DataFrame:
    expected_columns = getattr(model, "feature_names_in_", None)
    if expected_columns is None:
        return features
    aligned = features.copy()
    for column in expected_columns:
        if column not in aligned:
            aligned[column] = np.nan
    return aligned


def _classify_scores(
    scores: np.ndarray,
    threshold: float,
    model_name: ModelName,
) -> np.ndarray:
    if model_name == "autoencoder":
        return (scores > threshold).astype(int)
    return (scores >= threshold).astype(int)


def _autoencoder_score_configuration(
    preprocessor: object,
) -> tuple[str, np.ndarray | None]:
    """Load and validate the saved Autoencoder scoring contract."""
    metrics_path = MODEL_ARTIFACTS["autoencoder"][1]
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read Autoencoder scoring metadata at {metrics_path}."
        ) from exc

    method = payload.get("score_method", "mse")
    if method not in {"mse", "mae", "feature_normalized_mae"}:
        raise ValueError(
            f"Unsupported saved Autoencoder score method: {method!r}."
        )
    if method != "feature_normalized_mae":
        return str(method), None

    saved_names = payload.get("score_feature_names")
    saved_scales = payload.get("score_feature_scales")
    if not isinstance(saved_names, list) or not isinstance(saved_scales, list):
        raise ValueError(
            "Saved Autoencoder metrics are missing feature scoring metadata."
        )
    expected_names = [
        str(name) for name in preprocessor.get_feature_names_out()
    ]
    if saved_names != expected_names:
        raise ValueError(
            "Saved Autoencoder scoring feature order does not match its "
            "preprocessing pipeline."
        )
    scales = np.asarray(saved_scales, dtype=float)
    if scales.shape != (len(expected_names),):
        raise ValueError(
            "Saved Autoencoder feature scales do not match the preprocessor "
            "output width."
        )
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError(
            "Saved Autoencoder feature scales must be finite and positive."
        )
    return str(method), scales


def _score_transactions(
    transactions: pd.DataFrame,
    model_name: ModelName,
) -> tuple[np.ndarray, float, str, str]:
    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(f"Unknown model name: {model_name}")
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise ValueError("Transactions must be a non-empty pandas DataFrame.")
    if not (TRANSACTION_INPUT_COLUMNS & set(transactions.columns)):
        raise ValueError(
            "Input contains no recognized IBM AML transaction columns."
        )

    engineered = engineer_features(transactions)
    model_path, _ = MODEL_ARTIFACTS[model_name]
    if not model_path.is_file():
        if model_name == "autoencoder":
            message = (
                f"Missing autoencoder model artifact: {model_path}. "
                "Run: python main.py train-ae"
            )
        else:
            message = (
                f"Missing model artifact for {model_name.replace('_', ' ')}: "
                f"{model_path}. Train it first."
            )
        raise ModelNotTrainedError(message)

    threshold = get_model_threshold(model_name)

    if model_name == "random_forest":
        pipeline = joblib.load(model_path)
        engineered = _align_features(engineered, pipeline)
        probabilities = pipeline.predict_proba(engineered)
        classes = list(pipeline.classes_)
        if 1 not in classes:
            raise ValueError("Random Forest was not trained with label 1.")
        return (
            probabilities[:, classes.index(1)],
            threshold,
            "probability",
            "random_forest_class_1_probability",
        )

    if model_name == "isolation_forest":
        pipeline = joblib.load(model_path)
        engineered = _align_features(engineered, pipeline)
        scores = -pipeline.decision_function(engineered)
        return (
            np.asarray(scores, dtype=float),
            threshold,
            "anomaly_score",
            "negated_isolation_forest_decision_function",
        )

    if not AUTOENCODER_PREPROCESSOR_PATH.is_file():
        raise ModelNotTrainedError(
            "Missing autoencoder preprocessor artifact: "
            f"{AUTOENCODER_PREPROCESSOR_PATH}. Run: python main.py train-ae"
        )

    import tensorflow as tf

    preprocessor = joblib.load(AUTOENCODER_PREPROCESSOR_PATH)
    score_method, feature_scales = _autoencoder_score_configuration(
        preprocessor
    )
    engineered = _align_features(engineered, preprocessor)
    transformed = preprocessor.transform(engineered)
    batch_size = 2048
    scores: list[np.ndarray] = []
    model = tf.keras.models.load_model(model_path, compile=False)
    for start in range(0, transformed.shape[0], batch_size):
        batch = transformed[start : start + batch_size]
        if hasattr(batch, "toarray"):
            batch = batch.toarray()
        values = np.asarray(batch, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Transactions contain non-finite values.")
        reconstructed = model.predict(values, verbose=0)
        scores.append(reconstruction_scores(
            values,
            reconstructed,
            method=score_method,
            feature_scales=feature_scales,
        ))
    return (
        np.concatenate(scores),
        threshold,
        "reconstruction_error",
        score_method,
    )


def predict_transactions(
    transactions: pd.DataFrame,
    model_name: ModelName,
) -> tuple[np.ndarray, np.ndarray]:
    scores, threshold, _, _ = _score_transactions(transactions, model_name)
    predictions = _classify_scores(scores, threshold, model_name)
    return predictions, np.asarray(scores, dtype=float)


def predict_transaction_report(
    transactions: pd.DataFrame,
    model_name: ModelName,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Return investigation-oriented predictions; alerts are not convictions."""
    scores, saved_threshold, score_name, score_method = _score_transactions(
        transactions,
        model_name,
    )
    effective_threshold = saved_threshold if threshold is None else float(threshold)
    if not np.isfinite(effective_threshold):
        raise ValueError("threshold must be finite.")
    if model_name == "random_forest" and not 0.0 <= effective_threshold <= 1.0:
        raise ValueError("Random Forest threshold must be between 0 and 1.")

    result = transactions.copy().reset_index(drop=True)
    missing_columns = sorted(
        TRANSACTION_INPUT_COLUMNS.difference(transactions.columns)
    )
    unexpected_columns = sorted(
        set(transactions.columns)
        .difference(TRANSACTION_INPUT_COLUMNS)
        .difference({"Is Laundering"})
    )
    warnings = []
    if missing_columns:
        warnings.append(
            "Missing expected columns were imputed: "
            + ", ".join(missing_columns)
        )
    if unexpected_columns:
        warnings.append(
            "Unrecognized columns were ignored: "
            + ", ".join(unexpected_columns)
        )
    result["input_quality_warning"] = "; ".join(warnings) if warnings else ""
    if "Transaction ID" in result.columns:
        result.insert(0, "transaction_id", result.pop("Transaction ID"))
    elif "transaction_id" not in result.columns:
        result.insert(0, "input_row_index", transactions.index.to_numpy())

    predictions = _classify_scores(scores, effective_threshold, model_name)
    result["predicted_class"] = predictions
    result["alert_status"] = np.where(
        predictions == 1,
        "Suspicious transaction detected - investigate",
        "No alert",
    )
    score_margin = max(abs(effective_threshold) * 0.2, np.finfo(float).eps)
    elevated = scores >= effective_threshold - score_margin
    result["suspiciousness_level"] = np.select(
        [predictions == 1, elevated],
        ["High", "Elevated"],
        default="Lower",
    )
    result["model_used"] = model_name
    result["score_method"] = score_method
    result["threshold_used"] = effective_threshold
    result[score_name] = scores
    if model_name == "random_forest":
        result["laundering_probability"] = scores
    return result.sort_values(
        score_name,
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
