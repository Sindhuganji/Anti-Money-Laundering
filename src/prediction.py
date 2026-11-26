"""Unified transaction prediction interface."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN
from src.autoencoder_scoring import reconstruction_scores
from src.data_loader import normalize_target_column
from src.feature_engineering import engineer_features
from src.model_registry import (
    DEFAULT_MODEL_PROFILE,
    MODEL_REGISTRY,
    ModelName,
    get_model_artifacts,
    resolve_model_profile,
)

logger = logging.getLogger(__name__)

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
    "Amount_Log": ("Amount",),
}
NUMERIC_INPUT_COLUMNS = {
    "From Bank",
    "To Bank",
    "Amount Received",
    "Amount Paid",
}
SYNTHETIC_NUMERIC_INPUT_COLUMNS = {
    "Sender_Bank",
    "Receiver_Bank",
    "Amount",
    "Hour",
    "Day_Of_Week",
}
CATEGORICAL_INPUT_COLUMNS = {
    "Receiving Currency",
    "Payment Currency",
    "Payment Format",
}
CONTEXT_ONLY_COLUMNS = ("Account", "Account.1")
SYNTHETIC_INPUT_COLUMNS = {
    "Timestamp",
    "Sender_ID",
    "Receiver_ID",
    "Sender_Bank",
    "Receiver_Bank",
    "Transaction_Type",
    "Amount",
    "Currency",
    "Payment_Format",
    "Cross_Border",
    "Hour",
    "Day_Of_Week",
}
SYNTHETIC_INPUT_COLUMN_ORDER = (
    "Timestamp",
    "Sender_ID",
    "Receiver_ID",
    "Sender_Bank",
    "Receiver_Bank",
    "Transaction_Type",
    "Amount",
    "Currency",
    "Payment_Format",
    "Cross_Border",
    "Hour",
    "Day_Of_Week",
)
GROUND_TRUTH_COLUMNS = ("Is Laundering", "Is_Laundering")


@dataclass(frozen=True)
class ModelInputSchema:
    """Raw transaction fields required by a saved model preprocessor."""

    required_columns: tuple[str, ...]
    categorical_options: dict[str, tuple[str, ...]]
    profile: str = "ibm"
    numeric_columns: tuple[str, ...] = ()


class ModelNotTrainedError(FileNotFoundError):
    """Raised when a requested trained model artifact is missing."""


class ModelCompatibilityError(ValueError):
    """Raised when input fields do not match a model's training profile."""


def _validate_artifact_profile(metrics_path: Path, profile: str) -> None:
    if not metrics_path.is_file():
        raise ModelNotTrainedError(
            f"Missing {profile} model metadata: {metrics_path}."
        )
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read model compatibility metadata at {metrics_path}."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Model metadata at {metrics_path} must be an object.")
    saved_profile = payload.get("dataset_profile", "ibm")
    if saved_profile != profile:
        raise ModelCompatibilityError(
            f"Input profile is {profile}, but artifact metadata identifies "
            f"{saved_profile!r}; refusing to score with incompatible features."
        )


def _artifact_paths(
    model_name: ModelName,
    profile: str,
) -> tuple[Path, Path, Path, Path | None]:
    return get_model_artifacts(model_name, profile)


def model_profile_for_transactions(transactions: pd.DataFrame) -> str:
    """Infer a saved artifact profile from input columns only."""
    try:
        return resolve_model_profile(transactions)
    except ValueError as error:
        raise ModelCompatibilityError(str(error)) from error


def get_model_input_schema(
    model_name: ModelName,
    transactions: pd.DataFrame | None = None,
) -> ModelInputSchema:
    """Discover fields from the selected model's inferred-profile artifacts."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    profile = resolve_model_profile(transactions)

    model_path, metrics_path, _, preprocessor_path = _artifact_paths(
        model_name,
        profile,
    )
    if not model_path.is_file():
        raise ModelNotTrainedError(
            f"Missing model artifact for {model_name.replace('_', ' ')}: "
            f"{model_path}. Train the {profile} profile first."
        )
    _validate_artifact_profile(metrics_path, profile)

    if model_name == "autoencoder":
        if preprocessor_path is None or not preprocessor_path.is_file():
            raise ModelNotTrainedError(
                f"Missing {profile} Autoencoder preprocessor artifact: "
                f"{preprocessor_path}. Train the {profile} profile first."
            )
        preprocessor_pipeline = joblib.load(preprocessor_path)
        preprocessor = preprocessor_pipeline.named_steps["preprocessor"]
    else:
        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps["preprocessor"]

    selected_features: set[str] = set()
    categorical_columns: list[str] = []
    numeric_features: list[str] = []
    for name, _, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        selected_features.update(str(column) for column in columns)
        if name == "categorical":
            categorical_columns = [str(column) for column in columns]
        elif name == "numeric":
            numeric_features = [str(column) for column in columns]

    raw_columns: set[str] = set()
    for feature in selected_features:
        sources = (
            (feature,)
            if profile == "synthetic" and feature == "Hour"
            else ENGINEERED_FEATURE_SOURCES.get(feature, (feature,))
        )
        allowed_columns = (
            SYNTHETIC_INPUT_COLUMNS
            if profile == "synthetic"
            else TRANSACTION_INPUT_COLUMNS
        )
        unsupported = set(sources).difference(allowed_columns)
        if unsupported:
            raise ValueError(
                f"Model feature {feature!r} maps to unsupported raw columns: "
                f"{sorted(unsupported)}"
            )
        raw_columns.update(sources)

    numeric_raw_columns: set[str] = set()
    for feature in numeric_features:
        sources = (
            (feature,)
            if profile == "synthetic" and feature == "Hour"
            else ENGINEERED_FEATURE_SOURCES.get(feature, (feature,))
        )
        allowed_numeric = (
            SYNTHETIC_NUMERIC_INPUT_COLUMNS
            if profile == "synthetic"
            else NUMERIC_INPUT_COLUMNS
        )
        numeric_raw_columns.update(set(sources) & allowed_numeric)

    required_columns = tuple(
        column
        for column in (
            TRANSACTION_INPUT_COLUMN_ORDER
            if profile == "ibm"
            else SYNTHETIC_INPUT_COLUMN_ORDER
        )
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
        profile=profile,
        numeric_columns=tuple(
            column for column in required_columns
            if column in numeric_raw_columns
        ),
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
        elif column in schema.numeric_columns or column in NUMERIC_INPUT_COLUMNS:
            if column == "Cross_Border":
                normalized = str(value).strip().casefold()
                if normalized in {"true", "1", "yes"}:
                    row[column] = True
                    continue
                if normalized in {"false", "0", "no"}:
                    row[column] = False
                    continue
                raise ValueError(f"{column} must be a boolean value.")
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
        for column in (*schema.required_columns, *CONTEXT_ONLY_COLUMNS)
        if column in row
    ]
    return pd.DataFrame([row], columns=ordered_columns)


def model_status(
    transactions: pd.DataFrame | None = None,
) -> dict[str, dict[str, object]]:
    profile = resolve_model_profile(transactions)
    artifacts = {
        model_name: _artifact_paths(model_name, profile)
        for model_name in MODEL_REGISTRY
    }
    return {
        model_name: {
            "model": model_path.is_file()
            and (
                model_name != "autoencoder"
                or (
                    preprocessor_path is not None
                    and preprocessor_path.is_file()
                )
            ),
            "metrics": metrics_path.is_file(),
        }
        for model_name, (
            model_path,
            metrics_path,
            _,
            preprocessor_path,
        ) in artifacts.items()
    }


def get_model_metadata(
    model_name: ModelName,
    transactions: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Return metadata for the default or input-inferred saved model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    profile = resolve_model_profile(transactions)
    _, metrics_path, _, _ = _artifact_paths(model_name, profile)
    state = model_status(transactions)[model_name]
    payload: dict[str, object] = {}
    if metrics_path.is_file():
        _validate_artifact_profile(metrics_path, profile)
        loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Model metadata at {metrics_path} must be an object."
            )
        payload = loaded
    threshold = (
        get_model_threshold(model_name, transactions)
        if metrics_path.is_file()
        else None
    )

    return {
        "profile": profile,
        "algorithm": model_name,
        "status": (
            "Ready"
            if state["model"] and state["metrics"]
            else "Not trained"
        ),
        "version": str(payload.get("model_version", "legacy-unversioned")),
        "threshold": threshold,
        "threshold_strategy": payload.get("threshold_strategy"),
        "feature_schema": payload.get("feature_schema"),
    }


def get_model_threshold(
    model_name: ModelName,
    transactions: pd.DataFrame | None = None,
) -> float:
    """Load the validation-selected threshold saved with the model metrics."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    profile = resolve_model_profile(transactions)
    _, metrics_path, _, _ = _artifact_paths(model_name, profile)
    if not metrics_path.is_file():
        raise ModelNotTrainedError(
            f"Missing {model_name.replace('_', ' ')} metrics artifact: "
            f"{metrics_path}. Train the model first."
        )
    _validate_artifact_profile(metrics_path, profile)

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
    return aligned.loc[:, list(expected_columns)]


def _classify_scores(
    scores: np.ndarray,
    threshold: float,
    model_name: ModelName,
) -> np.ndarray:
    if model_name == "autoencoder":
        return (scores > threshold).astype(int)
    return (scores >= threshold).astype(int)


def _validated_scores(scores: np.ndarray, row_count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.shape != (row_count,):
        raise ValueError(
            "The model returned a score count that does not match the "
            "transaction count."
        )
    if not np.isfinite(values).all():
        raise ValueError("The model returned non-finite transaction scores.")
    return values


def _autoencoder_score_configuration(
    preprocessor: object,
    metrics_path: Path | None = None,
) -> tuple[str, np.ndarray | None]:
    """Load and validate the saved Autoencoder scoring contract."""
    metrics_path = (
        metrics_path
        or get_model_artifacts("autoencoder", DEFAULT_MODEL_PROFILE).metrics
    )
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
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise ValueError("Transactions must be a non-empty pandas DataFrame.")
    profile = model_profile_for_transactions(transactions)
    logger.info(
        "Scoring %d transaction(s) with %s model profile",
        len(transactions),
        profile,
    )
    recognized_columns = (
        SYNTHETIC_INPUT_COLUMNS
        if profile == "synthetic"
        else TRANSACTION_INPUT_COLUMNS
    )
    if not (recognized_columns & set(transactions.columns)):
        raise ModelCompatibilityError(
            f"Input contains no recognized {profile} AML transaction columns."
        )

    model_input = transactions.drop(
        columns=GROUND_TRUTH_COLUMNS,
        errors="ignore",
    )
    engineered = engineer_features(model_input)
    model_path, metrics_path, _, preprocessor_path = _artifact_paths(
        model_name,
        profile,
    )
    if not model_path.is_file():
        if profile == "synthetic":
            message = (
                f"No separately trained synthetic AML {model_name.replace('_', ' ')} "
                f"artifact exists at {model_path}. Train an isolated model "
                "using the synthetic CSV: python main.py train-all --dataset "
                "<synthetic-csv>. IBM HI-Small models "
                "are not compatible and were not used."
            )
        elif model_name == "autoencoder":
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

    threshold = get_model_threshold(model_name, transactions)
    schema = get_model_input_schema(model_name, transactions)
    missing_features = sorted(
        set(schema.required_columns).difference(model_input.columns)
    )
    if missing_features:
        raise ModelCompatibilityError(
            f"CSV is missing features required by the saved {profile} "
            f"{model_name.replace('_', ' ')} model: "
            f"{', '.join(missing_features)}."
        )

    if model_name == "random_forest":
        pipeline = joblib.load(model_path)
        engineered = _align_features(engineered, pipeline)
        probabilities = pipeline.predict_proba(engineered)
        classes = list(pipeline.classes_)
        if 1 not in classes:
            raise ValueError("Random Forest was not trained with label 1.")
        if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
            raise ValueError(
                "Random Forest probability columns do not match its saved "
                "class labels."
            )
        positive_scores = _validated_scores(
            probabilities[:, classes.index(1)],
            len(transactions),
        )
        return (
            positive_scores,
            threshold,
            "probability",
            "random_forest_class_1_probability",
        )

    if model_name == "isolation_forest":
        pipeline = joblib.load(model_path)
        engineered = _align_features(engineered, pipeline)
        scores = _validated_scores(
            -pipeline.decision_function(engineered),
            len(transactions),
        )
        return (
            np.asarray(scores, dtype=float),
            threshold,
            "anomaly_score",
            "negated_isolation_forest_decision_function",
        )

    if preprocessor_path is None or not preprocessor_path.is_file():
        raise ModelNotTrainedError(
            f"Missing {profile} Autoencoder preprocessor artifact: "
            f"{preprocessor_path}. Train the {profile} profile first."
        )

    import tensorflow as tf

    preprocessor = joblib.load(preprocessor_path)
    score_method, feature_scales = _autoencoder_score_configuration(
        preprocessor,
        metrics_path=metrics_path,
    )
    engineered = _align_features(engineered, preprocessor)
    transformed = preprocessor.transform(engineered)
    batch_size = 2048
    scores: list[np.ndarray] = []
    model = tf.keras.models.load_model(model_path, compile=False)
    expected_dimension = model.input_shape[-1]
    if expected_dimension != transformed.shape[1]:
        raise ModelCompatibilityError(
            "Autoencoder input dimension does not match the saved "
            "preprocessing output: "
            f"model expects {expected_dimension}, preprocessor produced "
            f"{transformed.shape[1]}."
        )
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
    profile = model_profile_for_transactions(transactions)
    effective_threshold = saved_threshold if threshold is None else float(threshold)
    if not np.isfinite(effective_threshold):
        raise ValueError("threshold must be finite.")
    if model_name == "random_forest" and not 0.0 <= effective_threshold <= 1.0:
        raise ValueError("Random Forest threshold must be between 0 and 1.")

    result = transactions.copy().reset_index(drop=True)
    target_column = next(
        (column for column in GROUND_TRUTH_COLUMNS if column in transactions),
        None,
    )
    if target_column is not None:
        labels = normalize_target_column(transactions[[target_column]])
        result["ground_truth_label"] = labels[TARGET_COLUMN].to_numpy(dtype=int)
    missing_columns = sorted(
        (
            SYNTHETIC_INPUT_COLUMNS
            if profile == "synthetic"
            else TRANSACTION_INPUT_COLUMNS
        ).difference(transactions.columns)
    )
    unexpected_columns = sorted(
        set(transactions.columns)
        .difference(
            SYNTHETIC_INPUT_COLUMNS
            if profile == "synthetic"
            else TRANSACTION_INPUT_COLUMNS
        )
        .difference(GROUND_TRUTH_COLUMNS)
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
    logger.info(
        "%s/%s scored %d transaction(s); %d alert(s) at threshold %.8g",
        profile,
        model_name,
        len(result),
        int(predictions.sum()),
        effective_threshold,
    )
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
    result["model_profile"] = profile
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
