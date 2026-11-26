"""Train an autoencoder AML anomaly detector in memory-bounded batches."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    AUTOENCODER_METRICS_PATH,
    AUTOENCODER_MODEL_PATH,
    AUTOENCODER_PREDICTIONS_PATH,
    AUTOENCODER_PREPROCESSOR_PATH,
    PLOTS_DIR,
    SYNTHETIC_PLOTS_DIR,
    TARGET_COLUMN,
    model_artifact_paths,
)
from src.autoencoder_scoring import reconstruction_scores
from src.data_loader import (
    detect_dataset_profile,
    load_training_sample,
    normalize_target_column,
)
from src.evaluation import evaluate_binary_predictions
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor
from src.threshold_optimization import (
    select_normal_validation_threshold,
)
from src.visualization import (
    save_confusion_matrix,
    save_precision_recall_curve,
    save_roc_curve,
    save_threshold_comparison,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an autoencoder AML anomaly detector",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dimension", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument(
        "--reconstruction-loss",
        choices=["mse", "mae", "huber"],
        default="mse",
    )
    parser.add_argument(
        "--score-method",
        choices=["mse", "mae", "feature_normalized_mae"],
        default="feature_normalized_mae",
        help="Anomaly score calculated from reconstruction residuals.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="Maximum rows sampled uniformly from the real dataset.",
    )
    parser.add_argument(
        "--normal-validation-quantile",
        type=float,
        default=0.999,
        help=(
            "Normal-score quantile used for the anomaly cutoff "
            "(default: 0.999, targeting a 0.1%% validation false-positive rate)."
        ),
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_autoencoder(
    input_dimension: int,
    latent_dimension: int | None = None,
    dropout_rate: float = 0.0,
    l2_regularization: float = 0.0,
    reconstruction_loss: str = "mse",
) -> tf.keras.Model:
    """Construct a regularized symmetric dense autoencoder."""
    if input_dimension <= 0:
        raise ValueError("input_dimension must be positive")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1).")
    if l2_regularization < 0:
        raise ValueError("l2_regularization cannot be negative.")
    if reconstruction_loss not in {"mse", "mae", "huber"}:
        raise ValueError("reconstruction_loss must be mse, mae, or huber.")

    if latent_dimension is None:
        latent_dimension = max(8, min(64, input_dimension // 2))
    if not 1 <= latent_dimension <= input_dimension:
        raise ValueError("latent_dimension must be between 1 and input_dimension.")

    regularizer = (
        tf.keras.regularizers.L2(l2_regularization)
        if l2_regularization
        else None
    )
    bottleneck_dimension = latent_dimension
    hidden_dimension = max(bottleneck_dimension * 2, bottleneck_dimension)
    layers: list[tf.keras.layers.Layer] = [
        tf.keras.layers.Input(shape=(input_dimension,)),
        tf.keras.layers.Dense(
            hidden_dimension,
            activation="relu",
            kernel_regularizer=regularizer,
        ),
    ]
    if dropout_rate:
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.extend([
        tf.keras.layers.Dense(bottleneck_dimension, activation="relu"),
        tf.keras.layers.Dense(
            hidden_dimension,
            activation="relu",
            kernel_regularizer=regularizer,
        ),
    ])
    if dropout_rate:
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(input_dimension))
    model = tf.keras.Sequential(layers)
    loss: str | tf.keras.losses.Loss = (
        tf.keras.losses.Huber()
        if reconstruction_loss == "huber"
        else reconstruction_loss
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
    )
    return model


def _dense_batches(
    matrix: Any,
    batch_size: int,
) -> Iterator[np.ndarray]:
    for start in range(0, matrix.shape[0], batch_size):
        batch = matrix[start : start + batch_size]
        if hasattr(batch, "toarray"):
            batch = batch.toarray()
        values = np.asarray(batch, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Transformed data contains non-finite values.")
        yield values


def _dataset(
    matrix: Any,
    batch_size: int,
    shuffle_seed: int | None,
    repeat: bool = False,
):
    dimension = int(matrix.shape[1])
    signature = (
        tf.TensorSpec(shape=(None, dimension), dtype=tf.float32),
        tf.TensorSpec(shape=(None, dimension), dtype=tf.float32),
    )

    generator_pass = 0

    def generate() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        nonlocal generator_pass
        starts = np.arange(
            0,
            matrix.shape[0],
            batch_size,
            dtype=np.int64,
        )
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed + generator_pass)
            rng.shuffle(starts)
            generator_pass += 1
        for start in starts:
            batch = matrix[start : start + batch_size]
            if hasattr(batch, "toarray"):
                batch = batch.toarray()
            batch = np.asarray(batch, dtype=np.float32)
            if not np.isfinite(batch).all():
                raise ValueError("Transformed data contains non-finite values.")
            yield batch, batch

    dataset = tf.data.Dataset.from_generator(
        generate,
        output_signature=signature,
    )
    return dataset.repeat() if repeat else dataset


def _reconstruction_errors(
    model: tf.keras.Model,
    matrix: Any,
    batch_size: int,
    score_method: str = "mse",
    feature_scales: np.ndarray | None = None,
) -> np.ndarray:
    errors = []
    for batch in _dense_batches(matrix, batch_size):
        reconstructed = model.predict(batch, verbose=0)
        errors.append(reconstruction_scores(
            batch,
            reconstructed,
            method=score_method,
            feature_scales=feature_scales,
        ))
    if not errors:
        raise ValueError("Cannot calculate errors for an empty dataset.")
    return np.concatenate(errors)


def _normal_validation_feature_scales(
    model: tf.keras.Model,
    matrix: Any,
    normal_mask: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Estimate per-feature MAE scales from normal validation rows only."""
    feature_count = int(matrix.shape[1])
    absolute_error_sum = np.zeros(feature_count, dtype=np.float64)
    normal_count = 0
    start = 0
    for batch in _dense_batches(matrix, batch_size):
        batch_mask = normal_mask[start : start + len(batch)]
        if batch_mask.any():
            reconstructed = model.predict(batch, verbose=0)
            absolute_error_sum += np.abs(
                batch[batch_mask] - reconstructed[batch_mask]
            ).sum(axis=0)
            normal_count += int(batch_mask.sum())
        start += len(batch)

    if normal_count == 0:
        raise ValueError(
            "Normal validation rows are required to estimate feature scales."
        )
    mean_absolute_errors = absolute_error_sum / normal_count
    floor = max(float(np.median(mean_absolute_errors)) * 0.1, 1e-8)
    return np.maximum(mean_absolute_errors, floor)


def _validation_threshold_table(
    labels: np.ndarray,
    scores: np.ndarray,
    selected_quantile: float,
) -> pd.DataFrame:
    """Compare normal-only calibration quantiles without test-set thresholds."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    normal_scores = scores[labels == 0]
    if normal_scores.size == 0:
        raise ValueError("Normal validation scores are required.")

    quantiles = sorted({0.98, 0.99, 0.995, 0.999, selected_quantile})
    rows = []
    for quantile in quantiles:
        threshold = float(np.quantile(normal_scores, quantile))
        predictions = (scores > threshold).astype(int)
        metrics = evaluate_binary_predictions(labels, predictions, scores)
        rows.append({
            "normal_validation_quantile": quantile,
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1_score"],
            "f2": metrics["f2_score"],
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
            "alerts": metrics["total_alerts"],
            "tp": metrics["true_positives"],
            "fp": metrics["false_positives"],
            "tn": metrics["true_negatives"],
            "fn": metrics["false_negatives"],
            "average_precision": metrics.get("average_precision"),
            "roc_auc": metrics.get("roc_auc"),
        })
    return pd.DataFrame(rows)


def _score_distribution(scores: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=float)
    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "standard_deviation": float(np.std(scores)),
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
        "p90": float(np.quantile(scores, 0.90)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "p99_5": float(np.quantile(scores, 0.995)),
    }


def train_autoencoder(
    transactions: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
    epochs: int = 50,
    batch_size: int = 256,
    normal_validation_quantile: float = 0.999,
    latent_dimension: int | None = 8,
    dropout_rate: float = 0.0,
    l2_regularization: float = 0.0,
    reconstruction_loss: str = "mse",
    score_method: str = "feature_normalized_mae",
) -> dict[str, Any]:
    """Fit on normal rows and calibrate the cutoff on normal validation rows."""
    profile = detect_dataset_profile(transactions.columns)
    transactions = normalize_target_column(transactions)
    if profile not in {"ibm", "synthetic"}:
        raise ValueError(
            "Training data columns do not match a supported IBM HI-Small or "
            "synthetic AML schema."
        )
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive.")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size plus validation_size must be less than 1.")
    if not 0.0 < normal_validation_quantile < 1.0:
        raise ValueError(
            "normal_validation_quantile must be between 0 and 1."
        )
    if score_method not in {"mse", "mae", "feature_normalized_mae"}:
        raise ValueError(
            "score_method must be mse, mae, or feature_normalized_mae."
        )

    target = transactions[TARGET_COLUMN].astype(int)
    if set(target.unique()) != {0, 1} or int(target.value_counts().min()) < 3:
        raise ValueError(
            "Training requires at least three examples of each target class."
        )
    tf.keras.utils.set_random_seed(random_state)

    development, test_df = train_test_split(
        transactions,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    train_df, validation_df = train_test_split(
        development,
        test_size=validation_size / (1.0 - test_size),
        random_state=random_state,
        stratify=development[TARGET_COLUMN],
    )

    x_train = engineer_features(train_df.drop(columns=[TARGET_COLUMN]))
    x_validation = engineer_features(
        validation_df.drop(columns=[TARGET_COLUMN])
    )
    x_test = engineer_features(test_df.drop(columns=[TARGET_COLUMN]))
    y_train = train_df[TARGET_COLUMN].astype(int)
    y_validation = validation_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)
    normal_training = y_train == 0

    preprocessor, _, _ = build_preprocessor(
        train_df.loc[normal_training]
    )
    preprocessing_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    x_train_normal = preprocessing_pipeline.fit_transform(
        x_train.loc[normal_training]
    )
    x_validation_transformed = preprocessing_pipeline.transform(x_validation)
    x_test_transformed = preprocessing_pipeline.transform(x_test)

    train_dataset = _dataset(
        x_train_normal,
        batch_size,
        shuffle_seed=random_state,
        repeat=True,
    )
    validation_normal = y_validation.to_numpy() == 0
    validation_dataset = _dataset(
        x_validation_transformed[validation_normal],
        batch_size,
        shuffle_seed=None,
    )
    steps = math.ceil(x_train_normal.shape[0] / batch_size)
    validation_steps = math.ceil(
        int(validation_normal.sum()) / batch_size
    )

    effective_latent_dimension = (
        max(8, min(64, int(x_train_normal.shape[1]) // 2))
        if latent_dimension is None
        else latent_dimension
    )
    model = build_autoencoder(
        int(x_train_normal.shape[1]),
        latent_dimension=effective_latent_dimension,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization,
        reconstruction_loss=reconstruction_loss,
    )
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )

    validation_labels = y_validation.to_numpy()
    normal_validation = validation_labels == 0
    feature_scales = None
    score_feature_names: list[str] = []
    if score_method == "feature_normalized_mae":
        feature_scales = _normal_validation_feature_scales(
            model,
            x_validation_transformed,
            normal_validation,
            batch_size,
        )
        score_feature_names = [
            str(name)
            for name in preprocessing_pipeline.get_feature_names_out()
        ]
    validation_errors = _reconstruction_errors(
        model,
        x_validation_transformed,
        batch_size,
        score_method=score_method,
        feature_scales=feature_scales,
    )
    test_errors = _reconstruction_errors(
        model,
        x_test_transformed,
        batch_size,
        score_method=score_method,
        feature_scales=feature_scales,
    )
    test_mse_errors = (
        test_errors
        if score_method == "mse"
        else _reconstruction_errors(
            model,
            x_test_transformed,
            batch_size,
            score_method="mse",
        )
    )
    threshold = select_normal_validation_threshold(
        validation_labels,
        validation_errors,
        quantile=normal_validation_quantile,
    )
    validation_predictions = (validation_errors > threshold).astype(int)
    predictions = (test_errors > threshold).astype(int)
    candidate_table = _validation_threshold_table(
        validation_labels,
        validation_errors,
        normal_validation_quantile,
    )

    metrics = evaluate_binary_predictions(y_test, predictions, test_errors)
    trained_at = datetime.now(timezone.utc)
    input_features = [
        str(name) for name in preprocessor.feature_names_in_
    ]
    transformed_features = [
        str(name) for name in preprocessor.get_feature_names_out()
    ]
    metrics.update({
        "dataset_profile": profile,
        "artifact_version": 1,
        "model_version": (
            f"{profile}-{trained_at.strftime('%Y%m%dT%H%M%S%fZ')}"
        ),
        "trained_at_utc": trained_at.isoformat(),
        "feature_schema": {
            "profile": profile,
            "input_features": input_features,
            "transformed_features": transformed_features,
        },
        "threshold": float(threshold),
        "threshold_strategy": "normal_validation_quantile",
        "threshold_quantile": float(normal_validation_quantile),
        "target_validation_false_positive_rate": float(
            1.0 - normal_validation_quantile
        ),
        "normal_validation_rows": int(normal_validation.sum()),
        "validation_normal_false_positive_rate": float(
            validation_predictions[normal_validation].mean()
        ),
        "score_direction": "higher reconstruction errors indicate more anomalous",
        "score_method": score_method,
        "score_feature_names": score_feature_names,
        "score_feature_scales": (
            feature_scales.tolist() if feature_scales is not None else []
        ),
        "score_feature_scale_method": (
            "mean absolute residual on normal validation rows with "
            "a 10%-of-median floor"
            if feature_scales is not None
            else None
        ),
        "validation_metrics": evaluate_binary_predictions(
            y_validation,
            validation_predictions,
            validation_errors,
        ),
        "validation_threshold_comparison": candidate_table.to_dict(
            orient="records"
        ),
        "test_score_distribution": {
            "normal": _score_distribution(test_errors[y_test.to_numpy() == 0]),
            "laundering": _score_distribution(
                test_errors[y_test.to_numpy() == 1]
            ),
        },
        "test_raw_mse_distribution": {
            "normal": _score_distribution(
                test_mse_errors[y_test.to_numpy() == 0]
            ),
            "laundering": _score_distribution(
                test_mse_errors[y_test.to_numpy() == 1]
            ),
        },
        "training_rows": int(normal_training.sum()),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "random_state": int(random_state),
        "epochs_requested": int(epochs),
        "epochs_trained": len(history.epoch),
        "batch_size": int(batch_size),
        "latent_dimension": int(effective_latent_dimension),
        "hidden_dimension": int(max(effective_latent_dimension * 2, effective_latent_dimension)),
        "dropout_rate": float(dropout_rate),
        "l2_regularization": float(l2_regularization),
        "reconstruction_loss": reconstruction_loss,
    })

    if profile == "ibm":
        model_path = AUTOENCODER_MODEL_PATH
        metrics_path = AUTOENCODER_METRICS_PATH
        predictions_path = AUTOENCODER_PREDICTIONS_PATH
        preprocessor_path = AUTOENCODER_PREPROCESSOR_PATH
        plots_dir = PLOTS_DIR
    else:
        model_path, metrics_path, predictions_path, preprocessor_path = (
            model_artifact_paths("autoencoder", profile)
        )
        if preprocessor_path is None:
            raise ValueError("Autoencoder profile has no preprocessor path.")
        plots_dir = SYNTHETIC_PLOTS_DIR
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(preprocessing_pipeline, preprocessor_path)
    save_json(metrics_path, metrics)
    candidate_table.to_csv(
        metrics_path.with_name("autoencoder_thresholds.csv"),
        index=False,
    )
    save_threshold_comparison(
        candidate_table,
        plots_dir / "autoencoder_threshold_comparison.png",
        "Autoencoder Validation Metrics with Normal-Quantile Calibration",
    )

    pd.DataFrame({
        "sample_row_index": test_df.index.to_numpy(),
        "actual": y_test.to_numpy(dtype=int),
        "prediction": predictions,
        "reconstruction_error": test_errors,
        "raw_reconstruction_mse": test_mse_errors,
    }).to_csv(predictions_path, index=False)
    save_confusion_matrix(
        y_test,
        predictions,
        plots_dir / "autoencoder_confusion_matrix.png",
        "Autoencoder Confusion Matrix",
    )
    save_precision_recall_curve(
        y_test,
        test_errors,
        plots_dir / "autoencoder_precision_recall.png",
        "Autoencoder Precision-Recall Curve",
    )
    save_roc_curve(
        y_test,
        test_errors,
        plots_dir / "autoencoder_roc.png",
        "Autoencoder ROC Curve",
    )
    return metrics


def main() -> None:
    arguments = parse_arguments()
    transactions = load_training_sample(
        arguments.dataset,
        max_rows=arguments.max_rows,
        random_seed=arguments.random_state,
    )
    metrics = train_autoencoder(
        transactions,
        test_size=arguments.test_size,
        validation_size=arguments.validation_size,
        random_state=arguments.random_state,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        normal_validation_quantile=arguments.normal_validation_quantile,
        latent_dimension=arguments.latent_dimension,
        dropout_rate=arguments.dropout_rate,
        l2_regularization=arguments.l2_regularization,
        reconstruction_loss=arguments.reconstruction_loss,
        score_method=arguments.score_method,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
