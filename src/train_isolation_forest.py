"""Train and evaluate an Isolation Forest AML anomaly detector."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ISOLATION_FOREST_METRICS_PATH,
    ISOLATION_FOREST_MODEL_PATH,
    ISOLATION_FOREST_PREDICTIONS_PATH,
    PLOTS_DIR,
    SYNTHETIC_PLOTS_DIR,
    TARGET_COLUMN,
    model_artifact_paths,
)
from src.data_loader import detect_dataset_profile, normalize_target_column
from src.evaluation import evaluate_binary_predictions
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor
from src.threshold_optimization import (
    normal_validation_threshold_table,
    select_normal_validation_threshold,
)
from src.visualization import (
    save_confusion_matrix,
    save_precision_recall_curve,
    save_roc_curve,
    save_threshold_comparison,
)


def parse_contamination(value: str) -> str | float:
    if value.lower() == "auto":
        return "auto"
    try:
        contamination = float(value)
    except ValueError as exc:
        raise ValueError("contamination must be 'auto' or a float in (0, 0.5].") from exc
    if not 0.0 < contamination <= 0.5:
        raise ValueError("contamination must be 'auto' or a float in (0, 0.5].")
    return contamination


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Isolation Forest AML detector",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the IBM AML CSV dataset",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of records reserved for final testing",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Fraction of records reserved for threshold selection",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--contamination",
        type=parse_contamination,
        default="auto",
        help="Expected proportion of anomalies",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=300,
        help="Number of isolation trees",
    )
    parser.add_argument(
        "--normal-validation-quantile",
        type=float,
        default=0.995,
        help="Anomaly cutoff quantile from legitimate validation scores.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="Maximum rows sampled uniformly from the real dataset.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def train_isolation_forest(
    transactions: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
    contamination: str | float = "auto",
    estimators: int = 300,
    normal_validation_quantile: float = 0.995,
) -> dict[str, Any]:
    """Train, evaluate, and persist an Isolation Forest model."""
    profile = detect_dataset_profile(transactions.columns)
    transactions = normalize_target_column(transactions)
    if profile not in {"ibm", "synthetic"}:
        raise ValueError(
            "Training data columns do not match a supported IBM HI-Small or "
            "synthetic AML schema."
        )
    if len(transactions) < 2:
        raise ValueError("At least two transactions are required")

    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size plus validation_size must be less than 1.")
    if not 0.0 < normal_validation_quantile < 1.0:
        raise ValueError(
            "normal_validation_quantile must be between 0 and 1."
        )
    if contamination != "auto":
        contamination = parse_contamination(str(contamination))

    target = transactions[TARGET_COLUMN].astype(int)
    if set(target.unique()) != {0, 1} or int(target.value_counts().min()) < 3:
        raise ValueError(
            "Training requires at least three examples of each target class."
        )

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

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("detector", IsolationForest(
            n_estimators=estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(x_train.loc[normal_training])

    validation_scores = -pipeline.decision_function(x_validation)
    threshold = select_normal_validation_threshold(
        y_validation.to_numpy(),
        validation_scores,
        quantile=normal_validation_quantile,
    )
    candidate_table = normal_validation_threshold_table(
        y_validation.to_numpy(),
        validation_scores,
        quantiles=tuple(sorted({
            0.98,
            0.99,
            0.995,
            0.999,
            normal_validation_quantile,
        })),
    )
    anomaly_scores = -pipeline.decision_function(x_test)
    predictions = (anomaly_scores >= threshold).astype(int)

    metrics = evaluate_binary_predictions(
        y_test,
        predictions,
        anomaly_scores,
    )
    trained_at = datetime.now(timezone.utc)
    metrics["threshold"] = float(threshold)
    metrics["threshold_strategy"] = "normal_validation_quantile"
    metrics["artifact_version"] = 1
    metrics["model_version"] = (
        f"{profile}-{trained_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    metrics["trained_at_utc"] = trained_at.isoformat()
    input_features = [str(name) for name in preprocessor.feature_names_in_]
    transformed_features = [
        str(name) for name in preprocessor.get_feature_names_out()
    ]
    metrics["feature_schema"] = {
        "profile": profile,
        "input_features": input_features,
        "transformed_features": transformed_features,
    }
    metrics["threshold_quantile"] = float(normal_validation_quantile)
    metrics["target_validation_false_positive_rate"] = float(
        1.0 - normal_validation_quantile
    )
    metrics["score_direction"] = "higher scores indicate more anomalous"
    metrics["score_method"] = "negated_decision_function"
    metrics["dataset_profile"] = profile
    metrics["contamination"] = contamination
    metrics["validation_metrics"] = evaluate_binary_predictions(
        y_validation,
        (validation_scores >= threshold).astype(int),
        validation_scores,
    )
    metrics["validation_normal_false_positive_rate"] = float(
        (
            validation_scores[y_validation.to_numpy() == 0] >= threshold
        ).mean()
    )
    metrics["training_rows"] = int(normal_training.sum())
    metrics["validation_rows"] = int(len(validation_df))
    metrics["test_rows"] = int(len(test_df))
    metrics["random_state"] = int(random_state)

    if profile == "ibm":
        model_path = ISOLATION_FOREST_MODEL_PATH
        metrics_path = ISOLATION_FOREST_METRICS_PATH
        predictions_path = ISOLATION_FOREST_PREDICTIONS_PATH
        plots_dir = PLOTS_DIR
    else:
        model_path, metrics_path, predictions_path, _ = model_artifact_paths(
            "isolation_forest",
            profile,
        )
        plots_dir = SYNTHETIC_PLOTS_DIR
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    save_json(metrics_path, metrics)
    candidate_table.to_csv(
        metrics_path.with_name("isolation_forest_thresholds.csv"),
        index=False,
    )
    save_threshold_comparison(
        candidate_table,
        plots_dir / "isolation_forest_threshold_comparison.png",
        "Isolation Forest Validation Threshold Comparison",
    )

    prediction_frame = pd.DataFrame({
        "sample_row_index": test_df.index.to_numpy(),
        "actual": y_test.to_numpy(dtype=int),
        "prediction": predictions,
        "anomaly_score": anomaly_scores,
    })
    prediction_frame.to_csv(
        predictions_path,
        index=False,
    )
    save_confusion_matrix(
        y_test,
        predictions,
        plots_dir / "isolation_forest_confusion_matrix.png",
        "Isolation Forest Confusion Matrix",
    )
    save_precision_recall_curve(
        y_test,
        anomaly_scores,
        plots_dir / "isolation_forest_precision_recall.png",
        "Isolation Forest Precision-Recall Curve",
    )
    save_roc_curve(
        y_test,
        anomaly_scores,
        plots_dir / "isolation_forest_roc.png",
        "Isolation Forest ROC Curve",
    )

    return metrics


def main() -> None:
    arguments = parse_arguments()
    from src.data_loader import load_training_sample

    transactions = load_training_sample(
        arguments.dataset,
        max_rows=arguments.max_rows,
        random_seed=arguments.random_state,
    )
    metrics = train_isolation_forest(
        transactions,
        test_size=arguments.test_size,
        validation_size=arguments.validation_size,
        random_state=arguments.random_state,
        contamination=arguments.contamination,
        estimators=arguments.estimators,
        normal_validation_quantile=arguments.normal_validation_quantile,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()