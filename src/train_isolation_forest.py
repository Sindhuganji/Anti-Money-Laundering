"""Train and evaluate an Isolation Forest AML anomaly detector."""

from __future__ import annotations

import argparse
import json
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
    METRICS_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    TARGET_COLUMN,
)
from src.evaluation import evaluate_binary_predictions
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor
from src.threshold_optimization import select_threshold
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
) -> dict[str, Any]:
    """Train, evaluate, and persist an Isolation Forest model."""
    if TARGET_COLUMN not in transactions.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    if len(transactions) < 2:
        raise ValueError("At least two transactions are required")

    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size plus validation_size must be less than 1.")
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
    threshold, candidate_table = select_threshold(
        y_validation.to_numpy(),
        validation_scores,
        strategy="f2",
    )
    anomaly_scores = -pipeline.decision_function(x_test)
    predictions = (anomaly_scores >= threshold).astype(int)

    metrics = evaluate_binary_predictions(
        y_test,
        predictions,
        anomaly_scores,
    )
    metrics["threshold"] = float(threshold)
    metrics["threshold_strategy"] = "f2"
    metrics["score_direction"] = "higher scores indicate more anomalous"
    metrics["contamination"] = contamination
    metrics["validation_metrics"] = evaluate_binary_predictions(
        y_validation,
        (validation_scores >= threshold).astype(int),
        validation_scores,
    )
    metrics["training_rows"] = int(normal_training.sum())
    metrics["validation_rows"] = int(len(validation_df))
    metrics["test_rows"] = int(len(test_df))
    metrics["random_state"] = int(random_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, ISOLATION_FOREST_MODEL_PATH)

    save_json(ISOLATION_FOREST_METRICS_PATH, metrics)
    candidate_table.to_csv(
        METRICS_DIR / "isolation_forest_thresholds.csv",
        index=False,
    )
    save_threshold_comparison(
        candidate_table,
        PLOTS_DIR / "isolation_forest_threshold_comparison.png",
        "Isolation Forest Validation Threshold Comparison",
    )

    prediction_frame = pd.DataFrame({
        "sample_row_index": test_df.index.to_numpy(),
        "actual": y_test.to_numpy(dtype=int),
        "prediction": predictions,
        "anomaly_score": anomaly_scores,
    })
    prediction_frame.to_csv(
        ISOLATION_FOREST_PREDICTIONS_PATH,
        index=False,
    )
    save_confusion_matrix(
        y_test,
        predictions,
        PLOTS_DIR / "isolation_forest_confusion_matrix.png",
        "Isolation Forest Confusion Matrix",
    )
    save_precision_recall_curve(
        y_test,
        anomaly_scores,
        PLOTS_DIR / "isolation_forest_precision_recall.png",
        "Isolation Forest Precision-Recall Curve",
    )
    save_roc_curve(
        y_test,
        anomaly_scores,
        PLOTS_DIR / "isolation_forest_roc.png",
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
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()