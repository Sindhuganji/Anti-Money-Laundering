"""Evaluation utilities for AML binary classifiers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Calculate classification and ranking metrics for a binary target."""
    raw_true_labels = np.asarray(y_true).ravel()
    raw_predicted_labels = np.asarray(y_pred).ravel()

    if raw_true_labels.shape[0] == 0:
        raise ValueError("y_true cannot be empty")
    if raw_true_labels.shape != raw_predicted_labels.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.isin(raw_true_labels, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isin(raw_predicted_labels, [0, 1]).all():
        raise ValueError("y_pred must contain only 0 and 1")
    true_labels = raw_true_labels.astype(int)
    predicted_labels = raw_predicted_labels.astype(int)

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    precision = float(precision_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    ))
    recall = float(recall_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    ))
    f1 = float(f1_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    ))

    beta = 2.0
    beta_squared = beta ** 2
    if precision + recall == 0:
        f2 = 0.0
    else:
        f2 = (
            (1 + beta_squared)
            * precision
            * recall
            / ((beta_squared * precision) + recall)
        )

    metrics: dict[str, Any] = {
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": float(f2),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_alerts": int(np.sum(predicted_labels)),
        "alert_rate": float(np.mean(predicted_labels)),
        "false_positive_rate": float(
            fp / (tn + fp) if tn + fp else 0.0
        ),
        "false_negative_rate": float(
            fn / (tp + fn) if tp + fn else 0.0
        ),
    }

    if y_scores is not None:
        scores = np.asarray(y_scores, dtype=float).ravel()
        if scores.shape != true_labels.shape:
            raise ValueError("y_scores must have the same shape as y_true")
        if not np.isfinite(scores).all():
            raise ValueError("y_scores must contain only finite values")
        if np.unique(true_labels).size == 2:
            metrics["roc_auc"] = float(
                roc_auc_score(true_labels, scores)
            )
            metrics["average_precision"] = float(
                average_precision_score(true_labels, scores)
            )

    return metrics


def evaluate_labeled_report(
    report: pd.DataFrame,
    score_column: str,
    target_column: str = "ground_truth_label",
    prediction_column: str = "predicted_class",
) -> dict[str, Any]:
    """Evaluate predictions against labels attached after inference."""
    required = {target_column, prediction_column, score_column}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(
            f"Prediction report is missing evaluation columns: {sorted(missing)}"
        )
    return evaluate_binary_predictions(
        report[target_column].to_numpy(),
        report[prediction_column].to_numpy(),
        report[score_column].to_numpy(),
    )