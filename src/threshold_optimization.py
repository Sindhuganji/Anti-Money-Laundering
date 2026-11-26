"""Validation-only threshold selection for binary classification scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from src.evaluation import evaluate_binary_predictions


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    false_positive_rate: float
    false_negative_rate: float
    alerts: int


def threshold_table(
    y_true: np.ndarray,
    positive_scores: np.ndarray,
    strict: bool = False,
) -> pd.DataFrame:
    """Evaluate candidate thresholds over validation labels and scores."""
    y_true = np.asarray(y_true).reshape(-1)
    positive_scores = np.asarray(positive_scores).reshape(-1)

    if y_true.shape[0] != positive_scores.shape[0]:
        raise ValueError("y_true and scores must have equal lengths.")
    if y_true.size == 0:
        raise ValueError("y_true and scores cannot be empty.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    y_true = y_true.astype(int)
    if not np.isfinite(positive_scores).all():
        raise ValueError("scores must contain only finite values.")

    thresholds = np.unique(np.quantile(
        positive_scores,
        np.linspace(0.0, 1.0, 201),
    ))
    thresholds = np.append(
        thresholds,
        np.nextafter(float(np.max(positive_scores)), np.inf),
    )

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        comparison = (
            positive_scores > threshold
            if strict
            else positive_scores >= threshold
        )
        predicted = comparison.astype(int)
        tn = int(((y_true == 0) & (predicted == 0)).sum())
        fp = int(((y_true == 0) & (predicted == 1)).sum())
        fn = int(((y_true == 1) & (predicted == 0)).sum())
        tp = int(((y_true == 1) & (predicted == 1)).sum())

        negative_count = tn + fp
        positive_count = tp + fn
        rows.append({
            "threshold": float(threshold),
            "precision": float(
                precision_score(y_true, predicted, zero_division=0)
            ),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "f2": float(
                fbeta_score(
                    y_true,
                    predicted,
                    beta=2,
                    zero_division=0,
                )
            ),
            "false_positive_rate": float(
                fp / negative_count if negative_count else 0.0
            ),
            "false_negative_rate": float(
                fn / positive_count if positive_count else 0.0
            ),
            "alerts": int(tp + fp),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_threshold(
    y_true: np.ndarray,
    positive_scores: np.ndarray,
    strategy: str = "f2",
) -> tuple[float, pd.DataFrame]:
    """Select a validation threshold by F2, recall, or F1."""
    table = threshold_table(y_true, positive_scores)
    if table.empty:
        raise ValueError("No thresholds could be evaluated.")

    strategy = strategy.lower()
    if strategy not in {"f1", "f2", "recall"}:
        raise ValueError("strategy must be 'f1', 'f2', or 'recall'")

    if strategy == "recall":
        candidates = table.sort_values(
            ["recall", "precision", "threshold"],
            ascending=[False, False, True],
        )
    else:
        candidates = table.sort_values(
            [strategy, "precision", "threshold"],
            ascending=[False, False, True],
        )

    selected = candidates.iloc[0]
    return float(selected["threshold"]), table


def select_normal_validation_threshold(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    quantile: float = 0.995,
) -> float:
    """Set an anomaly cutoff from normal-only validation scores."""
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(anomaly_scores, dtype=float).reshape(-1)
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError(
            "Labels and scores must be non-empty and have equal lengths."
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1.")

    normal_scores = scores[labels.astype(int) == 0]
    if normal_scores.size == 0:
        raise ValueError(
            "Normal validation scores are required for calibration."
        )
    return float(np.quantile(normal_scores, quantile))


def normal_validation_threshold_table(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    quantiles: tuple[float, ...] = (0.98, 0.99, 0.995, 0.999),
) -> pd.DataFrame:
    """Compare anomaly cutoffs calibrated only from legitimate validation rows."""
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(anomaly_scores, dtype=float).reshape(-1)
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError(
            "Labels and scores must be non-empty and have equal lengths."
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")
    if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("quantiles must contain values between 0 and 1.")

    normal_scores = scores[labels == 0]
    if normal_scores.size == 0:
        raise ValueError("Normal validation scores are required.")

    rows = []
    for quantile in sorted(set(quantiles)):
        threshold = float(np.quantile(normal_scores, quantile))
        predictions = (scores >= threshold).astype(int)
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