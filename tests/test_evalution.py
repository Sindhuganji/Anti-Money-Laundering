"""Tests for binary AML evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import evaluate_binary_predictions


def test_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    metrics = evaluate_binary_predictions(
        y_true,
        y_pred,
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["f2_score"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["total_alerts"] == 2
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0


def test_extreme_false_positive_case() -> None:
    y_true = np.array([0, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1, 1])

    metrics = evaluate_binary_predictions(
        y_true,
        y_pred,
        np.array([0.9, 0.8, 0.7, 0.6, 0.99]),
    )

    assert metrics["true_negatives"] == 0
    assert metrics["false_positives"] == 4
    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == pytest.approx(0.2)
    assert metrics["recall"] == 1.0
    assert metrics["total_alerts"] == 5


def test_no_positive_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])

    metrics = evaluate_binary_predictions(y_true, y_pred)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["f2_score"] == 0.0
    assert metrics["true_positives"] == 0
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 2
    assert metrics["total_alerts"] == 0


def test_scores_are_optional() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    metrics = evaluate_binary_predictions(y_true, y_pred)

    assert "roc_auc" not in metrics
    assert "average_precision" not in metrics


def test_threshold_candidates_are_bounded_and_support_anomaly_scores() -> None:
    from src.threshold_optimization import select_threshold, threshold_table

    labels = np.array([0, 0, 1, 1])
    scores = np.array([-12.0, -3.0, 5.0, 9.0])

    selected, table = select_threshold(labels, scores, strategy="f2")

    assert len(threshold_table(labels, scores)) <= 202
    assert selected in table["threshold"].to_numpy()
    assert table["threshold"].max() > 1.0


def test_threshold_optimizer_rejects_nonfinite_scores() -> None:
    from src.threshold_optimization import threshold_table

    with pytest.raises(ValueError, match="finite"):
        threshold_table(np.array([0, 1]), np.array([0.0, np.inf]))


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [
        ([0.5, 1], [0, 1]),
        ([0, 1], [0.25, 1]),
    ],
)
def test_evaluation_rejects_fractional_binary_labels(
    labels,
    predictions,
) -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        evaluate_binary_predictions(labels, predictions)


def test_evaluation_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_binary_predictions(
            [0, 1],
            [0, 1],
            [0.1, np.nan],
        )