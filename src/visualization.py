"""Visualization utilities for AML model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def ensure_output_directory(path: str | Path) -> Path:
    """Create the parent directory for a visualization."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a labeled binary confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Suspicious"],
    )
    display.plot(
        cmap="Blues",
        colorbar=False,
        ax=axis,
        values_format="d",
    )
    axis.set_title(title)
    figure.tight_layout()

    destination = ensure_output_directory(output_path)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_roc_curve(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    output_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """Save a receiver operating characteristic curve."""
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        y_scores,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    auc = float(np.trapezoid(true_positive_rate, false_positive_rate))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Model (AUC = {auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_title(title)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()

    destination = ensure_output_directory(output_path)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_precision_recall_curve(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    output_path: str | Path,
    title: str = "Precision-Recall Curve",
) -> None:
    """Save a precision-recall curve."""
    precision, recall, _ = precision_recall_curve(
        y_true,
        y_scores,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay(
        precision=precision,
        recall=recall,
    ).plot(ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    destination = ensure_output_directory(output_path)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_score_distribution(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    output_path: str | Path,
    title: str = "Prediction Score Distribution",
) -> None:
    """Save separate score distributions for legitimate and suspicious records."""
    labels = np.asarray(y_true)
    scores = np.asarray(y_scores, dtype=float)

    legitimate_scores = scores[labels == 0]
    suspicious_scores = scores[labels == 1]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.hist(
        legitimate_scores,
        bins=30,
        alpha=0.65,
        label="Legitimate",
        density=True,
    )
    axis.hist(
        suspicious_scores,
        bins=30,
        alpha=0.65,
        label="Suspicious",
        density=True,
    )
    axis.set_title(title)
    axis.set_xlabel("Prediction Score")
    axis.set_ylabel("Density")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()

    destination = ensure_output_directory(output_path)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_threshold_comparison(
    threshold_table: pd.DataFrame,
    output_path: str | Path,
    title: str = "Validation Threshold Comparison",
) -> None:
    """Plot precision, recall, F1, and F2 over validation thresholds."""
    required = {"threshold", "precision", "recall", "f1", "f2"}
    if not required.issubset(threshold_table.columns):
        raise ValueError(
            f"Threshold table must include {sorted(required)}."
        )

    figure, axis = plt.subplots(figsize=(8, 5))
    for metric in ("precision", "recall", "f1", "f2"):
        axis.plot(
            threshold_table["threshold"],
            threshold_table[metric],
            label=metric.upper(),
        )
    axis.set_title(title)
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Validation score")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    destination = ensure_output_directory(output_path)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_model_comparison(
    metrics: pd.DataFrame,
    output_path: str | Path,
    title: str = "Model Comparison",
) -> None:
    """Plot directly comparable classifier metrics from held-out predictions."""
    columns = [
        column
        for column in (
            "precision",
            "recall",
            "f1_score",
            "f2_score",
            "average_precision",
        )
        if column in metrics.columns
    ]
    if "model" not in metrics.columns or not columns:
        raise ValueError("Comparison data requires model and metric columns.")

    figure, axis = plt.subplots(figsize=(10, 6))
    metrics.set_index("model")[columns].plot.bar(ax=axis)
    axis.set_title(title)
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    destination = ensure_output_directory(output_path)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)