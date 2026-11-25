"""Shared Autoencoder reconstruction scoring for training and inference."""

from __future__ import annotations

import numpy as np


def reconstruction_scores(
    original: np.ndarray,
    reconstructed: np.ndarray,
    method: str = "mse",
    feature_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate higher-is-more-anomalous reconstruction scores."""
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if original.ndim != 2 or reconstructed.ndim != 2:
        raise ValueError("Reconstruction inputs must be two-dimensional.")
    if original.shape != reconstructed.shape:
        raise ValueError(
            "Original and reconstructed arrays must have the same shape."
        )
    if not np.isfinite(original).all() or not np.isfinite(reconstructed).all():
        raise ValueError("Reconstruction inputs must contain finite values.")

    absolute_errors = np.abs(original - reconstructed)
    if method == "mse":
        return np.mean(np.square(absolute_errors), axis=1)
    if method == "mae":
        return np.mean(absolute_errors, axis=1)
    if method == "feature_normalized_mae":
        if feature_scales is None:
            raise ValueError(
                "feature_scales are required for feature_normalized_mae."
            )
        scales = np.asarray(feature_scales, dtype=np.float64).reshape(-1)
        if scales.size != original.shape[1]:
            raise ValueError(
                "feature_scales must contain one value per input feature."
            )
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("feature_scales must be finite and positive.")
        return np.mean(absolute_errors / scales, axis=1)
    raise ValueError(
        "method must be 'mse', 'mae', or 'feature_normalized_mae'."
    )
