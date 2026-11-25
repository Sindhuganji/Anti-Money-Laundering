"""Tests for consistent Autoencoder reconstruction scoring."""

from __future__ import annotations

import numpy as np
import pytest

from src.autoencoder_scoring import reconstruction_scores


def test_feature_normalized_mae_uses_saved_per_feature_scales() -> None:
    original = np.array([[2.0, 6.0], [4.0, 12.0]])
    reconstructed = np.zeros_like(original)

    scores = reconstruction_scores(
        original,
        reconstructed,
        method="feature_normalized_mae",
        feature_scales=np.array([2.0, 3.0]),
    )

    np.testing.assert_allclose(scores, [1.5, 3.0])


def test_feature_normalized_mae_rejects_missing_or_invalid_scales() -> None:
    values = np.ones((1, 2))

    with pytest.raises(ValueError, match="feature_scales are required"):
        reconstruction_scores(
            values,
            np.zeros_like(values),
            method="feature_normalized_mae",
        )
    with pytest.raises(ValueError, match="one value per input feature"):
        reconstruction_scores(
            values,
            np.zeros_like(values),
            method="feature_normalized_mae",
            feature_scales=np.ones(1),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        reconstruction_scores(
            values,
            np.zeros_like(values),
            method="feature_normalized_mae",
            feature_scales=np.array([1.0, 0.0]),
        )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("mse", [2.5]),
        ("mae", [1.5]),
    ],
)
def test_unscaled_reconstruction_scores(method: str, expected: list[float]) -> None:
    original = np.array([[1.0, 3.0]])
    reconstructed = np.array([[0.0, 1.0]])

    scores = reconstruction_scores(original, reconstructed, method=method)

    np.testing.assert_allclose(scores, expected)
