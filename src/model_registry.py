"""Central registry for profile-isolated saved model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

import pandas as pd

from src.config import model_artifact_paths
from src.data_loader import detect_dataset_profile

ModelName = Literal[
    "random_forest",
    "isolation_forest",
    "autoencoder",
]
ModelProfile = Literal["ibm", "synthetic"]

MODEL_NAMES: tuple[ModelName, ...] = (
    "random_forest",
    "isolation_forest",
    "autoencoder",
)
MODEL_PROFILES: tuple[ModelProfile, ...] = ("ibm", "synthetic")
DEFAULT_MODEL_PROFILE: ModelProfile = "ibm"


class ModelArtifacts(NamedTuple):
    model: Path
    metrics: Path
    predictions: Path
    preprocessor: Path | None


MODEL_REGISTRY: dict[
    ModelName,
    dict[ModelProfile, ModelArtifacts],
] = {
    model_name: {
        profile: ModelArtifacts(*model_artifact_paths(model_name, profile))
        for profile in MODEL_PROFILES
    }
    for model_name in MODEL_NAMES
}


def get_model_artifacts(
    model_name: ModelName,
    profile: ModelProfile,
) -> ModelArtifacts:
    """Return one algorithm's artifacts for exactly one detected profile."""
    try:
        return MODEL_REGISTRY[model_name][profile]
    except KeyError as error:
        raise ValueError(
            f"No saved-model registry entry for {model_name!r}/{profile!r}."
        ) from error


def resolve_model_profile(
    transactions: pd.DataFrame | None = None,
) -> ModelProfile:
    """Infer profile from input columns or use IBM for form-based input."""
    if transactions is None:
        return DEFAULT_MODEL_PROFILE
    profile = detect_dataset_profile(transactions.columns)
    if profile not in MODEL_PROFILES:
        raise ValueError(
            "Transaction columns do not match a supported IBM HI-Small or "
            "synthetic AML schema."
        )
    return profile
