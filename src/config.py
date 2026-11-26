"""Central configuration and artifact paths for the AML project."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"

TARGET_COLUMN = "Is Laundering"
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_TRAINING_SAMPLE_SIZE = 500_000
RANDOM_SEED = 42
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
TRAIN_FRACTION = 0.70


def _detect_dataset() -> Path | None:
    override = os.getenv("AML_DATASET_PATH")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    if not RAW_DATA_DIR.is_dir():
        return None

    expected_name = "hi-medium_trans.csv"
    for path in RAW_DATA_DIR.iterdir():
        if path.is_file() and path.name.casefold() == expected_name:
            return path.resolve()
    return None


DATASET_PATH = _detect_dataset()
DEFAULT_DATASET_PATH = DATASET_PATH
DEFAULT_DATA_PATH = DEFAULT_DATASET_PATH

RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
RANDOM_FOREST_METRICS_PATH = METRICS_DIR / "random_forest_metrics.json"
RANDOM_FOREST_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "random_forest_predictions.csv"
)

ISOLATION_FOREST_MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
ISOLATION_FOREST_METRICS_PATH = METRICS_DIR / "isolation_forest_metrics.json"
ISOLATION_FOREST_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "isolation_forest_predictions.csv"
)

AUTOENCODER_MODEL_PATH = MODEL_DIR / "autoencoder.keras"
AUTOENCODER_PREPROCESSOR_PATH = MODEL_DIR / "autoencoder_preprocessor.joblib"
AUTOENCODER_METRICS_PATH = METRICS_DIR / "autoencoder_metrics.json"
AUTOENCODER_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "autoencoder_predictions.csv"
)
AUTOENCODER_THRESHOLD_PATH = AUTOENCODER_METRICS_PATH

SYNTHETIC_MODEL_DIR = MODEL_DIR / "synthetic-v1"
SYNTHETIC_METRICS_DIR = METRICS_DIR / "synthetic-v1"
SYNTHETIC_PREDICTIONS_DIR = PREDICTIONS_DIR / "synthetic-v1"
SYNTHETIC_PLOTS_DIR = PLOTS_DIR / "synthetic-v1"

MODEL_PATHS = {
    "random_forest": RANDOM_FOREST_MODEL_PATH,
    "isolation_forest": ISOLATION_FOREST_MODEL_PATH,
    "autoencoder": AUTOENCODER_MODEL_PATH,
}

METRICS_PATHS = {
    "random_forest": RANDOM_FOREST_METRICS_PATH,
    "isolation_forest": ISOLATION_FOREST_METRICS_PATH,
    "autoencoder": AUTOENCODER_METRICS_PATH,
}

PREDICTION_PATHS = {
    "random_forest": RANDOM_FOREST_PREDICTIONS_PATH,
    "isolation_forest": ISOLATION_FOREST_PREDICTIONS_PATH,
    "autoencoder": AUTOENCODER_PREDICTIONS_PATH,
}


def model_artifact_paths(
    model_name: str,
    profile: str = "ibm",
) -> tuple[Path, Path, Path, Path | None]:
    """Return model, metrics, predictions, and optional preprocessor paths."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model name: {model_name}")
    if profile == "ibm":
        preprocessor_path = (
            AUTOENCODER_PREPROCESSOR_PATH
            if model_name == "autoencoder"
            else None
        )
        return (
            MODEL_PATHS[model_name],
            METRICS_PATHS[model_name],
            PREDICTION_PATHS[model_name],
            preprocessor_path,
        )
    if profile != "synthetic":
        raise ValueError("profile must be 'ibm' or 'synthetic'.")

    extension = ".keras" if model_name == "autoencoder" else ".joblib"
    model_path = SYNTHETIC_MODEL_DIR / f"{model_name}{extension}"
    preprocessor_path = (
        SYNTHETIC_MODEL_DIR / "autoencoder_preprocessor.joblib"
        if model_name == "autoencoder"
        else None
    )
    return (
        model_path,
        SYNTHETIC_METRICS_DIR / f"{model_name}_metrics.json",
        SYNTHETIC_PREDICTIONS_DIR / f"{model_name}_predictions.csv",
        preprocessor_path,
    )