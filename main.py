"""Command-line entry point for the AML project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

TRAINING_MODULES = {
    "train-rf": "src.train_random_forest",
    "train-random-forest": "src.train_random_forest",
    "train-if": "src.train_isolation_forest",
    "train-isolation-forest": "src.train_isolation_forest",
    "train-ae": "src.train_autoencoder",
    "train-autoencoder": "src.train_autoencoder",
}

SCORE_COLUMNS = {
    "random_forest": "probability",
    "isolation_forest": "anomaly_score",
    "autoencoder": "reconstruction_error",
}


def run_command(command: Sequence[str]) -> int:
    """Run a command using this interpreter from the project root."""
    print(f"$ {' '.join(command)}", flush=True)
    result = subprocess.run(list(command), cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        print(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(command)}",
            file=sys.stderr,
        )
    return result.returncode


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AML project commands")
    parser.add_argument(
        "command",
        choices=[
            "explore",
            *TRAINING_MODULES,
            "train-all",
            "dashboard",
            "evaluate",
            "predict",
            "test",
            "sample-data",
        ],
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional options forwarded to the selected command",
    )
    return parser.parse_args()


def validate_dataset(extra: Sequence[str] = ()) -> int:
    """Inspect the configured real dataset without loading it all into memory."""
    parser = argparse.ArgumentParser(description="Inspect the real AML dataset")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    arguments = parser.parse_args(extra)

    try:
        from src.data_loader import inspect_dataset

        summary = inspect_dataset(
            arguments.dataset,
            chunksize=arguments.chunksize,
        )
    except (OSError, ValueError) as exc:
        print(f"Dataset inspection failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


def run_exploration(extra: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description="Explore the AML dataset")
    parser.add_argument("--dataset", type=Path, default=None)
    arguments = parser.parse_args(extra)
    try:
        from src.explore_data import run_exploration as explore

        summary = explore(arguments.dataset)
    except (OSError, ValueError) as exc:
        print(f"Dataset exploration failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


def evaluate_saved_predictions() -> int:
    """Recalculate test metrics from saved predictions and refresh plots."""
    import pandas as pd

    from src.config import (
        METRICS_DIR,
        METRICS_PATHS,
        PLOTS_DIR,
        PREDICTION_PATHS,
    )
    from src.evaluation import evaluate_binary_predictions
    from src.visualization import (
        save_confusion_matrix,
        save_model_comparison,
        save_precision_recall_curve,
        save_roc_curve,
    )

    evaluated: list[dict[str, object]] = []
    for model_name, prediction_path in PREDICTION_PATHS.items():
        if not prediction_path.is_file():
            print(f"{model_name}: test predictions not found ({prediction_path})")
            continue
        try:
            predictions = pd.read_csv(prediction_path)
            metrics_path = METRICS_PATHS[model_name]
            saved = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.is_file()
                else {}
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"{model_name}: unable to load saved results: {exc}")
            return 1

        score_column = SCORE_COLUMNS[model_name]
        required = {"actual", "prediction", score_column}
        missing = required.difference(predictions.columns)
        if missing:
            print(
                f"{model_name}: predictions are missing columns "
                f"{sorted(missing)}",
                file=sys.stderr,
            )
            return 1

        y_true = predictions["actual"].to_numpy()
        y_pred = predictions["prediction"].to_numpy()
        scores = predictions[score_column].to_numpy()
        threshold = saved.get("threshold")
        if threshold is None:
            print(
                f"{model_name}: saved metrics have no prediction threshold.",
                file=sys.stderr,
            )
            return 1
        threshold = float(threshold)
        if not np.isfinite(threshold):
            print(
                f"{model_name}: saved prediction threshold is not finite.",
                file=sys.stderr,
            )
            return 1
        threshold_predictions = (
            scores > threshold
            if model_name == "autoencoder"
            else scores >= threshold
        ).astype(int)
        if not np.array_equal(y_pred, threshold_predictions):
            print(
                f"{model_name}: saved predictions do not match the saved "
                "scores and threshold.",
                file=sys.stderr,
            )
            return 1
        result = evaluate_binary_predictions(y_true, y_pred, scores)
        for key in (
            "threshold",
            "threshold_strategy",
            "threshold_quantile",
            "target_validation_false_positive_rate",
            "normal_validation_rows",
            "validation_normal_false_positive_rate",
            "score_direction",
            "contamination",
            "validation_metrics",
            "training_rows",
            "validation_rows",
            "test_rows",
            "random_state",
            "epochs_requested",
            "epochs_trained",
            "batch_size",
            "latent_dimension",
            "hidden_dimension",
            "dropout_rate",
            "l2_regularization",
            "reconstruction_loss",
            "score_method",
            "score_feature_names",
            "score_feature_scales",
            "score_feature_scale_method",
            "test_score_distribution",
            "test_raw_mse_distribution",
            "validation_threshold_comparison",
        ):
            if key in saved:
                result[key] = saved[key]

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        save_confusion_matrix(
            y_true,
            y_pred,
            PLOTS_DIR / f"{model_name}_confusion_matrix.png",
            f"{model_name.replace('_', ' ').title()} Confusion Matrix",
        )
        save_precision_recall_curve(
            y_true,
            scores,
            PLOTS_DIR / f"{model_name}_precision_recall.png",
            f"{model_name.replace('_', ' ').title()} Precision-Recall Curve",
        )
        save_roc_curve(
            y_true,
            scores,
            PLOTS_DIR / f"{model_name}_roc.png",
            f"{model_name.replace('_', ' ').title()} ROC Curve",
        )
        evaluated.append({"model": model_name, **result})

    if not evaluated:
        print("No saved test predictions found. Train a model first.")
        return 1

    comparison = pd.DataFrame(evaluated)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    save_model_comparison(
        comparison,
        PLOTS_DIR / "model_comparison.png",
    )
    print(comparison.to_string(index=False))
    print(f"\nEvaluation report: {METRICS_DIR / 'model_comparison.csv'}")
    return 0


def run_prediction(extra: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Predict uploaded transactions")
    parser.add_argument(
        "--model",
        required=True,
        choices=["random_forest", "isolation_forest", "autoencoder"],
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    arguments = parser.parse_args(extra)

    try:
        import pandas as pd
        from src.prediction import predict_transaction_report

        transactions = pd.read_csv(arguments.input)
        result = predict_transaction_report(
            transactions,
            arguments.model,
            threshold=arguments.threshold,
        )
        output = arguments.output or (
            PROJECT_ROOT
            / "results"
            / "predictions"
            / f"{arguments.model}_uploaded_predictions.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    alerts = int((result["predicted_class"] == 1).sum())
    print(
        f"Analyzed {len(result):,} transactions; "
        f"{alerts:,} suspicious transaction alert(s)."
    )
    print("An alert is not a confirmed money-laundering determination.")
    print(f"Prediction report: {output}")
    return 0


def train_all(extra: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Train all three models from one real-data sample",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dimension", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument(
        "--reconstruction-loss",
        choices=["mse", "mae", "huber"],
        default="mse",
    )
    parser.add_argument(
        "--score-method",
        choices=["mse", "mae", "feature_normalized_mae"],
        default="feature_normalized_mae",
    )
    parser.add_argument(
        "--normal-validation-quantile",
        type=float,
        default=0.999,
        help=(
            "Autoencoder normal-score quantile for threshold calibration "
            "(default: 0.999)."
        ),
    )
    arguments = parser.parse_args(extra)

    try:
        from src.data_loader import load_training_sample
        from src.train_autoencoder import train_autoencoder
        from src.train_isolation_forest import (
            parse_contamination,
            train_isolation_forest,
        )
        from src.train_random_forest import train_random_forest

        transactions = load_training_sample(
            arguments.dataset,
            max_rows=arguments.max_rows,
            random_seed=arguments.random_state,
        )
        common = {
            "transactions": transactions,
            "test_size": arguments.test_size,
            "validation_size": arguments.validation_size,
            "random_state": arguments.random_state,
        }
        results = {
            "random_forest": train_random_forest(
                **common,
                estimators=arguments.estimators,
            ),
            "isolation_forest": train_isolation_forest(
                **common,
                contamination=parse_contamination(arguments.contamination),
                estimators=arguments.estimators,
            ),
            "autoencoder": train_autoencoder(
                **common,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                normal_validation_quantile=(
                    arguments.normal_validation_quantile
                ),
                latent_dimension=arguments.latent_dimension,
                dropout_rate=arguments.dropout_rate,
                l2_regularization=arguments.l2_regularization,
                reconstruction_loss=arguments.reconstruction_loss,
                score_method=arguments.score_method,
            ),
        }
    except (OSError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}")
        print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    arguments = parse_arguments()
    command = arguments.command
    extra = arguments.extra

    if command in TRAINING_MODULES:
        return run_command([
            sys.executable,
            "-m",
            TRAINING_MODULES[command],
            *extra,
        ])

    if command == "train-all":
        return train_all(extra)

    if command == "explore":
        return run_exploration(extra)

    if command == "dashboard":
        return run_command([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            *extra,
        ])

    if command == "evaluate":
        return evaluate_saved_predictions()

    if command == "predict":
        return run_prediction(extra)

    if command == "test":
        return run_command([
            sys.executable,
            "-m",
            "pytest",
            "tests",
            *extra,
        ])

    if command == "sample-data":
        return validate_dataset(extra)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
