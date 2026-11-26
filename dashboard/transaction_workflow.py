"""Model workflows shared by the dashboard transaction pages."""

from __future__ import annotations

import pandas as pd

from src.data_loader import normalize_target_column
from src.prediction import (
    ModelName,
    model_profile_for_transactions,
    predict_transaction_report,
)


def score_uploaded_transactions(
    transactions: pd.DataFrame,
    model_name: ModelName,
) -> pd.DataFrame:
    """Run inference using the saved model for the CSV's detected schema."""
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise ValueError("The uploaded CSV must contain transaction rows.")
    model_profile_for_transactions(transactions)
    return predict_transaction_report(transactions, model_name=model_name)


def train_selected_model(
    transactions: pd.DataFrame,
    model_name: ModelName,
) -> dict[str, object]:
    """Train one model, routing its artifacts from the training CSV schema."""
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise ValueError("The training CSV must contain transaction rows.")
    model_profile_for_transactions(transactions)
    labeled_transactions = normalize_target_column(transactions)

    if model_name == "random_forest":
        from src.train_random_forest import train_random_forest

        return train_random_forest(labeled_transactions)
    if model_name == "isolation_forest":
        from src.train_isolation_forest import train_isolation_forest

        return train_isolation_forest(labeled_transactions)
    if model_name == "autoencoder":
        from src.train_autoencoder import train_autoencoder

        return train_autoencoder(labeled_transactions)
    raise ValueError(f"Unknown model name: {model_name}")
