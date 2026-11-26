"""Tests for dashboard upload and explicit training boundaries."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard import transaction_workflow
from src import (
    train_autoencoder,
    train_isolation_forest,
    train_random_forest as train_random_forest_module,
)
from src.model_registry import resolve_model_profile


def _synthetic_upload() -> pd.DataFrame:
    return pd.DataFrame({
        "Timestamp": ["2026-01-01 12:00:00"],
        "Sender_ID": ["sender-1"],
        "Receiver_ID": ["receiver-1"],
        "Sender_Bank": ["Bank A"],
        "Receiver_Bank": ["Bank B"],
        "Transaction_Type": ["Wire"],
        "Amount": [150.0],
        "Currency": ["USD"],
        "Payment_Format": ["SWIFT"],
        "Cross_Border": [True],
        "Hour": [12],
        "Day_Of_Week": [3],
        "Is_Laundering": [0],
    })


def test_csv_upload_scoring_never_calls_model_training(monkeypatch) -> None:
    def fail_if_trained(*_args, **_kwargs):
        pytest.fail("CSV scoring must not invoke a training function")

    for module, name in (
        (train_random_forest_module, "train_random_forest"),
        (train_isolation_forest, "train_isolation_forest"),
        (train_autoencoder, "train_autoencoder"),
    ):
        monkeypatch.setattr(module, name, fail_if_trained)

    expected_report = pd.DataFrame({"predicted_class": [0]})
    monkeypatch.setattr(
        transaction_workflow,
        "predict_transaction_report",
        lambda transactions, model_name: expected_report,
    )

    report = transaction_workflow.score_uploaded_transactions(
        _synthetic_upload(),
        model_name="random_forest",
    )

    assert report is expected_report


def test_upload_infers_artifact_profile_from_transaction_columns() -> None:
    assert resolve_model_profile(_synthetic_upload()) == "synthetic"


def test_explicit_training_dispatches_only_the_selected_profile_model(
    monkeypatch,
) -> None:
    calls: list[tuple[pd.DataFrame, str]] = []

    def train_random_forest(transactions):
        calls.append((transactions, resolve_model_profile(transactions)))
        return {"threshold": 0.4}

    monkeypatch.setattr(
        train_random_forest_module,
        "train_random_forest",
        train_random_forest,
    )
    monkeypatch.setattr(
        train_isolation_forest,
        "train_isolation_forest",
        lambda *_args, **_kwargs: pytest.fail("unselected trainer invoked"),
    )
    monkeypatch.setattr(
        train_autoencoder,
        "train_autoencoder",
        lambda *_args, **_kwargs: pytest.fail("unselected trainer invoked"),
    )

    result = transaction_workflow.train_selected_model(
        _synthetic_upload(),
        model_name="random_forest",
    )

    assert result == {"threshold": 0.4}
    assert len(calls) == 1
    assert calls[0][1] == "synthetic"
