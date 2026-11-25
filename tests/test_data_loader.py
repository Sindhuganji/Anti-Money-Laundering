"""Tests for AML dataset loading and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    find_dataset,
    inspect_dataset,
    iter_transaction_chunks,
    load_dataset,
    load_training_sample,
)
from src.config import TARGET_COLUMN
from src import config


EXPECTED_COLUMNS = {
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    TARGET_COLUMN,
}


def test_missing_dataset_has_clear_error(tmp_path) -> None:
    missing_path = tmp_path / "missing_transactions.csv"

    with pytest.raises(FileNotFoundError, match="dataset not found"):
        find_dataset(missing_path)


def test_find_existing_dataset(tmp_path) -> None:
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame({
        "Timestamp": ["2022/09/01 00:17"],
        TARGET_COLUMN: [0],
    }).to_csv(csv_path, index=False)

    found = find_dataset(csv_path)
    assert found == csv_path.resolve()


def test_config_selects_only_original_medium_dataset(tmp_path, monkeypatch) -> None:
    (tmp_path / "transactions.csv").touch()
    (tmp_path / "HI-Medium_Trans.csv").touch()
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path)
    monkeypatch.delenv("AML_DATASET_PATH", raising=False)

    assert config._detect_dataset() == (tmp_path / "HI-Medium_Trans.csv").resolve()


def test_config_does_not_fall_back_to_another_csv(tmp_path, monkeypatch) -> None:
    (tmp_path / "different_dataset.csv").touch()
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path)
    monkeypatch.delenv("AML_DATASET_PATH", raising=False)

    assert config._detect_dataset() is None


def test_load_dataset_requires_target(tmp_path) -> None:
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame({
        "Timestamp": ["2022/09/01 00:17"],
        "Amount Received": [100.0],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing target column"):
        load_dataset(csv_path)


def test_load_dataset_rejects_missing_target_values(tmp_path) -> None:
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame({
        "Timestamp": ["2022/09/01 00:17", "2022/09/01 00:18"],
        "Amount Received": [100.0, 200.0],
        TARGET_COLUMN: [0, None],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing"):
        load_dataset(csv_path)


def test_load_dataset_accepts_binary_target(tmp_path) -> None:
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame({
        "Timestamp": [
            "2022/09/01 00:17",
            "2022/09/01 00:18",
        ],
        "Amount Received": [100.0, 200.0],
        TARGET_COLUMN: [0, 1],
    }).to_csv(csv_path, index=False)

    dataframe = load_dataset(csv_path)

    assert len(dataframe) == 2
    assert set(dataframe[TARGET_COLUMN].unique()).issubset({0, 1})


def test_load_dataset_rejects_invalid_target_values(tmp_path) -> None:
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame({
        "Timestamp": ["2022/09/01 00:17", "2022/09/01 00:18"],
        "Amount Received": [100.0, 200.0],
        TARGET_COLUMN: [0, 2],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="only 0 and 1"):
        load_dataset(csv_path)


def test_chunk_loader_validates_columns(tmp_path) -> None:
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame({
        "Timestamp": ["2022/09/01 00:17"],
        "Amount Received": [100.0],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        list(iter_transaction_chunks(csv_path, chunksize=1))


def test_chunk_loader_reads_small_dataset(tmp_path) -> None:
    complete_row = {
        "Timestamp": "2022/09/01 00:17",
        "From Bank": 20,
        "Account": "800104D70",
        "To Bank": 20,
        "Account.1": "800104D70",
        "Amount Received": 6794.63,
        "Receiving Currency": "US Dollar",
        "Amount Paid": 6794.63,
        "Payment Currency": "US Dollar",
        "Payment Format": "Reinvestment",
        TARGET_COLUMN: 0,
    }
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame([complete_row, complete_row]).to_csv(
        csv_path,
        index=False,
    )

    chunks = list(iter_transaction_chunks(csv_path, chunksize=1))

    assert len(chunks) == 2
    assert set(chunks[0].columns) == EXPECTED_COLUMNS


def test_inspect_dataset_returns_target_counts(tmp_path) -> None:
    complete_row = {
        "Timestamp": "2022/09/01 00:17",
        "From Bank": 20,
        "Account": "800104D70",
        "To Bank": 20,
        "Account.1": "800104D70",
        "Amount Received": 6794.63,
        "Receiving Currency": "US Dollar",
        "Amount Paid": 6794.63,
        "Payment Currency": "US Dollar",
        "Payment Format": "Reinvestment",
    }
    rows = [
        {**complete_row, TARGET_COLUMN: 0},
        {**complete_row, TARGET_COLUMN: 0},
        {**complete_row, TARGET_COLUMN: 1},
    ]
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = inspect_dataset(csv_path, chunksize=1)

    assert summary["rows"] == 3
    assert summary["target_counts"] == {0: 2, 1: 1}
    assert summary["missing_values"] == {}
    assert "columns" in summary


def test_training_sample_is_bounded_reproducible_and_real(tmp_path) -> None:
    rows = []
    for index in range(120):
        rows.append({
            "Timestamp": f"2022/09/01 00:{index % 60:02}",
            "From Bank": index,
            "Account": f"account-{index}",
            "To Bank": index + 1,
            "Account.1": f"account-{index + 1}",
            "Amount Received": float(index),
            "Receiving Currency": "US Dollar",
            "Amount Paid": float(index),
            "Payment Currency": "US Dollar",
            "Payment Format": "ACH",
            TARGET_COLUMN: index % 20 == 0,
        })
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    sample = load_training_sample(
        csv_path,
        max_rows=25,
        chunksize=13,
        random_seed=11,
    )
    repeated = load_training_sample(
        csv_path,
        max_rows=25,
        chunksize=13,
        random_seed=11,
    )

    assert len(sample) == 25
    pd.testing.assert_frame_equal(sample, repeated)
    assert set(sample["Amount Received"]).issubset(
        {row["Amount Received"] for row in rows}
    )


def test_training_sample_rejects_nonpositive_limit(tmp_path) -> None:
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame([{
        "Timestamp": "2022/09/01 00:17",
        TARGET_COLUMN: 0,
    }]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="max_rows must be positive"):
        load_training_sample(csv_path, max_rows=0)