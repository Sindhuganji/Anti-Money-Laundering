"""Dataset discovery, chunked loading, inspection, and data splitting."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_TRAINING_SAMPLE_SIZE,
    RANDOM_SEED,
    TARGET_COLUMN,
)


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


def find_dataset(explicit_path: str | Path | None = None) -> Path:
    """Find the configured dataset or raise an actionable error."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    elif DEFAULT_DATASET_PATH is not None:
        path = Path(DEFAULT_DATASET_PATH).expanduser()
    else:
        raise FileNotFoundError(
            f"No IBM AML dataset was detected in "
            f"{Path(__file__).resolve().parents[1] / 'data' / 'raw'}.\n"
            "Place the CSV in data/raw/ or set AML_DATASET_PATH to its path."
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"IBM AML dataset not found at: {path}\n"
            "Place the CSV in data/raw/ or pass its path explicitly."
        )

    return path.resolve()


def _validate_target(df: pd.DataFrame) -> None:
    """Validate that the target exists and contains only binary labels."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values.")

    values = set(df[TARGET_COLUMN].unique())
    if not values.issubset({0, 1}):
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} must contain only 0 and 1; "
            f"found: {sorted(values, key=str)}"
        )


def iter_transaction_chunks(
    path: str | Path | None = None,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Read the dataset in bounded-memory chunks and validate its columns."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    csv_path = find_dataset(path)

    for chunk_number, chunk in enumerate(
        pd.read_csv(csv_path, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        missing = EXPECTED_COLUMNS.difference(chunk.columns)
        if missing:
            raise ValueError(
                f"Chunk {chunk_number} has missing columns for the expected "
                f"IBM AML schema: "
                f"{sorted(missing)}"
            )

        _validate_target(chunk)
        yield chunk


def inspect_dataset(
    path: str | Path | None = None,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, object]:
    """Inspect a CSV in chunks without retaining the entire file in memory."""
    csv_path = find_dataset(path)

    rows = 0
    duplicates = 0
    missing: Counter[str] = Counter()
    target_counts: Counter[int] = Counter()
    dtypes: dict[str, str] = {}

    for chunk_number, chunk in enumerate(
        iter_transaction_chunks(csv_path, chunksize),
        start=1,
    ):
        rows += len(chunk)
        duplicates += int(chunk.duplicated().sum())
        missing.update(chunk.isna().sum().astype(int).to_dict())

        for label, count in chunk[TARGET_COLUMN].value_counts().items():
            target_counts[int(label)] += int(count)

        if chunk_number == 1:
            dtypes = {
                column: str(dtype)
                for column, dtype in chunk.dtypes.items()
            }

    if rows == 0:
        raise ValueError("The dataset contains no transaction rows.")

    return {
        "path": str(csv_path),
        "rows": rows,
        "columns": list(dtypes),
        "dtypes": dtypes,
        "missing_values": {
            column: count
            for column, count in missing.items()
            if count
        },
        "duplicate_rows_observed": duplicates,
        "target_counts": dict(sorted(target_counts.items())),
    }


def load_dataset(
    path: str | Path | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load the dataset into memory and validate its target column."""
    csv_path = find_dataset(path)
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    if df.empty:
        raise ValueError(f"The dataset is empty: {csv_path}")

    _validate_target(df)
    return df


def load_training_sample(
    path: str | Path | None = None,
    max_rows: int = DEFAULT_TRAINING_SAMPLE_SIZE,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Uniformly sample real rows from a large dataset in bounded memory.

    Random priorities are generated per row, retaining only the ``max_rows``
    smallest priorities seen so far. The resulting sample preserves the
    dataset's observed class prevalence and never synthesizes transactions.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    csv_path = find_dataset(path)
    rng = np.random.default_rng(random_seed)
    reservoir = pd.DataFrame()
    priority_column = "__aml_sampling_priority__"

    for chunk in iter_transaction_chunks(csv_path, chunksize):
        priorities = rng.random(len(chunk))
        candidate = chunk.copy()
        candidate[priority_column] = priorities

        if not reservoir.empty:
            candidate = pd.concat([reservoir, candidate], ignore_index=True)

        if len(candidate) > max_rows:
            candidate = candidate.nsmallest(max_rows, priority_column)
        reservoir = candidate

    if reservoir.empty:
        raise ValueError("The dataset contains no transaction rows.")

    return (
        reservoir.drop(columns=[priority_column])
        .sample(frac=1.0, random_state=random_seed)
        .reset_index(drop=True)
    )


def stratified_train_validation_test_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stratified train, validation, and test DataFrames.

    The test set is split off from the remaining data and should be reserved
    for final evaluation. Do not use it to fit preprocessing or select model
    thresholds.
    """
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    test_fraction = 1.0 - train_fraction - validation_fraction
    if test_fraction <= 0.0:
        raise ValueError(
            "train_fraction plus validation_fraction must be less than 1.0."
        )

    labels = df[target_column]
    if labels.isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")

    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "Stratified splitting requires at least two target classes."
        )
    if int(class_counts.min()) < 3:
        raise ValueError(
            "Each target class needs at least 3 rows for stratified "
            "train/validation/test splitting."
        )

    try:
        train_df, remaining_df = train_test_split(
            df,
            train_size=train_fraction,
            random_state=random_seed,
            shuffle=True,
            stratify=labels,
        )

        # Convert the desired overall validation fraction into the fraction
        # of the remaining rows that should become the validation set.
        remaining_fraction = validation_fraction + test_fraction
        validation_size = validation_fraction / remaining_fraction

        validation_df, test_df = train_test_split(
            remaining_df,
            train_size=validation_size,
            random_state=random_seed,
            shuffle=True,
            stratify=remaining_df[target_column],
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create stratified train/validation/test splits. "
            "Check that the dataset has enough examples of each target class "
            "for the requested fractions."
        ) from exc

    return train_df, validation_df, test_df