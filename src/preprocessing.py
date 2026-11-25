"""Training-fitted preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import TARGET_COLUMN
from src.feature_engineering import engineer_features


DROP_COLUMNS = {
    TARGET_COLUMN,
    "Timestamp",
    "Account",
    "Account.1",
}


def build_preprocessor(
    df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a transformer that must be fitted only on training data."""
    engineered = engineer_features(
        df.drop(columns=[TARGET_COLUMN], errors="ignore")
    )

    feature_columns = [
        column for column in engineered.columns
        if column not in DROP_COLUMNS
    ]
    numeric_features = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(engineered[column].dtype)
    ]
    categorical_features = [
        column for column in feature_columns
        if column not in numeric_features
    ]
    if not feature_columns:
        raise ValueError("No usable transaction features were provided.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor, numeric_features, categorical_features


def transform_dataframe(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Engineer and transform transactions using an already-fitted pipeline."""
    targets: np.ndarray | None = None
    if TARGET_COLUMN in df.columns:
        targets = df[TARGET_COLUMN].to_numpy(dtype=int)
    engineered = engineer_features(
        df.drop(columns=[TARGET_COLUMN], errors="ignore")
    )
    transformed = preprocessor.transform(engineered)
    return transformed, targets