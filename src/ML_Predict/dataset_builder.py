"""Build augmented datasets and model-ready matrices for ML prediction."""

from __future__ import annotations

import pandas as pd

from .constants import DEFAULT_DROP_COLUMNS, GROUP_COLUMN, INSTANCE_KEY_COLUMNS, TARGET_COLUMN
from .feature_engineering import build_all_stat_features, expand_valid_inequality_selection


def build_augmented_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Append notebook-derived stat features and bundle bit columns."""

    base_df = clean_df.reset_index(drop=True).copy()
    stat_df = build_all_stat_features(base_df)
    bundle_df = expand_valid_inequality_selection(base_df["有效不等式选择"])
    return pd.concat([base_df, stat_df, bundle_df], axis=1)


def encode_model_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Encode mixed-type feature columns into numeric form."""

    encoded_df = feature_df.copy()
    for column in encoded_df.columns:
        try:
            encoded_df[column] = pd.to_numeric(encoded_df[column])
        except (TypeError, ValueError):
            pass
    for column in encoded_df.select_dtypes(include=["object"]).columns:
        encoded_df[column] = encoded_df[column].astype("category").cat.codes
    return encoded_df


def build_model_inputs(
    augmented_df: pd.DataFrame,
    *,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create X, y, and metadata tables from the augmented dataset."""

    drop_columns = DEFAULT_DROP_COLUMNS if drop_columns is None else drop_columns
    available_drop_columns = [column for column in drop_columns if column in augmented_df.columns]

    modeling_df = augmented_df.drop(columns=available_drop_columns).copy()
    target = pd.to_numeric(modeling_df[TARGET_COLUMN], errors="coerce")
    features = modeling_df.drop(columns=[TARGET_COLUMN]).copy()
    encoded_features = encode_model_features(features)

    meta_columns = [GROUP_COLUMN, *INSTANCE_KEY_COLUMNS]
    if "有效不等式选择" in augmented_df.columns:
        meta_columns.append("有效不等式选择")

    metadata = augmented_df[meta_columns].reset_index(drop=True).copy()
    return encoded_features.reset_index(drop=True), target.reset_index(drop=True), metadata
