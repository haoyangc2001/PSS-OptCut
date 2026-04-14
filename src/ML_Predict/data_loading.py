"""Load and clean runtime experiment data from result.csv."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .constants import EXPECTED_BUNDLE_COUNT, GROUP_COLUMN, INSTANCE_KEY_COLUMNS, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"
DEFAULT_RESULT_CSV = DATA_DIR / "result.csv"


def default_log_run_dir(run_date: date | None = None) -> Path:
    current_date = run_date or date.today()
    return LOG_DIR / f"{current_date:%y%m%d}"


def load_clean_result_csv(
    result_path: Path | str = DEFAULT_RESULT_CSV,
    *,
    min_runtime: float = 0.5,
    require_complete_bundles: bool = True,
    expected_bundle_count: int = EXPECTED_BUNDLE_COUNT,
) -> pd.DataFrame:
    """Load result.csv and apply the notebook's cleaning rules."""

    result_path = Path(result_path)
    raw_df = pd.read_csv(result_path, dtype=str, keep_default_na=False, low_memory=False)

    header_mask = raw_df.eq(pd.Series(raw_df.columns, index=raw_df.columns), axis=1).all(axis=1)
    without_headers_df = raw_df.loc[~header_mask].copy()
    dedup_df = without_headers_df.drop_duplicates(keep="first").copy()

    if TARGET_COLUMN not in dedup_df.columns:
        raise KeyError(
            f"Expected target column '{TARGET_COLUMN}' in {result_path}, "
            "but it was not found. Please regenerate result.csv with the dual-runtime solver output."
        )

    dedup_df[TARGET_COLUMN] = pd.to_numeric(dedup_df[TARGET_COLUMN], errors="coerce")
    dedup_df["random_seed"] = pd.to_numeric(dedup_df["random_seed"], errors="coerce")
    clean_numeric_df = dedup_df.dropna(subset=[TARGET_COLUMN, "random_seed"]).copy()

    runtime_filtered_df = clean_numeric_df.loc[clean_numeric_df[TARGET_COLUMN] > min_runtime].copy()
    runtime_filtered_df[GROUP_COLUMN] = runtime_filtered_df[INSTANCE_KEY_COLUMNS].astype(str).agg("|".join, axis=1)

    if require_complete_bundles:
        bundle_counts_by_instance = runtime_filtered_df[GROUP_COLUMN].value_counts().sort_index()
        complete_instance_ids = bundle_counts_by_instance[
            bundle_counts_by_instance == expected_bundle_count
        ].index
        runtime_filtered_df = runtime_filtered_df.loc[
            runtime_filtered_df[GROUP_COLUMN].isin(complete_instance_ids)
        ].copy()

    return runtime_filtered_df.sort_values([GROUP_COLUMN, "有效不等式选择"]).reset_index(drop=True)
