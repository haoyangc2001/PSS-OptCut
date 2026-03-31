"""Feature engineering helpers rewritten from the notebook."""

from __future__ import annotations

import ast
import re
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st

from .constants import BUNDLE_COLUMN, STAT_FEATURE_SPECS

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_numeric_values(serialized: object) -> np.ndarray:
    """Parse a serialized vector or matrix cell into a flat numeric array."""

    values = NUMBER_PATTERN.findall("" if serialized is None else str(serialized))
    if not values:
        return np.array([], dtype=float)
    return np.asarray([float(value) for value in values], dtype=float)


def summarize_numeric_values(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return median, mode, skewness, and kurtosis for a numeric array."""

    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mode_result = st.mode(values, axis=None, keepdims=False)
        mode_value = float(mode_result.mode) if np.size(mode_result.mode) else np.nan
        skew_value = float(st.skew(values, bias=False, nan_policy="omit"))
        kurtosis_value = float(st.kurtosis(values, bias=False, nan_policy="omit"))

    return float(np.median(values)), mode_value, skew_value, kurtosis_value


def build_stat_feature_frame(df: pd.DataFrame, source_column: str, feature_prefix: str) -> pd.DataFrame:
    """Generate the notebook's 4 statistical summary features for a serialized column."""

    rows = []
    for cell in df[source_column]:
        values = extract_numeric_values(cell)
        rows.append(summarize_numeric_values(values))

    return pd.DataFrame(
        rows,
        columns=[
            f"{feature_prefix}中位数",
            f"{feature_prefix}众数",
            f"{feature_prefix}偏度",
            f"{feature_prefix}峰度",
        ],
    )


def build_all_stat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all statistical feature blocks used by the notebook."""

    frames = [
        build_stat_feature_frame(df, source_column, feature_prefix)
        for source_column, feature_prefix in STAT_FEATURE_SPECS
    ]
    return pd.concat(frames, axis=1)


def expand_valid_inequality_selection(series: pd.Series) -> pd.DataFrame:
    """Split the 6-bit cut-bundle tuple into Feature1..Feature6 columns."""

    rows = []
    for value in series:
        parsed = ast.literal_eval(str(value))
        if len(parsed) != 6:
            raise ValueError(f"Expected 6 cut bits, got {parsed!r}")
        rows.append(tuple(int(bit) for bit in parsed))

    return pd.DataFrame(rows, columns=[f"Feature{i}" for i in range(1, 7)])
