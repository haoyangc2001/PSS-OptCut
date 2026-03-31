"""Evaluation utilities for regression and bundle-selection quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .constants import GROUP_COLUMN


def summarize_regression_metrics(y_true, y_pred, *, prefix: str = "") -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        f"{prefix}mse": mse,
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
    }


def summarize_task_metrics(
    eval_df: pd.DataFrame,
    *,
    group_col: str = GROUP_COLUMN,
    topk: tuple[int, ...] = (1, 3, 5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_instance_rows = []

    for instance_id, group_df in eval_df.groupby(group_col):
        current_df = group_df.copy()
        if len(current_df) < 2:
            continue

        true_best_idx = current_df["true_runtime"].idxmin()
        pred_best_idx = current_df["pred_runtime"].idxmin()
        rho = spearmanr(current_df["true_runtime"], current_df["pred_runtime"]).correlation

        row = {
            group_col: instance_id,
            "bundle_count": len(current_df),
            "true_best_runtime": float(current_df.loc[true_best_idx, "true_runtime"]),
            "pred_selected_runtime": float(current_df.loc[pred_best_idx, "true_runtime"]),
            "regret": float(
                current_df.loc[pred_best_idx, "true_runtime"]
                - current_df.loc[true_best_idx, "true_runtime"]
            ),
            "top1_hit": int(pred_best_idx == true_best_idx),
            "spearman": np.nan if pd.isna(rho) else float(rho),
        }

        pred_ranked = current_df.nsmallest(len(current_df), "pred_runtime")
        for k in topk:
            topk_indices = set(pred_ranked.head(min(k, len(current_df))).index)
            row[f"top{k}_hit"] = int(true_best_idx in topk_indices)
        per_instance_rows.append(row)

    case_df = pd.DataFrame(per_instance_rows)
    if case_df.empty:
        summary_df = pd.DataFrame(
            [
                {
                    "instance_count": 0,
                    "avg_spearman": np.nan,
                    "top1_hit_rate": np.nan,
                    "top3_hit_rate": np.nan,
                    "top5_hit_rate": np.nan,
                    "mean_regret": np.nan,
                    "median_regret": np.nan,
                    "max_regret": np.nan,
                }
            ]
        )
        return case_df, summary_df

    summary_df = pd.DataFrame(
        [
            {
                "instance_count": int(len(case_df)),
                "avg_spearman": case_df["spearman"].dropna().mean(),
                "top1_hit_rate": case_df["top1_hit"].mean(),
                "top3_hit_rate": case_df.get("top3_hit", pd.Series(dtype=float)).mean(),
                "top5_hit_rate": case_df.get("top5_hit", pd.Series(dtype=float)).mean(),
                "mean_regret": case_df["regret"].mean(),
                "median_regret": case_df["regret"].median(),
                "max_regret": case_df["regret"].max(),
            }
        ]
    )
    return case_df, summary_df


def evaluate_random_holdout(
    model_builder,
    X_df: pd.DataFrame,
    y_series: pd.Series,
    *,
    test_size: float = 0.33,
    random_state: int = 7,
):
    train_idx, test_idx = train_test_split(
        np.arange(len(X_df)),
        test_size=test_size,
        random_state=random_state,
    )
    model = model_builder()
    model.fit(X_df.iloc[train_idx], y_series.iloc[train_idx])

    train_pred = model.predict(X_df.iloc[train_idx])
    test_pred = model.predict(X_df.iloc[test_idx])
    summary = {
        **summarize_regression_metrics(y_series.iloc[train_idx], train_pred, prefix="train_"),
        **summarize_regression_metrics(y_series.iloc[test_idx], test_pred, prefix="test_"),
    }
    return model, summary, train_idx, test_idx


def evaluate_grouped_holdout(
    model_builder,
    X_df: pd.DataFrame,
    y_series: pd.Series,
    meta_df: pd.DataFrame,
    *,
    group_col: str = GROUP_COLUMN,
    test_size: float = 0.33,
    random_state: int = 7,
):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X_df, y_series, groups=meta_df[group_col]))

    model = model_builder()
    model.fit(X_df.iloc[train_idx], y_series.iloc[train_idx])

    train_pred = model.predict(X_df.iloc[train_idx])
    test_pred = model.predict(X_df.iloc[test_idx])
    regression_summary = {
        **summarize_regression_metrics(y_series.iloc[train_idx], train_pred, prefix="train_"),
        **summarize_regression_metrics(y_series.iloc[test_idx], test_pred, prefix="test_"),
    }

    eval_df = meta_df.iloc[test_idx].reset_index(drop=True).copy()
    eval_df["true_runtime"] = y_series.iloc[test_idx].reset_index(drop=True)
    eval_df["pred_runtime"] = pd.Series(test_pred)
    task_case_df, task_summary_df = summarize_task_metrics(eval_df, group_col=group_col)
    return model, regression_summary, task_case_df, task_summary_df, train_idx, test_idx
