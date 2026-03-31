"""Experiment runners for the modular ML prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .constants import GROUP_COLUMN, MANUAL_GAIN_DROP_FEATURES, MANUAL_SPLIT_DROP_FEATURES
from .evaluation import evaluate_grouped_holdout, evaluate_random_holdout
from .models import MODEL_BUILDERS, make_lgbm_regressor


@dataclass
class ExperimentArtifacts:
    summary_df: pd.DataFrame
    grouped_task_summary_df: pd.DataFrame
    grouped_case_dfs: dict[str, pd.DataFrame]
    importances: dict[str, pd.DataFrame]
    errors: dict[str, str]


def build_importance_df(model, feature_names) -> pd.DataFrame:
    booster = model.booster_
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "split_importance": booster.feature_importance(importance_type="split"),
            "gain_importance": booster.feature_importance(importance_type="gain"),
        }
    )
    importance_df["split_rank"] = importance_df["split_importance"].rank(method="dense", ascending=False)
    importance_df["gain_rank"] = importance_df["gain_importance"].rank(method="dense", ascending=False)
    return importance_df.sort_values(by="gain_importance", ascending=False).reset_index(drop=True)


def run_baseline_experiments(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    meta_df: pd.DataFrame,
    *,
    test_size: float = 0.33,
    random_state: int = 7,
) -> ExperimentArtifacts:
    summary_rows = []
    grouped_task_rows = []
    grouped_case_dfs: dict[str, pd.DataFrame] = {}
    importances: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for model_name, model_builder in MODEL_BUILDERS.items():
        try:
            random_model, random_summary, _, _ = evaluate_random_holdout(
                model_builder,
                X_df,
                y_series,
                test_size=test_size,
                random_state=random_state,
            )
            grouped_model, grouped_summary, case_df, task_df, _, _ = evaluate_grouped_holdout(
                model_builder,
                X_df,
                y_series,
                meta_df,
                group_col=GROUP_COLUMN,
                test_size=test_size,
                random_state=random_state,
            )
        except Exception as exc:
            errors[model_name] = str(exc)
            continue

        summary_rows.append(
            {
                "model": model_name,
                **random_summary,
                **{f"group_{key}": value for key, value in grouped_summary.items()},
            }
        )
        grouped_task_row = task_df.iloc[0].to_dict()
        grouped_task_row["model"] = model_name
        grouped_task_rows.append(grouped_task_row)
        grouped_case_dfs[model_name] = case_df

        if model_name == "lightgbm":
            importances["lightgbm"] = build_importance_df(grouped_model, X_df.columns)

    return ExperimentArtifacts(
        summary_df=pd.DataFrame(summary_rows),
        grouped_task_summary_df=pd.DataFrame(grouped_task_rows),
        grouped_case_dfs=grouped_case_dfs,
        importances=importances,
        errors=errors,
    )


def run_manual_lightgbm_selection_experiments(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    meta_df: pd.DataFrame,
    *,
    split_drop_features: list[str] | None = None,
    gain_drop_features: list[str] | None = None,
    test_size: float = 0.33,
    random_state: int = 7,
) -> ExperimentArtifacts:
    split_drop_features = MANUAL_SPLIT_DROP_FEATURES if split_drop_features is None else split_drop_features
    gain_drop_features = MANUAL_GAIN_DROP_FEATURES if gain_drop_features is None else gain_drop_features

    schemes = {
        "split_manual": split_drop_features,
        "gain_manual": gain_drop_features,
    }

    summary_rows = []
    grouped_task_rows = []
    grouped_case_dfs: dict[str, pd.DataFrame] = {}
    importances: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for scheme_name, drop_features in schemes.items():
        current_features = X_df.drop(
            columns=[column for column in drop_features if column in X_df.columns],
            errors="ignore",
        )
        try:
            _, random_summary, _, _ = evaluate_random_holdout(
                make_lgbm_regressor,
                current_features,
                y_series,
                test_size=test_size,
                random_state=random_state,
            )
            grouped_model, grouped_summary, case_df, task_df, _, _ = evaluate_grouped_holdout(
                make_lgbm_regressor,
                current_features,
                y_series,
                meta_df,
                group_col=GROUP_COLUMN,
                test_size=test_size,
                random_state=random_state,
            )
        except Exception as exc:
            errors[scheme_name] = str(exc)
            continue

        summary_rows.append(
            {
                "scheme": scheme_name,
                "removed_feature_count": len([column for column in drop_features if column in X_df.columns]),
                **random_summary,
                **{f"group_{key}": value for key, value in grouped_summary.items()},
            }
        )
        grouped_task_row = task_df.iloc[0].to_dict()
        grouped_task_row["scheme"] = scheme_name
        grouped_task_rows.append(grouped_task_row)
        grouped_case_dfs[scheme_name] = case_df
        importances[scheme_name] = build_importance_df(grouped_model, current_features.columns)

    return ExperimentArtifacts(
        summary_df=pd.DataFrame(summary_rows),
        grouped_task_summary_df=pd.DataFrame(grouped_task_rows),
        grouped_case_dfs=grouped_case_dfs,
        importances=importances,
        errors=errors,
    )
