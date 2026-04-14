"""Dataset analysis helpers for the runtime-prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import (
    BASELINE_BUNDLE,
    BUNDLE_COLUMN,
    EXPECTED_BUNDLE_COUNT,
    GROUP_COLUMN,
    INSTANCE_KEY_COLUMNS,
    TARGET_COLUMN,
)
from .feature_engineering import expand_valid_inequality_selection, extract_numeric_values


@dataclass
class AnalysisArtifacts:
    output_dir: Path
    table_dir: Path
    plot_dir: Path
    report_dir: Path
    data_quality_summary_df: pd.DataFrame
    runtime_summary_df: pd.DataFrame
    runtime_by_stage_df: pd.DataFrame
    instance_summary_df: pd.DataFrame
    instance_overview_df: pd.DataFrame
    bundle_summary_df: pd.DataFrame
    bundle_stage_summary_df: pd.DataFrame
    bundle_effect_df: pd.DataFrame
    feature_target_signal_df: pd.DataFrame
    feature_optimal_signal_df: pd.DataFrame
    feature_scalar_summary_df: pd.DataFrame
    split_summary_df: pd.DataFrame
    split_stage_distribution_df: pd.DataFrame
    baseline_policy_summary_df: pd.DataFrame
    report_path: Path
    plot_paths: dict[str, Path]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def _parse_bundle_series(series: pd.Series) -> pd.DataFrame:
    return expand_valid_inequality_selection(series).rename(
        columns={f"Feature{i}": f"bundle_bit_{i}" for i in range(1, 7)}
    )


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _series_stats(series: pd.Series) -> dict[str, float]:
    clean = _safe_numeric(series).dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
            "skew": np.nan,
        }
    return {
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "skew": float(clean.skew()),
    }


def _structured_length_checks(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("服务时间", lambda row: int(row["服务数"])),
        ("服务价格", lambda row: int(row["服务数"])),
        ("服务成本", lambda row: int(row["服务数"])),
        ("库存", lambda row: int(row["产品数"])),
        ("客户权重", lambda row: int(row["客户数"])),
        ("服务时间窗Early", lambda row: int(row["客户数"])),
        ("服务时间窗Late", lambda row: int(row["客户数"])),
        ("产品服务对应关系矩阵", lambda row: int(row["服务数"]) * int(row["产品数"])),
        ("转移时间矩阵", lambda row: (int(row["客户数"]) + 2) ** 2),
        ("可变成本", lambda row: (int(row["客户数"]) + 2) ** 2),
    ]

    rows = []
    for column, expected_len_fn in specs:
        observed_lengths = df[column].map(lambda cell: len(extract_numeric_values(cell)))
        expected_lengths = df.apply(expected_len_fn, axis=1)
        mismatch_mask = observed_lengths != expected_lengths
        rows.append(
            {
                "column": column,
                "checked_rows": int(len(df)),
                "mismatch_count": int(mismatch_mask.sum()),
                "mismatch_rate": float(mismatch_mask.mean()),
            }
        )
    return pd.DataFrame(rows)


def analyze_data_quality(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    instance_bundle_counts = clean_df.groupby(GROUP_COLUMN).size()
    summary_df = pd.DataFrame(
        [
            {
                "row_count": int(len(clean_df)),
                "instance_count": int(clean_df[GROUP_COLUMN].nunique()),
                "duplicate_row_count": int(clean_df.duplicated().sum()),
                "missing_target_count": int(_safe_numeric(clean_df[TARGET_COLUMN]).isna().sum()),
                "missing_seed_count": int(_safe_numeric(clean_df["random_seed"]).isna().sum()),
                "complete_instance_count": int((instance_bundle_counts == EXPECTED_BUNDLE_COUNT).sum()),
                "incomplete_instance_count": int((instance_bundle_counts != EXPECTED_BUNDLE_COUNT).sum()),
                "min_bundle_count_per_instance": int(instance_bundle_counts.min()),
                "max_bundle_count_per_instance": int(instance_bundle_counts.max()),
            }
        ]
    )
    length_check_df = _structured_length_checks(clean_df)
    return summary_df, length_check_df


def analyze_runtime_distribution(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runtime_summary_df = pd.DataFrame([_series_stats(clean_df[TARGET_COLUMN])])

    runtime_by_stage_df = (
        clean_df.groupby(["scale_phase", "scale_stage"], dropna=False)[TARGET_COLUMN]
        .apply(lambda s: pd.Series(_series_stats(s)))
        .unstack()
        .reset_index()
    )

    runtime_by_signature_df = (
        clean_df.groupby("size_signature", dropna=False)[TARGET_COLUMN]
        .apply(lambda s: pd.Series(_series_stats(s)))
        .unstack()
        .reset_index()
        .sort_values("median", ascending=False)
        .reset_index(drop=True)
    )
    return runtime_summary_df, runtime_by_stage_df, runtime_by_signature_df


def analyze_instance_level_behavior(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for instance_id, group_df in clean_df.groupby(GROUP_COLUMN):
        runtime = _safe_numeric(group_df[TARGET_COLUMN])
        best_idx = runtime.idxmin()
        worst_idx = runtime.idxmax()

        baseline_row = group_df.loc[group_df[BUNDLE_COLUMN] == BASELINE_BUNDLE]
        baseline_runtime = (
            float(_safe_numeric(baseline_row[TARGET_COLUMN]).iloc[0]) if not baseline_row.empty else np.nan
        )
        best_runtime = float(runtime.loc[best_idx])
        mean_runtime = float(runtime.mean())

        rows.append(
            {
                GROUP_COLUMN: instance_id,
                "bundle_count": int(len(group_df)),
                "best_bundle": group_df.loc[best_idx, BUNDLE_COLUMN],
                "worst_bundle": group_df.loc[worst_idx, BUNDLE_COLUMN],
                "best_runtime": best_runtime,
                "worst_runtime": float(runtime.loc[worst_idx]),
                "mean_runtime": mean_runtime,
                "runtime_std": float(runtime.std(ddof=0)),
                "runtime_range": float(runtime.max() - runtime.min()),
                "baseline_runtime": baseline_runtime,
                "best_vs_baseline_gain": float(baseline_runtime - best_runtime)
                if not pd.isna(baseline_runtime)
                else np.nan,
                "best_vs_baseline_gain_pct": float((baseline_runtime - best_runtime) / baseline_runtime)
                if not pd.isna(baseline_runtime) and baseline_runtime > 0
                else np.nan,
                "best_vs_mean_gain": float(mean_runtime - best_runtime),
                "scale_phase": group_df["scale_phase"].iloc[0],
                "scale_stage": group_df["scale_stage"].iloc[0],
                "size_signature": group_df["size_signature"].iloc[0],
            }
        )

    instance_summary_df = pd.DataFrame(rows)
    instance_overview_df = pd.DataFrame(
        [
            {
                "instance_count": int(len(instance_summary_df)),
                "avg_best_runtime": float(instance_summary_df["best_runtime"].mean()),
                "avg_runtime_range": float(instance_summary_df["runtime_range"].mean()),
                "median_runtime_range": float(instance_summary_df["runtime_range"].median()),
                "avg_best_vs_baseline_gain": float(instance_summary_df["best_vs_baseline_gain"].mean()),
                "median_best_vs_baseline_gain": float(instance_summary_df["best_vs_baseline_gain"].median()),
                "avg_best_vs_baseline_gain_pct": float(
                    instance_summary_df["best_vs_baseline_gain_pct"].dropna().mean()
                ),
            }
        ]
    )
    return instance_summary_df, instance_overview_df


def analyze_bundle_performance(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle_bits_df = _parse_bundle_series(clean_df[BUNDLE_COLUMN])
    enriched_df = pd.concat([clean_df.reset_index(drop=True), bundle_bits_df], axis=1)
    enriched_df[TARGET_COLUMN] = _safe_numeric(enriched_df[TARGET_COLUMN])

    best_runtime_by_instance = enriched_df.groupby(GROUP_COLUMN)[TARGET_COLUMN].transform("min")
    enriched_df["is_instance_best"] = np.isclose(enriched_df[TARGET_COLUMN], best_runtime_by_instance)
    enriched_df["runtime_rank_in_instance"] = enriched_df.groupby(GROUP_COLUMN)[TARGET_COLUMN].rank(
        method="min", ascending=True
    )

    bundle_summary_df = (
        enriched_df.groupby(BUNDLE_COLUMN, dropna=False)
        .agg(
            sample_count=(TARGET_COLUMN, "size"),
            avg_runtime=(TARGET_COLUMN, "mean"),
            median_runtime=(TARGET_COLUMN, "median"),
            std_runtime=(TARGET_COLUMN, "std"),
            win_count=("is_instance_best", "sum"),
            avg_rank=("runtime_rank_in_instance", "mean"),
        )
        .reset_index()
    )
    instance_count = enriched_df[GROUP_COLUMN].nunique()
    bundle_summary_df["win_rate"] = bundle_summary_df["win_count"] / max(instance_count, 1)
    bundle_summary_df = bundle_summary_df.sort_values(
        ["win_count", "avg_runtime"], ascending=[False, True]
    ).reset_index(drop=True)

    bundle_stage_summary_df = (
        enriched_df.groupby(["scale_stage", BUNDLE_COLUMN], dropna=False)
        .agg(
            avg_runtime=(TARGET_COLUMN, "mean"),
            median_runtime=(TARGET_COLUMN, "median"),
            win_count=("is_instance_best", "sum"),
        )
        .reset_index()
        .sort_values(["scale_stage", "win_count", "avg_runtime"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    effect_rows = []
    for bit_column in [f"bundle_bit_{i}" for i in range(1, 7)]:
        current = enriched_df.groupby(bit_column)[TARGET_COLUMN].mean()
        effect_rows.append(
            {
                "feature": bit_column,
                "runtime_when_0": float(current.get(0, np.nan)),
                "runtime_when_1": float(current.get(1, np.nan)),
                "delta_1_minus_0": float(current.get(1, np.nan) - current.get(0, np.nan)),
            }
        )

    enriched_df["root_active_count"] = enriched_df[[f"bundle_bit_{i}" for i in range(1, 4)]].sum(axis=1)
    enriched_df["nonroot_active_count"] = enriched_df[[f"bundle_bit_{i}" for i in range(4, 7)]].sum(axis=1)
    root_nonroot_df = (
        enriched_df.groupby(["root_active_count", "nonroot_active_count"])[TARGET_COLUMN]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COLUMN: "avg_runtime"})
    )
    bundle_effect_df = pd.concat([pd.DataFrame(effect_rows), root_nonroot_df], ignore_index=True, sort=False)
    return bundle_summary_df, bundle_stage_summary_df, bundle_effect_df


def _candidate_scalar_features(df: pd.DataFrame) -> pd.DataFrame:
    excluded_columns = {
        TARGET_COLUMN,
        GROUP_COLUMN,
        BUNDLE_COLUMN,
        "instance_id",
        "序号",
    }
    numeric_feature_map: dict[str, pd.Series] = {}
    for column in df.columns:
        if column in excluded_columns:
            continue
        numeric_series = _safe_numeric(df[column])
        valid_ratio = float(numeric_series.notna().mean())
        if valid_ratio < 0.95:
            continue
        if numeric_series.nunique(dropna=True) <= 1:
            continue
        numeric_feature_map[column] = numeric_series
    return pd.DataFrame(numeric_feature_map)


def analyze_feature_signal(
    augmented_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_features_df = _candidate_scalar_features(augmented_df)
    target = _safe_numeric(augmented_df[TARGET_COLUMN])
    best_runtime_by_instance = augmented_df.groupby(GROUP_COLUMN)[TARGET_COLUMN].transform("min")
    optimal_flag = np.isclose(target, _safe_numeric(best_runtime_by_instance)).astype(int)

    scalar_summary_rows = []
    target_signal_rows = []
    optimal_signal_rows = []
    for column in numeric_features_df.columns:
        feature = numeric_features_df[column]
        scalar_summary_rows.append(
            {
                "feature": column,
                "valid_ratio": float(feature.notna().mean()),
                "nunique": int(feature.nunique(dropna=True)),
                "mean": float(feature.mean()),
                "std": float(feature.std(ddof=0)),
            }
        )
        joined_target = pd.concat([feature, target], axis=1).dropna()
        joined_optimal = pd.concat([feature, pd.Series(optimal_flag, index=feature.index)], axis=1).dropna()
        target_signal_rows.append(
            {
                "feature": column,
                "pearson_with_runtime": float(joined_target.iloc[:, 0].corr(joined_target.iloc[:, 1])),
                "abs_pearson_with_runtime": float(abs(joined_target.iloc[:, 0].corr(joined_target.iloc[:, 1]))),
            }
        )
        optimal_signal_rows.append(
            {
                "feature": column,
                "pearson_with_optimal_flag": float(joined_optimal.iloc[:, 0].corr(joined_optimal.iloc[:, 1])),
                "abs_pearson_with_optimal_flag": float(
                    abs(joined_optimal.iloc[:, 0].corr(joined_optimal.iloc[:, 1]))
                ),
            }
        )

    feature_scalar_summary_df = pd.DataFrame(scalar_summary_rows).sort_values(
        "valid_ratio", ascending=False
    ).reset_index(drop=True)
    feature_target_signal_df = pd.DataFrame(target_signal_rows).sort_values(
        "abs_pearson_with_runtime", ascending=False
    ).reset_index(drop=True)
    feature_optimal_signal_df = pd.DataFrame(optimal_signal_rows).sort_values(
        "abs_pearson_with_optimal_flag", ascending=False
    ).reset_index(drop=True)
    return feature_target_signal_df, feature_optimal_signal_df, feature_scalar_summary_df


def _policy_per_instance(
    df: pd.DataFrame,
    *,
    selected_bundle_by_instance: pd.Series,
    policy_name: str,
) -> pd.DataFrame:
    rows = []
    runtime_df = df.copy()
    runtime_df[TARGET_COLUMN] = _safe_numeric(runtime_df[TARGET_COLUMN])
    for instance_id, group_df in runtime_df.groupby(GROUP_COLUMN):
        best_row = group_df.loc[group_df[TARGET_COLUMN].idxmin()]
        selected_bundle = selected_bundle_by_instance.get(instance_id)
        if selected_bundle is None:
            continue
        selected_rows = group_df.loc[group_df[BUNDLE_COLUMN] == selected_bundle]
        if selected_rows.empty:
            continue
        selected_row = selected_rows.iloc[0]
        rows.append(
            {
                GROUP_COLUMN: instance_id,
                "policy": policy_name,
                "selected_bundle": selected_bundle,
                "selected_runtime": float(selected_row[TARGET_COLUMN]),
                "oracle_runtime": float(best_row[TARGET_COLUMN]),
                "regret": float(selected_row[TARGET_COLUMN] - best_row[TARGET_COLUMN]),
                "top1_hit": int(selected_bundle == best_row[BUNDLE_COLUMN]),
            }
        )
    return pd.DataFrame(rows)


def analyze_evaluation_readiness(
    clean_df: pd.DataFrame,
    *,
    test_size: float = 0.33,
    random_state: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(clean_df, groups=clean_df[GROUP_COLUMN]))

    split_df = clean_df[[GROUP_COLUMN, "scale_phase", "scale_stage"]].copy()
    split_df["split"] = "train"
    split_df.iloc[test_idx, split_df.columns.get_loc("split")] = "test"

    split_summary_df = pd.DataFrame(
        [
            {
                "train_row_count": int(len(train_idx)),
                "test_row_count": int(len(test_idx)),
                "train_instance_count": int(clean_df.iloc[train_idx][GROUP_COLUMN].nunique()),
                "test_instance_count": int(clean_df.iloc[test_idx][GROUP_COLUMN].nunique()),
            }
        ]
    )
    split_stage_distribution_df = (
        split_df.groupby(["split", "scale_phase", "scale_stage"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )

    runtime_df = clean_df.copy()
    runtime_df[TARGET_COLUMN] = _safe_numeric(runtime_df[TARGET_COLUMN])
    global_fastest_bundle = (
        runtime_df.groupby(BUNDLE_COLUMN)[TARGET_COLUMN].mean().sort_values().index[0]
    )
    global_fastest_selection = pd.Series(
        global_fastest_bundle, index=runtime_df[GROUP_COLUMN].drop_duplicates().sort_values()
    )
    baseline_selection = pd.Series(
        BASELINE_BUNDLE, index=runtime_df[GROUP_COLUMN].drop_duplicates().sort_values()
    )

    rng = np.random.default_rng(random_state)
    random_selection = (
        runtime_df.groupby(GROUP_COLUMN)[BUNDLE_COLUMN]
        .apply(lambda s: s.iloc[int(rng.integers(0, len(s)))])
        .sort_index()
    )

    policy_case_dfs = [
        _policy_per_instance(runtime_df, selected_bundle_by_instance=baseline_selection, policy_name="baseline_bundle"),
        _policy_per_instance(
            runtime_df,
            selected_bundle_by_instance=global_fastest_selection,
            policy_name="global_fastest_bundle",
        ),
        _policy_per_instance(
            runtime_df,
            selected_bundle_by_instance=random_selection,
            policy_name="random_bundle",
        ),
    ]
    baseline_policy_df = pd.concat(policy_case_dfs, ignore_index=True)
    baseline_policy_summary_df = (
        baseline_policy_df.groupby("policy")
        .agg(
            instance_count=(GROUP_COLUMN, "nunique"),
            avg_selected_runtime=("selected_runtime", "mean"),
            avg_regret=("regret", "mean"),
            median_regret=("regret", "median"),
            top1_hit_rate=("top1_hit", "mean"),
        )
        .reset_index()
        .sort_values("avg_regret", ascending=True)
        .reset_index(drop=True)
    )
    return split_summary_df, split_stage_distribution_df, baseline_policy_summary_df


def _plot_data_quality(
    summary_df: pd.DataFrame,
    structured_length_check_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}

    row = summary_df.iloc[0]
    plt.figure(figsize=(8, 4.5))
    labels = [
        "rows",
        "instances",
        "complete_instances",
        "incomplete_instances",
        "duplicate_rows",
    ]
    values = [
        row["row_count"],
        row["instance_count"],
        row["complete_instance_count"],
        row["incomplete_instance_count"],
        row["duplicate_row_count"],
    ]
    plt.bar(labels, values, color=["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b"])
    plt.title("Data Quality Overview")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plot_paths["data_quality_overview"] = _save_current_figure(output_dir / "01_data_quality_overview.png")

    plt.figure(figsize=(10, 4.5))
    plt.bar(
        structured_length_check_df["column"],
        structured_length_check_df["mismatch_rate"],
        color="#457b9d",
    )
    plt.title("Structured Field Length Mismatch Rate")
    plt.ylabel("Mismatch Rate")
    plt.xticks(rotation=45, ha="right")
    plot_paths["structured_length_mismatch"] = _save_current_figure(
        output_dir / "01_structured_length_mismatch.png"
    )

    return plot_paths


def _plot_runtime_distribution(
    clean_df: pd.DataFrame,
    runtime_by_stage_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}
    runtime = _safe_numeric(clean_df[TARGET_COLUMN]).dropna()

    plt.figure(figsize=(8, 4.5))
    plt.hist(runtime, bins=40, color="#2a9d8f", edgecolor="white")
    plt.title("Full Runtime Distribution")
    plt.xlabel("full_runtime")
    plt.ylabel("Frequency")
    plot_paths["runtime_histogram"] = _save_current_figure(output_dir / "02_runtime_histogram.png")

    stage_df = runtime_by_stage_df.copy()
    stage_df["stage_label"] = stage_df["scale_phase"].astype(str) + "|" + stage_df["scale_stage"].astype(str)
    plt.figure(figsize=(9, 4.5))
    plt.bar(stage_df["stage_label"], stage_df["median"], color="#264653")
    plt.title("Median Runtime by Stage")
    plt.ylabel("Median full_runtime")
    plt.xticks(rotation=30, ha="right")
    plot_paths["runtime_by_stage"] = _save_current_figure(output_dir / "02_runtime_by_stage.png")

    return plot_paths


def _plot_instance_behavior(
    instance_summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}

    plt.figure(figsize=(8, 4.5))
    plt.hist(instance_summary_df["runtime_range"].dropna(), bins=40, color="#f4a261", edgecolor="white")
    plt.title("Instance Runtime Range Distribution")
    plt.xlabel("runtime_range")
    plt.ylabel("Frequency")
    plot_paths["instance_runtime_range"] = _save_current_figure(output_dir / "03_instance_runtime_range.png")

    scatter_df = instance_summary_df.dropna(subset=["baseline_runtime", "best_runtime"]).copy()
    plt.figure(figsize=(6, 6))
    plt.scatter(scatter_df["baseline_runtime"], scatter_df["best_runtime"], alpha=0.6, color="#e76f51")
    diagonal_max = max(scatter_df["baseline_runtime"].max(), scatter_df["best_runtime"].max())
    plt.plot([0, diagonal_max], [0, diagonal_max], linestyle="--", color="gray")
    plt.title("Baseline vs Best Runtime")
    plt.xlabel("Baseline Runtime")
    plt.ylabel("Best Runtime")
    plot_paths["baseline_vs_best_runtime"] = _save_current_figure(
        output_dir / "03_baseline_vs_best_runtime.png"
    )

    return plot_paths


def _plot_bundle_performance(
    bundle_summary_df: pd.DataFrame,
    bundle_effect_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}

    top_bundle_df = bundle_summary_df.head(10).copy()
    plt.figure(figsize=(10, 4.8))
    plt.bar(top_bundle_df[BUNDLE_COLUMN], top_bundle_df["win_rate"], color="#6a994e")
    plt.title("Top Bundle Win Rates")
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45, ha="right")
    plot_paths["bundle_win_rate"] = _save_current_figure(output_dir / "04_bundle_win_rate.png")

    bit_effect_df = bundle_effect_df.loc[bundle_effect_df["feature"].notna()].copy()
    plt.figure(figsize=(8, 4.5))
    plt.bar(bit_effect_df["feature"], bit_effect_df["delta_1_minus_0"], color="#bc4749")
    plt.axhline(0, color="gray", linewidth=1)
    plt.title("Bundle Bit Runtime Effect (1 - 0)")
    plt.ylabel("Delta Runtime")
    plot_paths["bundle_bit_effect"] = _save_current_figure(output_dir / "04_bundle_bit_effect.png")

    return plot_paths


def _plot_feature_signal(
    feature_target_signal_df: pd.DataFrame,
    feature_optimal_signal_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}

    top_runtime_df = feature_target_signal_df.head(10).copy()
    plt.figure(figsize=(10, 4.8))
    plt.bar(top_runtime_df["feature"], top_runtime_df["abs_pearson_with_runtime"], color="#577590")
    plt.title("Top Feature Signal to Runtime")
    plt.ylabel("Absolute Pearson Correlation")
    plt.xticks(rotation=45, ha="right")
    plot_paths["feature_signal_runtime"] = _save_current_figure(output_dir / "05_feature_signal_runtime.png")

    top_optimal_df = feature_optimal_signal_df.head(10).copy()
    plt.figure(figsize=(10, 4.8))
    plt.bar(top_optimal_df["feature"], top_optimal_df["abs_pearson_with_optimal_flag"], color="#4d908e")
    plt.title("Top Feature Signal to Optimal Flag")
    plt.ylabel("Absolute Pearson Correlation")
    plt.xticks(rotation=45, ha="right")
    plot_paths["feature_signal_optimal"] = _save_current_figure(output_dir / "05_feature_signal_optimal.png")

    return plot_paths


def _plot_evaluation_readiness(
    split_stage_distribution_df: pd.DataFrame,
    baseline_policy_summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}

    pivot = (
        split_stage_distribution_df.pivot_table(
            index="scale_stage", columns="split", values="row_count", aggfunc="sum", fill_value=0
        )
        .sort_index()
    )
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.35
    plt.bar(x - width / 2, pivot.get("train", pd.Series(index=pivot.index, dtype=float)), width, label="train")
    plt.bar(x + width / 2, pivot.get("test", pd.Series(index=pivot.index, dtype=float)), width, label="test")
    plt.title("Grouped Split Stage Distribution")
    plt.ylabel("Row Count")
    plt.xticks(x, pivot.index)
    plt.legend()
    plot_paths["split_stage_distribution"] = _save_current_figure(
        output_dir / "06_split_stage_distribution.png"
    )

    plt.figure(figsize=(8, 4.5))
    plt.bar(baseline_policy_summary_df["policy"], baseline_policy_summary_df["avg_regret"], color="#9c6644")
    plt.title("Baseline Policy Average Regret")
    plt.ylabel("Average Regret")
    plt.xticks(rotation=20)
    plot_paths["baseline_policy_regret"] = _save_current_figure(output_dir / "06_baseline_policy_regret.png")

    return plot_paths


def _build_report_text(
    artifacts: AnalysisArtifacts,
) -> str:
    quality_row = artifacts.data_quality_summary_df.iloc[0]
    runtime_row = artifacts.runtime_summary_df.iloc[0]
    instance_row = artifacts.instance_overview_df.iloc[0]

    bundle_top = artifacts.bundle_summary_df.head(5)[[BUNDLE_COLUMN, "win_count", "win_rate", "avg_runtime"]]
    feature_top = artifacts.feature_target_signal_df.head(10)[
        ["feature", "pearson_with_runtime", "abs_pearson_with_runtime"]
    ]
    policy_top = artifacts.baseline_policy_summary_df.copy()

    lines = [
        "# 数据分析报告",
        "",
        "## 1. 数据质量分析",
        f"- 样本行数: {int(quality_row['row_count'])}",
        f"- 实例数: {int(quality_row['instance_count'])}",
        f"- 完整实例数: {int(quality_row['complete_instance_count'])}",
        f"- 不完整实例数: {int(quality_row['incomplete_instance_count'])}",
        "",
        "## 2. 标签分布分析",
        f"- 标签列: `{TARGET_COLUMN}`",
        f"- 平均值: {runtime_row['mean']:.6f}",
        f"- 中位数: {runtime_row['median']:.6f}",
        f"- 75%分位: {runtime_row['p75']:.6f}",
        f"- 最大值: {runtime_row['max']:.6f}",
        "",
        "## 3. 实例级分析",
        f"- 平均最优runtime: {instance_row['avg_best_runtime']:.6f}",
        f"- 平均实例内极差: {instance_row['avg_runtime_range']:.6f}",
        f"- 平均相对baseline收益: {instance_row['avg_best_vs_baseline_gain_pct']:.6f}",
        "",
        "## 4. Bundle表现分析",
        bundle_top.to_string(index=False),
        "",
        "## 5. 特征信号分析",
        feature_top.to_string(index=False),
        "",
        "## 6. 评估前分析",
        policy_top.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def run_full_data_analysis(
    clean_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    output_dir: Path | str,
    *,
    test_size: float = 0.33,
    random_state: int = 7,
) -> AnalysisArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots"
    report_dir = output_dir / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}

    data_quality_summary_df, structured_length_check_df = analyze_data_quality(clean_df)
    runtime_summary_df, runtime_by_stage_df, runtime_by_signature_df = analyze_runtime_distribution(clean_df)
    instance_summary_df, instance_overview_df = analyze_instance_level_behavior(clean_df)
    bundle_summary_df, bundle_stage_summary_df, bundle_effect_df = analyze_bundle_performance(clean_df)
    (
        feature_target_signal_df,
        feature_optimal_signal_df,
        feature_scalar_summary_df,
    ) = analyze_feature_signal(augmented_df)
    (
        split_summary_df,
        split_stage_distribution_df,
        baseline_policy_summary_df,
    ) = analyze_evaluation_readiness(clean_df, test_size=test_size, random_state=random_state)

    _write_csv(data_quality_summary_df, table_dir / "01_data_quality_summary.csv")
    _write_csv(structured_length_check_df, table_dir / "01_structured_length_checks.csv")
    _write_csv(runtime_summary_df, table_dir / "02_runtime_summary.csv")
    _write_csv(runtime_by_stage_df, table_dir / "02_runtime_by_stage.csv")
    _write_csv(runtime_by_signature_df, table_dir / "02_runtime_by_signature.csv")
    _write_csv(instance_summary_df, table_dir / "03_instance_summary.csv")
    _write_csv(instance_overview_df, table_dir / "03_instance_overview.csv")
    _write_csv(bundle_summary_df, table_dir / "04_bundle_summary.csv")
    _write_csv(bundle_stage_summary_df, table_dir / "04_bundle_stage_summary.csv")
    _write_csv(bundle_effect_df, table_dir / "04_bundle_effect_summary.csv")
    _write_csv(feature_target_signal_df, table_dir / "05_feature_signal_to_runtime.csv")
    _write_csv(feature_optimal_signal_df, table_dir / "05_feature_signal_to_optimal.csv")
    _write_csv(feature_scalar_summary_df, table_dir / "05_feature_scalar_summary.csv")
    _write_csv(split_summary_df, table_dir / "06_split_summary.csv")
    _write_csv(split_stage_distribution_df, table_dir / "06_split_stage_distribution.csv")
    _write_csv(baseline_policy_summary_df, table_dir / "06_baseline_policy_summary.csv")

    plot_paths.update(_plot_data_quality(data_quality_summary_df, structured_length_check_df, plot_dir))
    plot_paths.update(_plot_runtime_distribution(clean_df, runtime_by_stage_df, plot_dir))
    plot_paths.update(_plot_instance_behavior(instance_summary_df, plot_dir))
    plot_paths.update(_plot_bundle_performance(bundle_summary_df, bundle_effect_df, plot_dir))
    plot_paths.update(_plot_feature_signal(feature_target_signal_df, feature_optimal_signal_df, plot_dir))
    plot_paths.update(
        _plot_evaluation_readiness(split_stage_distribution_df, baseline_policy_summary_df, plot_dir)
    )

    artifacts = AnalysisArtifacts(
        output_dir=output_dir,
        table_dir=table_dir,
        plot_dir=plot_dir,
        report_dir=report_dir,
        data_quality_summary_df=data_quality_summary_df,
        runtime_summary_df=runtime_summary_df,
        runtime_by_stage_df=runtime_by_stage_df,
        instance_summary_df=instance_summary_df,
        instance_overview_df=instance_overview_df,
        bundle_summary_df=bundle_summary_df,
        bundle_stage_summary_df=bundle_stage_summary_df,
        bundle_effect_df=bundle_effect_df,
        feature_target_signal_df=feature_target_signal_df,
        feature_optimal_signal_df=feature_optimal_signal_df,
        feature_scalar_summary_df=feature_scalar_summary_df,
        split_summary_df=split_summary_df,
        split_stage_distribution_df=split_stage_distribution_df,
        baseline_policy_summary_df=baseline_policy_summary_df,
        report_path=report_dir / "analysis_report.md",
        plot_paths=plot_paths,
    )

    artifacts.report_path.write_text(_build_report_text(artifacts), encoding="utf-8")
    return artifacts
