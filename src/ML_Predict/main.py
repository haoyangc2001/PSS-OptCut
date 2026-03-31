"""CLI entry point for the modular runtime-prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_loading import DATA_DIR, DEFAULT_RESULT_CSV, load_clean_result_csv
from .dataset_builder import build_augmented_dataset, build_model_inputs
from .experiments import run_baseline_experiments, run_manual_lightgbm_selection_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular ML runtime-prediction pipeline.")
    parser.add_argument("--result-csv", type=Path, default=DEFAULT_RESULT_CSV, help="Input result.csv path.")
    parser.add_argument(
        "--derived-csv",
        type=Path,
        default=DATA_DIR / "总数据.csv",
        help="Path for the derived augmented dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "ml_predict_outputs",
        help="Directory for evaluation summaries.",
    )
    parser.add_argument("--min-runtime", type=float, default=0.5, help="Minimum runtime filter.")
    parser.add_argument("--test-size", type=float, default=0.33, help="Holdout split ratio.")
    parser.add_argument("--random-state", type=int, default=7, help="Random seed for holdout splits.")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only build the derived dataset without running model experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.derived_csv.parent.mkdir(parents=True, exist_ok=True)

    clean_df = load_clean_result_csv(args.result_csv, min_runtime=args.min_runtime)
    augmented_df = build_augmented_dataset(clean_df)
    augmented_df.to_csv(args.derived_csv, index=False)

    print(f"Clean samples: {len(clean_df)}")
    print(f"Derived dataset saved to: {args.derived_csv}")

    if args.skip_training:
        return

    X_df, y_series, meta_df = build_model_inputs(augmented_df)

    baseline_artifacts = run_baseline_experiments(
        X_df,
        y_series,
        meta_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    baseline_artifacts.summary_df.to_csv(args.output_dir / "baseline_model_summary.csv", index=False)
    baseline_artifacts.grouped_task_summary_df.to_csv(
        args.output_dir / "baseline_grouped_task_summary.csv",
        index=False,
    )
    for model_name, case_df in baseline_artifacts.grouped_case_dfs.items():
        case_df.to_csv(args.output_dir / f"baseline_{model_name}_grouped_cases.csv", index=False)
    for model_name, importance_df in baseline_artifacts.importances.items():
        importance_df.to_csv(args.output_dir / f"{model_name}_importance.csv", index=False)
    if baseline_artifacts.errors:
        error_rows = [{"model": key, "error": value} for key, value in baseline_artifacts.errors.items()]
        import pandas as pd

        pd.DataFrame(error_rows).to_csv(args.output_dir / "baseline_errors.csv", index=False)

    selection_artifacts = run_manual_lightgbm_selection_experiments(
        X_df,
        y_series,
        meta_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    selection_artifacts.summary_df.to_csv(args.output_dir / "lightgbm_manual_selection_summary.csv", index=False)
    selection_artifacts.grouped_task_summary_df.to_csv(
        args.output_dir / "lightgbm_manual_selection_grouped_task_summary.csv",
        index=False,
    )
    for scheme_name, case_df in selection_artifacts.grouped_case_dfs.items():
        case_df.to_csv(args.output_dir / f"lightgbm_{scheme_name}_grouped_cases.csv", index=False)
    for scheme_name, importance_df in selection_artifacts.importances.items():
        importance_df.to_csv(args.output_dir / f"lightgbm_{scheme_name}_importance.csv", index=False)
    if selection_artifacts.errors:
        error_rows = [{"scheme": key, "error": value} for key, value in selection_artifacts.errors.items()]
        import pandas as pd

        pd.DataFrame(error_rows).to_csv(args.output_dir / "lightgbm_manual_selection_errors.csv", index=False)

    print(f"Experiment outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
