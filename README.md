# PSS-OptCut

PSS-OptCut is an experimental repository for joint optimization of product-service system (PSS) configuration and delivery. It combines service-level selection, multi-team routing, and customer time-window scheduling in a mixed-integer linear programming (MILP) model, and then studies how different valid-cut bundles affect Gurobi performance at different branch-and-bound stages.

The repository also contains a modular machine-learning pipeline that turns solver logs into a runtime-prediction dataset. The target is not the optimal objective value. The target is solver runtime for a given instance under a given 6-bit cut bundle.

## Current Repository Scope

The current codebase supports two connected workflows:

1. Optimization experiment generation
   - Generate random PSS instances.
   - Enumerate all `2^6 = 64` valid-cut bundles.
   - Solve each bundle on the same instance.
   - Record instance parameters, model size, cut selection, and runtime metrics.

2. Runtime prediction experiments
   - Clean raw solver results.
   - Build an augmented dataset with statistical features and expanded cut-bundle bits.
   - Run baseline regressors and LightGBM-based feature-selection experiments.
   - Export analysis tables and evaluation reports.

## Repository Layout

```text
PSS-OptCut/
├── data/
│   ├── result.csv
│   ├── result_20260331.csv
│   ├── 总数据.csv
│   └── other exported experiment files
├── document/
│   ├── Integrated Optimization of Product-Service System Configuration and Delivery with Differentiated Valid Cut Bundle Selection across Various Branch-and-Bound Solving Stages.pdf
│   └── internal design and business documents
├── notebooks/
│   ├── 整体流程_预测时间 .ipynb
│   └── 整体流程_预测时间 _origin.ipynb
├── src/
│   ├── Gurobi_Solver/
│   │   ├── Pamas_generator.py
│   │   ├── Solver_builder.py
│   │   ├── main.py
│   │   └── sample_generator.py
│   └── ML_Predict/
│       ├── constants.py
│       ├── data_analysis.py
│       ├── data_loading.py
│       ├── dataset_builder.py
│       ├── evaluation.py
│       ├── experiments.py
│       ├── feature_engineering.py
│       ├── main.py
│       └── models.py
├── environment.yml
└── run_rsc.sh
```

## Optimization Workflow

The optimization entry point is:

```bash
python -m src.Gurobi_Solver.main
```

The provided wrapper script configures the expected Python and Gurobi paths before launching the same module:

```bash
bash run_rsc.sh
```

### What the solver currently does

- Uses `src/Gurobi_Solver/main.py` as the top-level experiment runner.
- Generates `400` random instances by default (`TOTAL_TIMES = 400`).
- Splits the experiment into:
  - `baseline_300`: instances `1..300`
  - `stress_100`: instances `301..400`
- For each instance, enumerates all 64 six-bit cut bundles.
- Solves every bundle through `TrainSetGenerator` in `sample_generator.py`.
- Writes results to a fresh dated file: `data/result_YYYYMMDD.csv`.

A full default run therefore produces:

```text
400 instances x 64 bundles = 25,600 rows
```

### Instance generation

`src/Gurobi_Solver/Pamas_generator.py` generates:

- customer count
- service count
- product count
- team count
- service times, prices, and costs
- customer utility matrix
- inventory
- customer weights
- time windows
- transfer-time matrix
- fixed and variable costs
- service-product consumption relations

`Num_Product` is always set equal to `Num_Service` in the current experiment design.

### Current scale schedule

`sample_instance_size()` in `src/Gurobi_Solver/main.py` uses five stages:

| Stage | Instance range | Customers | Services = Products | Teams |
| --- | --- | --- | --- | --- |
| S1 | `1..300` | `5..9` | `3..5` | `3..7` |
| S2 | `301..325` | `10..12` | `5..6` | `6..8` |
| S3 | `326..350` | `12..14` | `6..7` | `7..9` |
| S4 | `351..375` | `14..16` | `7..8` | `8..10` |
| S5 | `376..400` | `16..18` | `8..10` | `9..12` |

### Objective

The current default objective coefficients are:

```python
OBJ_COEF = [1, -1, 1, 0]
```

This combines:

- utility-price performance
- penalty on service start times
- profit
- an inactive fourth auxiliary term

### Valid-cut experiment design

`build_valid_cut_pool()` in `src/Gurobi_Solver/main.py` defines three cut families:

1. Reverse-arc exclusion cut
2. Time-window cut using service duration
3. Time-window cut using earliest start plus minimum service duration

The current main experiment path uses `trigger4` in `src/Gurobi_Solver/Solver_builder.py`.

For the 6-bit bundle:

- bits `1..3`: cuts statically added at the root stage
- bits `4..6`: cuts added later through the MIP-node callback

The solver records:

- `base_num_vars`
- `base_num_constrs`
- `root_num_constrs`
- `求解时间`
- `legacy_runtime`
- `full_runtime`

In the current implementation:

- `求解时间` is the legacy trigger-4 runtime stored in `time_final`
- `legacy_runtime` is the same legacy runtime measure
- `full_runtime` is recomputed from a full trigger-4 solve and is the ML target used by `src/ML_Predict/constants.py`

## Machine-Learning Workflow

The ML entry point is:

```bash
python -m src.ML_Predict.main --result-csv data/result_20260331.csv --run-analysis
```

### Important input-path note

- `src/Gurobi_Solver/main.py` writes fresh results to `data/result_YYYYMMDD.csv`
- `src/ML_Predict/data_loading.py` still defaults to `data/result.csv`

If you want to train on a newly generated solver run, pass `--result-csv` explicitly.

### What the ML pipeline does

`src/ML_Predict/main.py` performs the following steps:

1. Load and clean the raw result file.
2. Remove repeated header rows and duplicate records.
3. Convert runtime and seed fields to numeric form.
4. Filter out rows with runtime at or below the configured threshold.
5. Keep only instances with the complete set of 64 bundles.
6. Build an augmented dataset:
   - notebook-style statistical summaries for serialized arrays and matrices
   - `Feature1` to `Feature6` extracted from the cut bundle
7. Optionally run six analysis modules.
8. Run baseline regressors and manual LightGBM feature-selection experiments.

### Current modeling target

The ML target is:

```python
TARGET_COLUMN = "full_runtime"
```

### Current model set

`src/ML_Predict/models.py` currently defines:

- LightGBM
- XGBoost
- Random Forest
- Gradient Boosting
- SVR
- Linear Regression

### Output locations

By default, ML outputs are written under:

```text
log/YYMMDD/
├── analysis/
├── dataset/derived/总数据.csv
└── evaluate/
    ├── baseline/
    └── manual_lightgbm_selection/
```

## Environment and Dependencies

The base Conda environment file is:

```bash
conda env create -f environment.yml
conda activate PSS
```

`environment.yml` currently includes:

- Python 3.11
- numpy
- pandas
- openpyxl
- gurobipy

The ML pipeline requires additional packages that are imported by the current code but are not listed in `environment.yml`:

```bash
pip install scipy scikit-learn matplotlib lightgbm xgboost
```

If you want to work with notebooks:

```bash
pip install notebook
```

## Gurobi Runtime Requirements

`run_rsc.sh` currently expects:

- Python: `/pub/data/caohy/miniconda/envs/PSS/bin/python`
- Gurobi home: `/home/caohy/app/gurobi/gurobi1301/linux64`
- License file: `/home/caohy/opt/gurobi/gurobi.lic`

If your local paths differ, update the script before running it.

## Data and Documents Included in the Repository

The repository already contains historical experiment artifacts, including:

- `data/result.csv`
- `data/result_20260331.csv`
- `data/总数据.csv`

The `document/` directory contains the paper PDF and several internal project documents. Those supporting documents are currently written mostly in Chinese even though this `README` is now fully in English.

## Notebooks

The repository contains two notebooks:

- `notebooks/整体流程_预测时间 .ipynb`: the active exploratory notebook
- `notebooks/整体流程_预测时间 _origin.ipynb`: backup copy

The modular code in `src/ML_Predict/` is the maintained script-based version of the runtime-prediction workflow.

## Testing Status

There is currently no automated test suite in the repository. Validation is primarily done by:

- running the solver workflow end to end
- checking the generated CSV schema and row counts
- running the ML pipeline on a complete result file

## Quick Start

Create the environment and install the ML extras:

```bash
conda env create -f environment.yml
conda activate PSS
pip install scipy scikit-learn matplotlib lightgbm xgboost
```

Run optimization:

```bash
bash run_rsc.sh
```

Run the ML pipeline on a specific solver export:

```bash
python -m src.ML_Predict.main \
  --result-csv data/result_20260331.csv \
  --run-analysis
```
