"""Model builders for runtime prediction experiments."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def make_lgbm_regressor():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        colsample_bytree=0.7,
        learning_rate=0.2,
        max_depth=3,
        n_estimators=200,
        n_jobs=-1,
        random_state=0,
        subsample=0.85,
        reg_alpha=0,
        reg_lambda=0.01,
    )


def make_xgb_regressor():
    from xgboost import XGBRegressor

    return XGBRegressor(
        colsample_bytree=0.7,
        reg_lambda=0.01,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=2,
        n_estimators=200,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=0,
    )


def make_random_forest_regressor():
    return RandomForestRegressor(n_estimators=200, random_state=7, n_jobs=-1)


def make_gradient_boosting_regressor():
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=7)


def make_svr_regressor():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR()),
        ]
    )


def make_linear_regressor():
    return LinearRegression()


MODEL_BUILDERS = {
    "lightgbm": make_lgbm_regressor,
    "xgboost": make_xgb_regressor,
    "random_forest": make_random_forest_regressor,
    "gradient_boosting": make_gradient_boosting_regressor,
    "svr": make_svr_regressor,
    "linear_regression": make_linear_regressor,
}
