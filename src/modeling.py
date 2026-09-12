from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .bte_calculation import build_bte_calculation_metadata, calculate_bte
from .config import (
    COMPOSITION_FEATURES,
    MODEL_FEATURES,
    PERFORMANCE_TARGETS,
    PROPERTY_TARGETS,
    RANDOM_STATE,
)


DEEP_HOLDOUT_RANDOM_STATE = 2026
DEEP_HIDDEN_LAYERS = (128, 128, 64, 32)


def deep_performance_model(target: str) -> TransformedTargetRegressor:
    """Create the target-specific four-hidden-layer neural regressor."""

    if target not in PERFORMANCE_TARGETS:
        raise ValueError(f"Unsupported performance target: {target}")
    if target == "co_vol_pct":
        feature_transformer: Any = StandardScaler()
    else:
        feature_transformer = ColumnTransformer(
            [
                ("scaled_numeric", StandardScaler(), MODEL_FEATURES),
                (
                    "operating_levels",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["speed_rpm", "compression_ratio"],
                ),
            ]
        )
    activation = "relu" if target == "bte_pct" else "tanh"
    learning_rate = 0.001 if target == "bte_pct" else 0.0007
    network = MLPRegressor(
        hidden_layer_sizes=DEEP_HIDDEN_LAYERS,
        activation=activation,
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate_init=learning_rate,
        max_iter=2200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=120,
        tol=1e-5,
        random_state=RANDOM_STATE,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("features", feature_transformer),
                ("network", network),
            ]
        ),
        transformer=StandardScaler(),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _feature_importance(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=8,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    values = np.maximum(result.importances_mean, 0.0)
    total = values.sum()
    return values / total if total > 0 else np.repeat(1.0 / len(values), len(values))


def _nearest_support_threshold(compositions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mean = compositions.mean(axis=0)
    scale = compositions.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (compositions - mean) / scale
    distances = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    threshold = max(float(np.quantile(nearest, 0.95)), 0.75)
    return mean, scale, threshold


def train_models(data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Validate deep regressors on unseen blend groups, then refit all data."""

    x = data[MODEL_FEATURES].copy()
    groups = data["blend_group"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=DEEP_HOLDOUT_RANDOM_STATE,
    )
    dev_index, test_index = next(splitter.split(x, groups=groups))
    x_dev, x_test = x.iloc[dev_index], x.iloc[test_index]
    groups_dev = groups.iloc[dev_index]
    fold_count = min(5, groups_dev.nunique())
    grouped_cv = GroupKFold(n_splits=fold_count)

    cv_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    selected_models: dict[str, Any] = {}
    error_quantiles: dict[str, float] = {}

    bte_calculation = build_bte_calculation_metadata(data)

    for target in PERFORMANCE_TARGETS:
        y = data[target].astype(float)
        y_dev, y_test = y.iloc[dev_index], y.iloc[test_index]
        prototype = deep_performance_model(target)
        model_name = "Deep Neural Network (128-128-64-32)"
        for fold, (train_fold, val_fold) in enumerate(
            grouped_cv.split(x_dev, y_dev, groups_dev), start=1
        ):
            model = clone(prototype)
            model.fit(x_dev.iloc[train_fold], y_dev.iloc[train_fold])
            prediction = model.predict(x_dev.iloc[val_fold])
            metrics = regression_metrics(y_dev.iloc[val_fold].to_numpy(), prediction)
            cv_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "fold": fold,
                    **metrics,
                }
            )

        holdout_model = clone(prototype).fit(x_dev, y_dev)
        holdout_prediction = holdout_model.predict(x_test)
        metrics = regression_metrics(y_test.to_numpy(), holdout_prediction)
        abs_error = np.abs(y_test.to_numpy() - holdout_prediction)
        error_quantiles[target] = float(np.quantile(abs_error, 0.90))
        holdout_rows.append(
            {
                "target": target,
                "selected_model": model_name,
                "test_rows": int(len(test_index)),
                "test_blend_groups": int(groups.iloc[test_index].nunique()),
                "absolute_error_p90": error_quantiles[target],
                **metrics,
            }
        )

        importance = _feature_importance(holdout_model, x_test, y_test)
        for feature, value in zip(MODEL_FEATURES, importance):
            importance_rows.append(
                {"target": target, "feature": feature, "importance": float(value)}
            )

        target_predictions = pd.DataFrame(
            {
                "source_excel_row": data.iloc[test_index]["source_excel_row"].to_numpy(),
                "blend_group": groups.iloc[test_index].to_numpy(),
                "compression_ratio": x_test["compression_ratio"].to_numpy(),
                "target": target,
                "actual": y_test.to_numpy(),
                "predicted_raw": holdout_prediction,
            }
        )
        if target == "bte_pct":
            traces = [
                calculate_bte(
                    raw_bte_pct=prediction,
                    compression_ratio=float(row["compression_ratio"]),
                    blend=row,
                    calculation=bte_calculation,
                )
                for prediction, (_, row) in zip(holdout_prediction, x_test.iterrows())
            ]
            target_predictions["predicted_output"] = [trace["final_bte_pct"] for trace in traces]
            target_predictions["bte_rule_id"] = [trace["rule_id"] for trace in traces]
        else:
            target_predictions["predicted_output"] = np.clip(
                holdout_prediction,
                float(data[target].min()),
                float(data[target].max()),
            )
        prediction_rows.append(target_predictions)

        selected_models[target] = clone(prototype).fit(x, y)

    # New blend properties are learned from one median record per unique composition.
    blend_properties = (
        data.groupby("blend_group", as_index=False)[COMPOSITION_FEATURES + PROPERTY_TARGETS]
        .median(numeric_only=True)
        .reset_index(drop=True)
    )
    property_x = blend_properties[COMPOSITION_FEATURES]
    property_y = blend_properties[PROPERTY_TARGETS]
    property_model = ExtraTreesRegressor(
        n_estimators=420,
        min_samples_leaf=1,
        max_features=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    property_model.fit(property_x, property_y)

    property_rows: list[dict[str, Any]] = []
    property_cv = KFold(n_splits=min(5, len(blend_properties)), shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_fold, val_fold) in enumerate(property_cv.split(property_x), start=1):
        model = clone(property_model).fit(property_x.iloc[train_fold], property_y.iloc[train_fold])
        pred = model.predict(property_x.iloc[val_fold])
        for column_index, target in enumerate(PROPERTY_TARGETS):
            metrics = regression_metrics(property_y.iloc[val_fold, column_index].to_numpy(), pred[:, column_index])
            property_rows.append({"property": target, "fold": fold, **metrics})

    unique_compositions = blend_properties[COMPOSITION_FEATURES].to_numpy(dtype=float)
    support_mean, support_scale, support_threshold = _nearest_support_threshold(unique_compositions)

    model_ranges = {
        column: {"min": float(data[column].min()), "max": float(data[column].max())}
        for column in MODEL_FEATURES
    }
    property_ranges = {
        column: {"min": float(data[column].min()), "max": float(data[column].max())}
        for column in PROPERTY_TARGETS
    }
    performance_ranges = {
        column: {"min": float(data[column].min()), "max": float(data[column].max())}
        for column in PERFORMANCE_TARGETS
    }

    bundle = {
        "version": "2.1.0",
        "model_family": "Deep Neural Network",
        "deep_hidden_layers": list(DEEP_HIDDEN_LAYERS),
        "holdout_random_state": DEEP_HOLDOUT_RANDOM_STATE,
        "model_features": MODEL_FEATURES,
        "composition_features": COMPOSITION_FEATURES,
        "performance_targets": PERFORMANCE_TARGETS,
        "property_targets": PROPERTY_TARGETS,
        "performance_models": selected_models,
        "property_model": property_model,
        "error_p90": error_quantiles,
        "model_ranges": model_ranges,
        "performance_ranges": performance_ranges,
        "known_compression_ratios": sorted(float(v) for v in data["compression_ratio"].unique()),
        "known_speeds_rpm": sorted(float(v) for v in data["speed_rpm"].unique()),
        "property_ranges": property_ranges,
        "support_compositions": unique_compositions,
        "support_mean": support_mean,
        "support_scale": support_scale,
        "support_distance_threshold": support_threshold,
        "training_rows": int(len(data)),
        "training_blend_groups": int(data["blend_group"].nunique()),
        "bte_calculation": bte_calculation,
        "methodology_notes": [
            "All four performance targets use four-hidden-layer deep neural networks with standardized inputs and targets.",
            "The final holdout contains complete unseen fuel-blend groups and uses random state 2026.",
            "Fuel properties are estimated separately so arbitrary property values cannot be injected into optimization.",
            "Grouped cross-validation evaluates the deep models without mixing the same fuel blend across folds.",
            "For CR >= 9.5, the internal DNN BTE signal is normalized within the cleaned dataset envelope at that CR and mapped into the applicable user-supplied blend range.",
            "A global 37% BTE ceiling applies whenever n-Methylaniline is present, including below CR 9.5.",
            "Prediction, Optimization, and Simulation all call one canonical inference function.",
            "Unsupported engine inputs are bounded to the nearest trained edge, and reported performance outputs are bounded to measured target envelopes to prevent impossible extrapolation artifacts.",
            "The final rule-guided BTE is a hybrid decision-support calculation; raw holdout metrics do not validate the user-supplied range rules.",
        ],
    }

    reports = {
        "cv_metrics": pd.DataFrame(cv_rows),
        "holdout_metrics": pd.DataFrame(holdout_rows),
        "holdout_predictions": pd.concat(prediction_rows, ignore_index=True),
        "feature_importance": pd.DataFrame(importance_rows),
        "property_cv_metrics": pd.DataFrame(property_rows),
    }
    return bundle, reports


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
