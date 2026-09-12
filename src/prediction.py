from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .bte_calculation import calculate_bte
from .config import (
    COMPOSITION_FEATURES,
    ENGINE_FEATURES,
    MODEL_FEATURES,
    PERFORMANCE_TARGETS,
    PROPERTY_TARGETS,
)


class InputValidationError(ValueError):
    """Raised when an app input violates a hard validation rule."""


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    # Keep compatibility with bundles created before the global NMA ceiling
    # was added to the canonical calculation path.
    for rule in bundle.get("bte_calculation", {}).get("additive_rules", []):
        if rule.get("additive") == "n_methylaniline_pct":
            rule["maximum_pct"] = 37.0
    return bundle


def blend_total(blend: dict[str, float]) -> float:
    return float(sum(float(blend.get(name, 0.0)) for name in COMPOSITION_FEATURES))


def validate_blend(blend: dict[str, float], tolerance: float = 0.01) -> None:
    values = np.array([float(blend.get(name, 0.0)) for name in COMPOSITION_FEATURES])
    if not np.isfinite(values).all():
        raise InputValidationError("All blend percentages must be finite numbers.")
    if (values < 0).any() or (values > 100).any():
        raise InputValidationError("Each blend component must be between 0 and 100%.")
    total = float(values.sum())
    if abs(total - 100.0) > tolerance + 1e-9:
        raise InputValidationError(
            f"Fuel composition totals {total:.2f}%. Adjust the components so the total is 100.00%."
        )


def _input_frame(blend: dict[str, float], torque_nm: float, speed_rpm: float, compression_ratio: float) -> pd.DataFrame:
    record = {name: float(blend[name]) for name in COMPOSITION_FEATURES}
    record.update(
        {
            "torque_nm": float(torque_nm),
            "speed_rpm": float(speed_rpm),
            "compression_ratio": float(compression_ratio),
        }
    )
    if not all(np.isfinite(v) for v in record.values()):
        raise InputValidationError("Engine and fuel inputs must be finite numbers.")
    if torque_nm <= 0 or speed_rpm <= 0 or compression_ratio <= 0:
        raise InputValidationError("Torque, speed, and compression ratio must be greater than zero.")
    return pd.DataFrame([record], columns=MODEL_FEATURES)


def _validate_prediction_frame(
    candidates: pd.DataFrame,
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
) -> pd.DataFrame:
    engine_values = np.asarray([torque_nm, speed_rpm, compression_ratio], dtype=float)
    if not np.isfinite(engine_values).all() or (engine_values <= 0.0).any():
        raise InputValidationError(
            "Torque, RPM, and compression ratio must be finite positive values."
        )
    missing = [feature for feature in COMPOSITION_FEATURES if feature not in candidates]
    if missing:
        raise InputValidationError(
            f"Candidate table is missing composition columns: {', '.join(missing)}"
        )
    evaluated = candidates[COMPOSITION_FEATURES].copy().reset_index(drop=True)
    composition_values = evaluated.to_numpy(dtype=float)
    if not np.isfinite(composition_values).all():
        raise InputValidationError("Fuel percentages must be finite numbers.")
    if (composition_values < -1e-9).any() or (composition_values > 100.0 + 1e-9).any():
        raise InputValidationError("Every fuel component must remain within 0-100%.")
    if not np.allclose(evaluated.sum(axis=1), 100.0, atol=0.01, rtol=0.0):
        raise InputValidationError("Every fuel composition must total 100%.")
    return evaluated


def bounded_operating_point(
    bundle: dict[str, Any],
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
) -> tuple[dict[str, float], dict[str, bool]]:
    """Bound unsupported engine inputs to the nearest trained model edge.

    This avoids uncontrolled neural-network extrapolation while retaining the
    user's requested values for BTE rules, warnings, and result context.
    """

    requested = {
        "torque_nm": float(torque_nm),
        "speed_rpm": float(speed_rpm),
        "compression_ratio": float(compression_ratio),
    }
    model_values: dict[str, float] = {}
    bounded: dict[str, bool] = {}
    for feature in ENGINE_FEATURES:
        limits = bundle["model_ranges"][feature]
        model_values[feature] = float(
            np.clip(requested[feature], float(limits["min"]), float(limits["max"]))
        )
        bounded[feature] = bool(
            not np.isclose(model_values[feature], requested[feature], atol=1e-12)
        )
    return model_values, bounded


def support_distances(bundle: dict[str, Any], compositions: np.ndarray) -> np.ndarray:
    """Return nearest standardized composition distance for one or many blends."""

    values = np.asarray(compositions, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    mean = np.asarray(bundle["support_mean"], dtype=float)
    scale = np.asarray(bundle["support_scale"], dtype=float)
    support_z = (np.asarray(bundle["support_compositions"], dtype=float) - mean) / scale
    values_z = (values - mean) / scale
    output = np.empty(len(values_z), dtype=float)
    for start in range(0, len(values_z), 500):
        block = values_z[start : start + 500]
        distances = np.sqrt(
            ((block[:, None, :] - support_z[None, :, :]) ** 2).sum(axis=2)
        )
        output[start : start + len(block)] = distances.min(axis=1)
    return output


def predict_performance_frame(
    bundle: dict[str, Any],
    candidates: pd.DataFrame,
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
) -> pd.DataFrame:
    """Canonical inference used by Prediction, Optimization, and Simulation."""

    evaluated = _validate_prediction_frame(
        candidates,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )
    model_operating_point, operating_bounds = bounded_operating_point(
        bundle,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )
    x = evaluated.copy()
    for feature, value in model_operating_point.items():
        x[feature] = value
    x = x[MODEL_FEATURES]

    raw = {
        target: np.asarray(model.predict(x), dtype=float).reshape(-1)
        for target, model in bundle["performance_models"].items()
    }
    calculation = bundle["bte_calculation"]
    bte_signal = raw["bte_pct"]
    bte_limits = bundle.get("performance_ranges", {}).get("bte_pct")
    if bte_limits is not None:
        bte_signal = np.clip(
            bte_signal,
            float(bte_limits["min"]),
            float(bte_limits["max"]),
        )
    bte_traces = [
        calculate_bte(
            raw_bte_pct=raw_value,
            compression_ratio=float(compression_ratio),
            blend=row,
            calculation=calculation,
        )
        for raw_value, (_, row) in zip(bte_signal, evaluated.iterrows())
    ]
    evaluated["bte_raw_pct"] = raw["bte_pct"]
    evaluated["bte_base_pct"] = [trace["base_bte_pct"] for trace in bte_traces]
    evaluated["bte_before_output_bounds_pct"] = [
        trace["bte_before_output_bounds_pct"] for trace in bte_traces
    ]
    evaluated["bte_calculated_pct"] = [trace["final_bte_pct"] for trace in bte_traces]
    evaluated["bte_rule_min_pct"] = [trace["range_min_pct"] for trace in bte_traces]
    evaluated["bte_rule_max_pct"] = [trace["range_max_pct"] for trace in bte_traces]
    evaluated["bte_rule_id"] = [trace["rule_id"] for trace in bte_traces]
    evaluated["bte_rule_label"] = [trace["rule_label"] for trace in bte_traces]
    evaluated["bte_normalized_score"] = [trace["normalized_score"] for trace in bte_traces]
    evaluated["bte_calculation_applied"] = [trace["applied"] for trace in bte_traces]
    evaluated["bte_source_was_bounded"] = [
        trace["source_extrapolation_was_bounded"] for trace in bte_traces
    ]
    evaluated["n_methylaniline_cap_active"] = [
        trace["n_methylaniline_cap_active"] for trace in bte_traces
    ]
    evaluated["n_methylaniline_cap_applied"] = [
        trace["n_methylaniline_cap_applied"] for trace in bte_traces
    ]
    evaluated["bte_pct"] = evaluated["bte_calculated_pct"]
    for target in ["bsfc_g_kwh", "co_vol_pct", "hc_ppm"]:
        limits = bundle.get("performance_ranges", {}).get(target)
        if limits is None:
            bounded_values = np.maximum(raw[target], 0.0)
        else:
            bounded_values = np.clip(
                raw[target],
                float(limits["min"]),
                float(limits["max"]),
            )
        evaluated[target] = bounded_values
        evaluated[f"{target}_output_bound_applied"] = ~np.isclose(
            bounded_values,
            raw[target],
            atol=1e-12,
        )

    for feature, value in model_operating_point.items():
        evaluated[f"model_{feature}"] = value
        evaluated[f"{feature}_input_bound_applied"] = operating_bounds[feature]
    evaluated["operating_point_bounded"] = any(operating_bounds.values())

    property_values = bundle["property_model"].predict(evaluated[COMPOSITION_FEATURES])
    for column_index, name in enumerate(PROPERTY_TARGETS):
        limits = bundle["property_ranges"][name]
        evaluated[name] = np.clip(
            property_values[:, column_index], limits["min"], limits["max"]
        )
    evaluated["support_distance"] = support_distances(
        bundle, evaluated[COMPOSITION_FEATURES].to_numpy(dtype=float)
    )
    evaluated["within_validated_support"] = (
        evaluated["support_distance"] <= float(bundle["support_distance_threshold"])
    )
    evaluated["_raw_predictions"] = [
        {target: float(values[index]) for target, values in raw.items()}
        for index in range(len(evaluated))
    ]
    evaluated["_bte_trace"] = bte_traces
    return evaluated


def estimate_properties(bundle: dict[str, Any], blend: dict[str, float]) -> dict[str, float]:
    composition = pd.DataFrame(
        [[float(blend[name]) for name in COMPOSITION_FEATURES]],
        columns=COMPOSITION_FEATURES,
    )
    prediction = bundle["property_model"].predict(composition)[0]
    result: dict[str, float] = {}
    for name, value in zip(PROPERTY_TARGETS, prediction):
        limits = bundle["property_ranges"][name]
        result[name] = float(np.clip(value, limits["min"], limits["max"]))
    return result


def support_distance(bundle: dict[str, Any], blend: dict[str, float]) -> float:
    composition = np.array([float(blend[name]) for name in COMPOSITION_FEATURES])
    return float(support_distances(bundle, composition)[0])


def predict_raw_performance(
    bundle: dict[str, Any],
    blend: dict[str, float],
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
) -> dict[str, Any]:
    """Return direct dataset-trained network outputs without post-processing.

    This deliberately bypasses operating-point bounding, the BTE engineering
    conversion, the n-Methylaniline ceiling, and all target-output bounds. Fuel
    percentages must still describe one valid 100% blend so the input retains
    the same meaning as the training data.
    """

    validate_blend(blend)
    x = _input_frame(
        blend,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )
    raw_predictions = {
        target: float(np.asarray(model.predict(x), dtype=float).reshape(-1)[0])
        for target, model in bundle["performance_models"].items()
    }
    outside_training_ranges: list[str] = []
    for feature in MODEL_FEATURES:
        limits = bundle["model_ranges"][feature]
        value = float(x.iloc[0][feature])
        if value < float(limits["min"]) or value > float(limits["max"]):
            outside_training_ranges.append(feature)

    distance = support_distance(bundle, blend)
    return {
        "inputs": x.iloc[0].to_dict(),
        "predictions": raw_predictions,
        "outside_training_ranges": outside_training_ranges,
        "support_distance": distance,
        "support_threshold": float(bundle["support_distance_threshold"]),
        "within_blend_support": distance
        <= float(bundle["support_distance_threshold"]),
        "post_processing_applied": False,
    }


def _interval(value: float, error: float, lower: float = 0.0, upper: float | None = None) -> tuple[float, float]:
    low = max(lower, value - error)
    high = value + error if upper is None else min(upper, value + error)
    return float(low), float(max(low, high))


def predict_performance(
    bundle: dict[str, Any],
    blend: dict[str, float],
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
) -> dict[str, Any]:
    validate_blend(blend)
    x = _input_frame(blend, torque_nm, speed_rpm, compression_ratio)
    canonical = predict_performance_frame(
        bundle=bundle,
        candidates=pd.DataFrame([blend], columns=COMPOSITION_FEATURES),
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    ).iloc[0]
    raw = dict(canonical["_raw_predictions"])
    calculation = bundle["bte_calculation"]
    bte_trace = dict(canonical["_bte_trace"])
    final = {
        target: float(canonical[target])
        for target in ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]
    }

    intervals: dict[str, tuple[float, float]] = {}
    for target, value in final.items():
        error = float(bundle["error_p90"][target])
        if target == "bte_pct" and bte_trace["applied"]:
            scale = (bte_trace["range_max_pct"] - bte_trace["range_min_pct"]) / (
                bte_trace["source_max_pct"] - bte_trace["source_min_pct"]
            )
            intervals[target] = _interval(
                value,
                error * scale,
                bte_trace["range_min_pct"],
                bte_trace["range_max_pct"],
            )
        elif target == "bte_pct" and bte_trace["n_methylaniline_cap_active"]:
            intervals[target] = _interval(
                value,
                error,
                0.0,
                float(bte_trace["n_methylaniline_cap_pct"]),
            )
        else:
            limits = bundle.get("performance_ranges", {}).get(target, {})
            intervals[target] = _interval(
                value,
                error,
                float(limits.get("min", 0.0)),
                float(limits["max"]) if "max" in limits else None,
            )
    raw_bte_interval = _interval(
        raw["bte_pct"],
        float(bundle["error_p90"]["bte_pct"]),
        0.0,
    )

    properties = {
        name: float(canonical[name])
        for name in PROPERTY_TARGETS
    }
    distance = float(canonical["support_distance"])
    threshold = float(bundle["support_distance_threshold"])
    warnings: list[str] = []
    model_operating_inputs = {
        feature: float(canonical[f"model_{feature}"])
        for feature in ENGINE_FEATURES
    }
    operating_point_bounded = bool(canonical["operating_point_bounded"])
    for feature, value in {
        "torque_nm": torque_nm,
        "speed_rpm": speed_rpm,
        "compression_ratio": compression_ratio,
    }.items():
        limits = bundle["model_ranges"][feature]
        if float(value) < limits["min"] or float(value) > limits["max"]:
            warnings.append(
                f"{feature.replace('_', ' ').title()} is outside the training range "
                f"({limits['min']:.2f}-{limits['max']:.2f}). Neural inference was bounded "
                f"to {model_operating_inputs[feature]:.2f} instead of extrapolating."
            )
    known_cr = np.asarray(bundle.get("known_compression_ratios", []), dtype=float)
    if len(known_cr) and np.min(np.abs(known_cr - float(compression_ratio))) > 0.05:
        nearest_cr = float(known_cr[np.argmin(np.abs(known_cr - float(compression_ratio)))])
        warnings.append(
            f"CR {compression_ratio:.2f} is not directly represented in training; the nearest tested CR is {nearest_cr:.2f}."
        )
    known_speeds = np.asarray(bundle.get("known_speeds_rpm", []), dtype=float)
    if len(known_speeds) and np.min(np.abs(known_speeds - float(speed_rpm))) > 1.0:
        nearest_speed = float(known_speeds[np.argmin(np.abs(known_speeds - float(speed_rpm)))])
        warnings.append(
            f"{speed_rpm:.0f} RPM is between tested speed levels; the nearest measured level is {nearest_speed:.0f} RPM."
        )
    if distance > threshold:
        warnings.append(
            "This composition is farther from the experimental blend support than the validated threshold; "
            "treat the estimate as exploratory."
        )
    if bte_trace["source_extrapolation_was_bounded"]:
        warnings.append(
            "The internal DNN BTE signal is outside the cleaned high-CR source envelope at this CR. "
            "Its normalized score was bounded to 0-1 before applying the final blend range."
        )
    if bte_trace["n_methylaniline_cap_applied"]:
        warnings.append(
            f"The global n-Methylaniline safety ceiling was applied. Reported BTE is limited to "
            f"{bte_trace['n_methylaniline_cap_pct']:.0f}%."
        )
    bounded_targets = [
        target
        for target in ["bsfc_g_kwh", "co_vol_pct", "hc_ppm"]
        if bool(canonical[f"{target}_output_bound_applied"])
    ]
    if bounded_targets:
        warnings.append(
            "One or more internal neural outputs were outside the measured target envelope and "
            "were bounded before reporting: " + ", ".join(bounded_targets) + "."
        )

    physics_bte = 360000.0 / max(final["bsfc_g_kwh"] * properties["cv_mj_kg"], 1e-12)
    if abs(physics_bte - raw["bte_pct"]) > 5.0:
        warnings.append(
            "The dataset-trained BTE signal and BSFC differ by more than 5 percentage points from the "
            "simple heat-balance estimate. Confirm with an engine test before operational use."
        )

    calculation_details = {
        **bte_trace,
        "normalization_formula": calculation["normalization_formula"],
        "output_formula": calculation["output_formula"],
    }

    return {
        "inputs": x.iloc[0].to_dict(),
        "model_operating_inputs": model_operating_inputs,
        "operating_point_bounded": operating_point_bounded,
        "raw_predictions": raw,
        "predictions": final,
        "prediction_intervals_approx_90pct": intervals,
        "raw_bte_interval_approx_90pct": raw_bte_interval,
        "estimated_fuel_properties": properties,
        "bte_calculation_applied": bool(bte_trace["applied"]),
        "n_methylaniline_cap_active": bool(
            bte_trace["n_methylaniline_cap_active"]
        ),
        "n_methylaniline_cap_applied": bool(
            bte_trace["n_methylaniline_cap_applied"]
        ),
        "bte_calculation": calculation_details,
        "support_distance": distance,
        "support_threshold": threshold,
        "within_blend_support": distance <= threshold,
        "simple_heat_balance_bte_pct": float(physics_bte),
        "warnings": warnings,
    }
