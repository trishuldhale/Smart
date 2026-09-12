from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .bte_calculation import rule_anchor_compositions
from .config import COMPOSITION_FEATURES, PERFORMANCE_TARGETS, RANDOM_STATE
from .prediction import predict_performance_frame


DEFAULT_COMPONENT_LIMITS = {
    "ethanol_pct": 20.0,
    "pentanol_pct": 20.0,
    "propanol_pct": 20.0,
    "butanol_pct": 20.0,
    "n_methylaniline_pct": 0.0,
}

ADDITIVE_LABELS = {
    "pentanol_pct": "Pentanol",
    "propanol_pct": "Propanol",
    "butanol_pct": "Butanol",
    "n_methylaniline_pct": "n-Methylaniline",
}

ABSOLUTE_MIN_GASOLINE_PCT = 75.0
MAX_SUPPORT_PENALTY = 0.03


def generate_candidates(
    bundle: dict[str, Any],
    n_candidates: int = 2500,
    min_gasoline_pct: float = ABSOLUTE_MIN_GASOLINE_PCT,
    component_limits: dict[str, float] | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create feasible blends from empirical mixtures plus explicit rule anchors."""

    if not np.isfinite(min_gasoline_pct) or not (
        ABSOLUTE_MIN_GASOLINE_PCT <= float(min_gasoline_pct) <= 100.0
    ):
        raise ValueError(
            f"Minimum gasoline must be between {ABSOLUTE_MIN_GASOLINE_PCT:.0f} and 100%."
        )
    limits = {**DEFAULT_COMPONENT_LIMITS, **(component_limits or {})}
    support = np.asarray(bundle["support_compositions"], dtype=float)
    rng = np.random.default_rng(random_state)
    rule_anchors = rule_anchor_compositions().to_numpy(dtype=float)
    generated = [support, rule_anchors]
    remaining = max(0, n_candidates - len(support) - len(rule_anchors))
    if remaining:
        indices = rng.integers(0, len(support), size=(remaining, 3))
        weights = rng.dirichlet([1.2, 1.2, 1.2], size=remaining)
        mixed = (support[indices] * weights[:, :, None]).sum(axis=1)
        generated.append(mixed)

    values = np.vstack(generated)
    values = values / values.sum(axis=1, keepdims=True) * 100.0
    frame = pd.DataFrame(values, columns=COMPOSITION_FEATURES)
    mask = frame["gasoline_pct"] >= float(min_gasoline_pct)
    mask &= frame["gasoline_pct"] > frame["ethanol_pct"]
    for additive in ADDITIVE_LABELS:
        mask &= frame["gasoline_pct"] > frame[additive]
    for component, maximum in limits.items():
        mask &= frame[component] <= float(maximum) + 1e-9
    frame = frame.loc[mask].round(4).drop_duplicates().reset_index(drop=True)
    frame["gasoline_pct"] += 100.0 - frame[COMPOSITION_FEATURES].sum(axis=1)
    return frame


def _inclusive_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    values = np.arange(float(minimum), float(maximum) + float(step) * 0.5, float(step))
    values = values[values <= float(maximum) + 1e-9]
    if len(values) == 0 or not np.isclose(values[-1], float(maximum)):
        values = np.append(values, float(maximum))
    return np.unique(np.round(values, 6))


def generate_single_additive_candidates(
    selected_additive: str | None,
    min_gasoline_pct: float = ABSOLUTE_MIN_GASOLINE_PCT,
    min_ethanol_pct: float = 0.0,
    max_ethanol_pct: float = 20.0,
    min_additive_pct: float | None = None,
    max_additive_pct: float = 5.0,
    grid_step_pct: float = 0.5,
    fixed_ethanol_pct: float | None = None,
    fixed_gasoline_pct: float | None = None,
) -> pd.DataFrame:
    """Generate valid gasoline/ethanol/single-additive blends.

    At most one base component may be fixed. With fixed ethanol, only the
    additive is swept and gasoline is the balance. With fixed gasoline, only
    the additive is swept and ethanol is the balance. If neither is fixed,
    ethanol and the selected additive are both swept.
    """

    if selected_additive is not None and selected_additive not in ADDITIVE_LABELS:
        raise ValueError("The selected additive is not supported.")
    numeric_values = [
        min_gasoline_pct,
        min_ethanol_pct,
        max_ethanol_pct,
        max_additive_pct,
        grid_step_pct,
    ]
    numeric_values.extend(
        value
        for value in [fixed_ethanol_pct, fixed_gasoline_pct]
        if value is not None
    )
    if not np.isfinite(numeric_values).all():
        raise ValueError("Blend-search limits must be finite numbers.")
    if fixed_ethanol_pct is not None and fixed_gasoline_pct is not None:
        raise ValueError("Fix either gasoline or ethanol, not both.")
    if not ABSOLUTE_MIN_GASOLINE_PCT <= float(min_gasoline_pct) <= 100.0:
        raise ValueError(
            f"Minimum gasoline must be between {ABSOLUTE_MIN_GASOLINE_PCT:.0f} and 100%."
        )
    if not 0.0 <= float(min_ethanol_pct) <= float(max_ethanol_pct) <= 100.0:
        raise ValueError("The ethanol range must be ordered and remain within 0-100%.")
    if not 0.0 <= float(max_additive_pct) <= 100.0:
        raise ValueError("Maximum additive must be between 0 and 100%.")
    if float(grid_step_pct) <= 0.0:
        raise ValueError("Blend-grid resolution must be greater than zero.")
    if fixed_ethanol_pct is not None and not 0.0 <= float(fixed_ethanol_pct) <= 100.0:
        raise ValueError("Fixed ethanol must be between 0 and 100%.")
    if fixed_gasoline_pct is not None:
        if not ABSOLUTE_MIN_GASOLINE_PCT <= float(fixed_gasoline_pct) <= 100.0:
            raise ValueError(
                f"Fixed gasoline must be between {ABSOLUTE_MIN_GASOLINE_PCT:.0f} and 100%."
            )
        if float(fixed_gasoline_pct) < float(min_gasoline_pct) - 1e-9:
            raise ValueError("Fixed gasoline cannot be below the selected minimum gasoline.")

    if selected_additive is None:
        additive_values = np.asarray([0.0])
    else:
        if float(max_additive_pct) <= 0.0:
            raise ValueError("Maximum additive must be greater than zero for the selected additive.")
        additive_minimum = (
            min(float(grid_step_pct), float(max_additive_pct))
            if min_additive_pct is None
            else float(min_additive_pct)
        )
        if not np.isfinite(additive_minimum):
            raise ValueError("Minimum additive must be a finite number.")
        if not 0.0 <= additive_minimum <= float(max_additive_pct):
            raise ValueError("The additive range must be ordered and remain within 0-100%.")
        additive_values = _inclusive_grid(
            additive_minimum,
            float(max_additive_pct),
            float(grid_step_pct),
        )

    if fixed_ethanol_pct is not None:
        ethanol_values = np.asarray([float(fixed_ethanol_pct)])
    elif fixed_gasoline_pct is None:
        ethanol_values = _inclusive_grid(
            float(min_ethanol_pct),
            float(max_ethanol_pct),
            float(grid_step_pct),
        )
    else:
        # Ethanol is calculated as the balance for each additive value.
        ethanol_values = np.asarray([0.0])

    rows: list[dict[str, float]] = []
    for ethanol in ethanol_values:
        for additive in additive_values:
            if fixed_gasoline_pct is not None:
                gasoline = float(fixed_gasoline_pct)
                candidate_ethanol = 100.0 - gasoline - float(additive)
            else:
                candidate_ethanol = float(ethanol)
                gasoline = 100.0 - candidate_ethanol - float(additive)

            if (
                gasoline < float(min_gasoline_pct) - 1e-9
                or gasoline < ABSOLUTE_MIN_GASOLINE_PCT - 1e-9
                or candidate_ethanol < -1e-9
                or gasoline <= candidate_ethanol + 1e-9
                or gasoline <= float(additive) + 1e-9
            ):
                continue
            row = {feature: 0.0 for feature in COMPOSITION_FEATURES}
            row["gasoline_pct"] = gasoline
            row["ethanol_pct"] = candidate_ethanol
            if selected_additive is not None:
                row[selected_additive] = float(additive)
            rows.append(row)

    if not rows:
        raise ValueError("No blend can satisfy the selected gasoline, ethanol, and additive limits.")
    frame = pd.DataFrame(rows, columns=COMPOSITION_FEATURES).round(6)
    frame["gasoline_pct"] += 100.0 - frame[COMPOSITION_FEATURES].sum(axis=1)
    return frame.drop_duplicates().reset_index(drop=True)


def evaluate_candidates(
    bundle: dict[str, Any],
    candidates: pd.DataFrame,
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
) -> pd.DataFrame:
    """Apply the one canonical prediction path to optimizer/simulator rows."""

    return predict_performance_frame(
        bundle=bundle,
        candidates=candidates,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )


def _validate_target_values(target_values: dict[str, float]) -> dict[str, float]:
    """Validate the complete exact performance profile entered by the user."""

    supplied = set(target_values)
    expected = set(PERFORMANCE_TARGETS)
    missing = expected - supplied
    unknown = supplied - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError(
            "Exact performance values are incomplete: " + "; ".join(details) + "."
        )

    normalized = {
        target: float(target_values[target]) for target in PERFORMANCE_TARGETS
    }
    values = np.asarray(list(normalized.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Exact performance values must be finite numbers.")
    if (values < 0.0).any():
        raise ValueError("Exact performance values cannot be negative.")
    return normalized


def _score_exact_target_candidates(
    bundle: dict[str, Any],
    supported: pd.DataFrame,
    target_values: dict[str, float],
    support_threshold: float,
) -> pd.DataFrame:
    """Score every supported blend by closeness to four exact entered values.

    Each absolute gap is divided by that output's unseen-blend P90 error so
    BTE, BSFC, CO, and HC can participate equally despite their different
    units. The support penalty remains deliberately small and only breaks
    otherwise similar profile matches in favor of experimental evidence.
    """

    exact = _validate_target_values(target_values)
    scored = supported.copy()
    profile_score = np.zeros(len(scored), dtype=float)
    normalized_gaps: list[np.ndarray] = []
    equal_weight = 1.0 / len(PERFORMANCE_TARGETS)

    for target in PERFORMANCE_TARGETS:
        predictions = scored[target].to_numpy(dtype=float)
        entered = exact[target]
        absolute_gap = np.abs(predictions - entered)
        error_scale = max(float(bundle["error_p90"][target]), 1e-12)
        normalized_gap = absolute_gap / error_scale
        match_utility = 1.0 / (1.0 + normalized_gap)
        contribution = equal_weight * match_utility

        scored[f"{target}_entered_value"] = entered
        scored[f"{target}_absolute_gap"] = absolute_gap
        scored[f"{target}_normalized_gap"] = normalized_gap
        scored[f"{target}_match_utility"] = match_utility
        scored[f"{target}_match_contribution"] = contribution
        scored[f"{target}_error_scale"] = error_scale
        profile_score += contribution
        normalized_gaps.append(normalized_gap)

    gap_matrix = np.column_stack(normalized_gaps)
    support_ratio = np.clip(
        scored["support_distance"].to_numpy(dtype=float) / support_threshold,
        0.0,
        1.0,
    )
    scored["target_matching_active"] = True
    scored["target_match_score_pct"] = 100.0 * profile_score
    scored["mean_normalized_target_gap"] = gap_matrix.mean(axis=1)
    scored["largest_normalized_target_gap"] = gap_matrix.max(axis=1)
    scored["support_confidence"] = 1.0 - support_ratio
    scored["support_penalty"] = MAX_SUPPORT_PENALTY * support_ratio
    scored["weighted_score"] = profile_score - scored["support_penalty"]
    scored["recommendation_score_pct"] = 100.0 * np.clip(
        scored["weighted_score"], 0.0, 1.0
    )
    return scored


def _deterministic_target_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank exact-profile matches reproducibly."""

    return frame.sort_values(
        [
            "weighted_score",
            "mean_normalized_target_gap",
            "largest_normalized_target_gap",
            "support_distance",
            "gasoline_pct",
            "ethanol_pct",
        ],
        ascending=[False, True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _attach_baseline_comparison(
    best: pd.Series,
    scored_supported: pd.DataFrame,
    selected_additive: str | None,
) -> pd.Series:
    """Attach an honest comparison with the best zero-additive alternative."""

    result = best.copy()
    result["baseline_available"] = False
    if selected_additive is None:
        return result
    baseline = scored_supported.loc[
        scored_supported[selected_additive].abs() <= 1e-12
    ]
    if baseline.empty:
        return result
    baseline_best = _deterministic_target_rank(baseline).iloc[0]
    result["baseline_available"] = True
    for feature in COMPOSITION_FEATURES:
        result[f"baseline_{feature}"] = float(baseline_best[feature])
    for target in PERFORMANCE_TARGETS:
        result[f"baseline_{target}"] = float(baseline_best[target])
        result[f"gap_vs_baseline_{target}"] = float(
            result[target] - baseline_best[target]
        )
    result["baseline_target_match_score_pct"] = float(
        baseline_best["target_match_score_pct"]
    )
    result["target_match_score_delta_vs_baseline_pct"] = float(
        result["target_match_score_pct"] - baseline_best["target_match_score_pct"]
    )
    return result


def optimize_blend(
    bundle: dict[str, Any],
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
    n_candidates: int = 2500,
    min_gasoline_pct: float = ABSOLUTE_MIN_GASOLINE_PCT,
    component_limits: dict[str, float] | None = None,
    candidate_frame: pd.DataFrame | None = None,
    selected_additive: str | None = None,
    target_values: dict[str, float] | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Rank supported blends by their complete exact performance-profile match."""

    if target_values is None:
        raise ValueError("Enter exact values for BTE, BSFC, CO, and HC.")
    exact_target_values = _validate_target_values(target_values)
    if not np.isfinite(min_gasoline_pct) or not (
        ABSOLUTE_MIN_GASOLINE_PCT <= float(min_gasoline_pct) <= 100.0
    ):
        raise ValueError(
            f"Minimum gasoline must be between {ABSOLUTE_MIN_GASOLINE_PCT:.0f} and 100%."
        )
    candidates = (
        generate_candidates(
            bundle,
            n_candidates=n_candidates,
            min_gasoline_pct=min_gasoline_pct,
            component_limits=component_limits,
        )
        if candidate_frame is None
        else candidate_frame.copy()
    )
    if all(feature in candidates for feature in COMPOSITION_FEATURES):
        dominant_gasoline = candidates["gasoline_pct"] >= float(min_gasoline_pct)
        dominant_gasoline &= candidates["gasoline_pct"] > candidates["ethanol_pct"]
        for additive in ADDITIVE_LABELS:
            dominant_gasoline &= candidates["gasoline_pct"] > candidates[additive]
        candidates = candidates.loc[dominant_gasoline].reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No candidate blend satisfies the selected component limits.")
    candidates = evaluate_candidates(
        bundle,
        candidates,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )
    threshold = float(bundle["support_distance_threshold"])
    supported = candidates.loc[
        candidates["support_distance"] <= threshold
    ].reset_index(drop=True)
    if supported.empty:
        raise ValueError("Candidate blends were outside the validated experimental support.")

    supported = _score_exact_target_candidates(
        bundle,
        supported,
        exact_target_values,
        support_threshold=threshold,
    )
    ranked = _deterministic_target_rank(supported)
    best = ranked.iloc[0].copy()
    if len(ranked) > 1:
        score_margin = float(
            best["weighted_score"] - ranked.iloc[1]["weighted_score"]
        )
    else:
        score_margin = float("nan")
    best["score_margin_to_runner_up"] = score_margin
    output_bounds_applied = any(
        bool(best.get(f"{target}_output_bound_applied", False))
        for target in ["bsfc_g_kwh", "co_vol_pct", "hc_ppm"]
    )
    best["performance_output_bound_applied"] = output_bounds_applied
    mean_gap = float(best["mean_normalized_target_gap"])
    largest_gap = float(best["largest_normalized_target_gap"])
    if bool(best.get("operating_point_bounded", False)) or output_bounds_applied:
        confidence = "Exploratory"
        confidence_reason = (
            "The requested operating point or an internal output required bounding "
            "to the measured data envelope."
        )
    elif mean_gap <= 0.5 and largest_gap <= 1.0:
        confidence = "High"
        confidence_reason = (
            "The predicted four-output profile is close to every entered value "
            "relative to unseen-blend error."
        )
    elif mean_gap <= 1.0 and largest_gap <= 2.0:
        confidence = "Moderate"
        confidence_reason = (
            "The overall profile is close, although at least one entered value has "
            "a noticeable model-scale gap."
        )
    else:
        confidence = "Low"
        confidence_reason = (
            "No supported blend closely matches the complete entered performance profile."
        )
    best["recommendation_confidence"] = confidence
    best["recommendation_confidence_reason"] = confidence_reason
    best["recommendation_method"] = (
        "Equal four-output exact-profile matching with unseen-blend error scaling "
        "and support penalty"
    )
    best = _attach_baseline_comparison(best, supported, selected_additive)
    return best, ranked, supported


def optimize_single_additive_blend(
    bundle: dict[str, Any],
    selected_additive: str | None,
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
    min_gasoline_pct: float = ABSOLUTE_MIN_GASOLINE_PCT,
    max_ethanol_pct: float = 20.0,
    max_additive_pct: float = 5.0,
    grid_step_pct: float = 0.5,
    fixed_ethanol_pct: float | None = None,
    fixed_gasoline_pct: float | None = None,
    target_values: dict[str, float] | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Recommend a blend containing only gasoline, ethanol, and one chosen additive."""

    candidates = generate_single_additive_candidates(
        selected_additive=selected_additive,
        min_gasoline_pct=min_gasoline_pct,
        min_ethanol_pct=0.0,
        max_ethanol_pct=max_ethanol_pct,
        min_additive_pct=0.0,
        max_additive_pct=max_additive_pct,
        grid_step_pct=grid_step_pct,
        fixed_ethanol_pct=fixed_ethanol_pct,
        fixed_gasoline_pct=fixed_gasoline_pct,
    )
    return optimize_blend(
        bundle=bundle,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
        min_gasoline_pct=min_gasoline_pct,
        candidate_frame=candidates,
        selected_additive=selected_additive,
        target_values=target_values,
    )
