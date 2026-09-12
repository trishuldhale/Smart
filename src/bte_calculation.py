from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import (
    BTE_BASE_RANGE_MAX,
    BTE_BASE_RANGE_MIN,
    BTE_CALCULATION_CR_THRESHOLD,
    COMPOSITION_FEATURES,
)


N_METHYLANILINE_BTE_MAX_PCT = 37.0


# These are user-supplied engineering ranges, not values learned by the deep
# network. The E5 interpretation resolves the invalid "95% gasoline + 27%
# ethanol" statement into the consistent E0/E5/E10/E15/E20 sequence.
ETHANOL_ONLY_RULES = (
    {"rule_id": "E0", "gasoline_pct": 100.0, "ethanol_pct": 0.0, "minimum_pct": 25.0, "maximum_pct": 30.0},
    {"rule_id": "E5", "gasoline_pct": 95.0, "ethanol_pct": 5.0, "minimum_pct": 27.0, "maximum_pct": 32.0},
    {"rule_id": "E10", "gasoline_pct": 90.0, "ethanol_pct": 10.0, "minimum_pct": 31.0, "maximum_pct": 35.0},
    {"rule_id": "E15", "gasoline_pct": 85.0, "ethanol_pct": 15.0, "minimum_pct": 26.0, "maximum_pct": 31.0},
    {"rule_id": "E20", "gasoline_pct": 80.0, "ethanol_pct": 20.0, "minimum_pct": 23.0, "maximum_pct": 30.0},
)

ADDITIVE_RULES = (
    {"rule_id": "E10_P5", "gasoline_pct": 85.0, "ethanol_pct": 10.0, "additive": "pentanol_pct", "additive_pct": 5.0, "minimum_pct": 32.0, "maximum_pct": 37.0},
    {"rule_id": "E10_Pr5", "gasoline_pct": 85.0, "ethanol_pct": 10.0, "additive": "propanol_pct", "additive_pct": 5.0, "minimum_pct": 31.0, "maximum_pct": 35.0},
    {"rule_id": "E10_B5", "gasoline_pct": 85.0, "ethanol_pct": 10.0, "additive": "butanol_pct", "additive_pct": 5.0, "minimum_pct": 31.0, "maximum_pct": 35.0},
    {"rule_id": "E10_NMA5", "gasoline_pct": 85.0, "ethanol_pct": 10.0, "additive": "n_methylaniline_pct", "additive_pct": 5.0, "minimum_pct": 31.0, "maximum_pct": 37.0},
)


def build_bte_calculation_metadata(data: pd.DataFrame) -> dict[str, Any]:
    """Build auditable CR-specific source envelopes from cleaned measurements."""

    high_cr = data[data["compression_ratio"] >= BTE_CALCULATION_CR_THRESHOLD]
    if high_cr.empty:
        raise ValueError(
            f"No cleaned BTE records were found at CR >= {BTE_CALCULATION_CR_THRESHOLD:.1f}."
        )
    envelopes = []
    for compression_ratio, group in high_cr.groupby("compression_ratio", sort=True):
        envelopes.append(
            {
                "compression_ratio": float(compression_ratio),
                "source_min_pct": float(group["bte_pct"].min()),
                "source_max_pct": float(group["bte_pct"].max()),
                "source_rows": int(len(group)),
            }
        )
    return {
        "compression_ratio_threshold": BTE_CALCULATION_CR_THRESHOLD,
        "base_range_min_pct": BTE_BASE_RANGE_MIN,
        "base_range_max_pct": BTE_BASE_RANGE_MAX,
        "source_envelopes_by_cr": envelopes,
        "ethanol_only_rules": [dict(rule) for rule in ETHANOL_ONLY_RULES],
        "additive_rules": [dict(rule) for rule in ADDITIVE_RULES],
        "normalization_formula": "q = clip((raw_bte - source_min_at_cr) / (source_max_at_cr - source_min_at_cr), 0, 1)",
        "output_formula": "final_bte = blend_range_min + q * (blend_range_max - blend_range_min)",
        "e5_assumption": "95% gasoline + 5% ethanol; 27% in the prompt was interpreted as the lower BTE bound because 95% + 27% is invalid.",
    }


def source_envelope_at_cr(calculation: Mapping[str, Any], compression_ratio: float) -> tuple[float, float]:
    """Interpolate the observed BTE envelope across measured high-CR levels."""

    envelopes = calculation["source_envelopes_by_cr"]
    cr_points = np.asarray([row["compression_ratio"] for row in envelopes], dtype=float)
    minima = np.asarray([row["source_min_pct"] for row in envelopes], dtype=float)
    maxima = np.asarray([row["source_max_pct"] for row in envelopes], dtype=float)
    cr = float(compression_ratio)
    return float(np.interp(cr, cr_points, minima)), float(np.interp(cr, cr_points, maxima))


def _blend_range(blend: Mapping[str, float], tolerance: float = 0.15) -> tuple[float, float, str, str]:
    gasoline = float(blend["gasoline_pct"])
    ethanol = float(blend["ethanol_pct"])
    additives = {
        name: float(blend[name])
        for name in [
            "pentanol_pct",
            "propanol_pct",
            "butanol_pct",
            "n_methylaniline_pct",
        ]
    }
    additive_total = sum(additives.values())

    # Ethanol-only blends use piecewise-linear interpolation between the five
    # explicit E0/E5/E10/E15/E20 anchor ranges.
    if additive_total <= tolerance and abs(gasoline + ethanol - 100.0) <= tolerance and 0.0 <= ethanol <= 20.0:
        ethanol_points = np.asarray([rule["ethanol_pct"] for rule in ETHANOL_ONLY_RULES])
        lower_points = np.asarray([rule["minimum_pct"] for rule in ETHANOL_ONLY_RULES])
        upper_points = np.asarray([rule["maximum_pct"] for rule in ETHANOL_ONLY_RULES])
        lower = float(np.interp(ethanol, ethanol_points, lower_points))
        upper = float(np.interp(ethanol, ethanol_points, upper_points))
        exact = next(
            (rule for rule in ETHANOL_ONLY_RULES if abs(ethanol - rule["ethanol_pct"]) <= tolerance),
            None,
        )
        if exact is not None:
            return lower, upper, str(exact["rule_id"]), f"{exact['rule_id']} user range"
        return lower, upper, "ethanol_interpolated", "Interpolated ethanol-only user range"

    # For E10 plus up to 5% additive, the range transitions from E10. Pentanol
    # supplies the requested improvement. n-Methylaniline now has the requested
    # 37% upper limit at 5%; propanol and butanol retain the E10 31-35% range.
    # If multiple additives are entered outside the constrained optimizer, the
    # largest single improvement is used rather than inventing synergy.
    if (
        abs(ethanol - 10.0) <= tolerance
        and tolerance < additive_total <= 5.0 + tolerance
        and abs(gasoline + ethanol + additive_total - 100.0) <= tolerance
    ):
        pentanol_fraction = float(np.clip(additives["pentanol_pct"] / 5.0, 0.0, 1.0))
        nma_fraction = float(
            np.clip(additives["n_methylaniline_pct"] / 5.0, 0.0, 1.0)
        )
        lower = 31.0 + pentanol_fraction
        upper = 35.0 + 2.0 * max(pentanol_fraction, nma_fraction)
        active = [name for name, value in additives.items() if value > tolerance]
        exact = next(
            (
                rule
                for rule in ADDITIVE_RULES
                if len(active) == 1
                and active[0] == rule["additive"]
                and abs(additives[active[0]] - rule["additive_pct"]) <= tolerance
            ),
            None,
        )
        if exact is not None:
            return lower, upper, str(exact["rule_id"]), f"{exact['rule_id']} user range"
        return lower, upper, "e10_additive_calculated", "Calculated E10 + additive range"

    return BTE_BASE_RANGE_MIN, BTE_BASE_RANGE_MAX, "base_26_36", "Base high-CR range"


def calculate_bte(
    raw_bte_pct: float,
    compression_ratio: float,
    blend: Mapping[str, float],
    calculation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return raw, normalized, and rule-guided BTE with full calculation trace."""

    raw = float(raw_bte_pct)
    threshold = float(calculation["compression_ratio_threshold"])
    if float(compression_ratio) < threshold:
        trace = {
            "applied": False,
            "raw_bte_pct": raw,
            "normalized_score": None,
            "base_bte_pct": raw,
            "final_bte_pct": raw,
            "range_min_pct": None,
            "range_max_pct": None,
            "rule_id": "dnn_signal_below_cr_threshold",
            "rule_label": "Deep-network BTE below CR threshold",
            "source_min_pct": None,
            "source_max_pct": None,
            "source_extrapolation_was_bounded": False,
        }
        return _apply_output_bounds(trace, blend)

    source_min, source_max = source_envelope_at_cr(calculation, compression_ratio)
    if source_max <= source_min:
        raise ValueError("The BTE source envelope is invalid for this compression ratio.")
    unbounded_score = (raw - source_min) / (source_max - source_min)
    score = float(np.clip(unbounded_score, 0.0, 1.0))
    base_bte = float(
        calculation["base_range_min_pct"]
        + score * (calculation["base_range_max_pct"] - calculation["base_range_min_pct"])
    )
    range_min, range_max, rule_id, rule_label = _blend_range(blend)
    final_bte = float(range_min + score * (range_max - range_min))
    trace = {
        "applied": True,
        "raw_bte_pct": raw,
        "normalized_score": score,
        "base_bte_pct": base_bte,
        "final_bte_pct": final_bte,
        "range_min_pct": float(range_min),
        "range_max_pct": float(range_max),
        "rule_id": rule_id,
        "rule_label": rule_label,
        "source_min_pct": source_min,
        "source_max_pct": source_max,
        "source_extrapolation_was_bounded": bool(not np.isclose(score, unbounded_score)),
    }
    return _apply_output_bounds(trace, blend)


def _apply_output_bounds(
    trace: dict[str, Any],
    blend: Mapping[str, float],
) -> dict[str, Any]:
    """Apply final physical bounds and the global n-Methylaniline ceiling.

    The 37% ceiling is intentionally independent of compression ratio and of
    which blend-range rule was selected. This makes the same safety rule apply
    to Prediction, Optimization, and Simulation, including CR < 9.5.
    """

    result = dict(trace)
    before_bounds = float(result["final_bte_pct"])
    bounded = max(0.0, before_bounds)
    nma_active = float(blend.get("n_methylaniline_pct", 0.0)) > 1e-12
    if nma_active:
        bounded = min(bounded, N_METHYLANILINE_BTE_MAX_PCT)
    result["bte_before_output_bounds_pct"] = before_bounds
    result["final_bte_pct"] = float(bounded)
    result["physical_lower_bound_applied"] = bool(before_bounds < 0.0)
    result["n_methylaniline_cap_active"] = bool(nma_active)
    result["n_methylaniline_cap_pct"] = (
        N_METHYLANILINE_BTE_MAX_PCT if nma_active else None
    )
    result["n_methylaniline_cap_applied"] = bool(
        nma_active and before_bounds > N_METHYLANILINE_BTE_MAX_PCT
    )
    if result["n_methylaniline_cap_applied"]:
        result["rule_id_before_output_cap"] = result["rule_id"]
        result["rule_label_before_output_cap"] = result["rule_label"]
        result["rule_id"] = "nma_global_cap_37"
        result["rule_label"] = "Global n-Methylaniline 37% BTE ceiling"
    return result


def rule_anchor_compositions() -> pd.DataFrame:
    """Return exact user-rule blends so the optimizer always evaluates them."""

    rows: list[dict[str, float]] = []
    for rule in ETHANOL_ONLY_RULES:
        row = {name: 0.0 for name in COMPOSITION_FEATURES}
        row["gasoline_pct"] = float(rule["gasoline_pct"])
        row["ethanol_pct"] = float(rule["ethanol_pct"])
        rows.append(row)
    for rule in ADDITIVE_RULES:
        row = {name: 0.0 for name in COMPOSITION_FEATURES}
        row["gasoline_pct"] = float(rule["gasoline_pct"])
        row["ethanol_pct"] = float(rule["ethanol_pct"])
        row[str(rule["additive"])] = float(rule["additive_pct"])
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPOSITION_FEATURES)
