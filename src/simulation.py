from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import COMPOSITION_FEATURES
from .optimization import (
    ABSOLUTE_MIN_GASOLINE_PCT,
    ADDITIVE_LABELS,
    evaluate_candidates,
    generate_single_additive_candidates,
)
from .prediction import validate_blend


def simulate_fixed_blend(
    bundle: dict[str, Any],
    selected_additive: str | None,
    gasoline_pct: float,
    ethanol_pct: float,
    additive_pct: float,
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
    min_gasoline_pct: float = ABSOLUTE_MIN_GASOLINE_PCT,
) -> pd.DataFrame:
    """Evaluate one fully user-defined gasoline/ethanol/additive blend."""

    if selected_additive is not None and selected_additive not in ADDITIVE_LABELS:
        raise ValueError("The selected additive is not supported.")
    values = np.asarray([gasoline_pct, ethanol_pct, additive_pct], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Fixed blend values must be finite numbers.")
    if selected_additive is None and abs(float(additive_pct)) > 1e-12:
        raise ValueError("Additive must be 0% when no additive is selected.")

    blend = {feature: 0.0 for feature in COMPOSITION_FEATURES}
    blend["gasoline_pct"] = float(gasoline_pct)
    blend["ethanol_pct"] = float(ethanol_pct)
    if selected_additive is not None:
        blend[selected_additive] = float(additive_pct)
    validate_blend(blend)
    if float(gasoline_pct) < float(min_gasoline_pct) - 1e-9:
        raise ValueError(
            f"Gasoline must be at least {float(min_gasoline_pct):.0f}% for simulation."
        )
    if float(gasoline_pct) <= float(ethanol_pct) + 1e-9:
        raise ValueError("Gasoline must be greater than ethanol.")
    if float(gasoline_pct) <= float(additive_pct) + 1e-9:
        raise ValueError("Gasoline must be greater than the selected additive.")

    return evaluate_candidates(
        bundle=bundle,
        candidates=pd.DataFrame([blend], columns=COMPOSITION_FEATURES),
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )


def simulate_blend_grid(
    bundle: dict[str, Any],
    selected_additive: str | None,
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
    min_ethanol_pct: float = 0.0,
    max_ethanol_pct: float = 20.0,
    min_additive_pct: float = 0.0,
    max_additive_pct: float = 5.0,
    min_gasoline_pct: float = ABSOLUTE_MIN_GASOLINE_PCT,
    grid_step_pct: float = 1.0,
    fixed_ethanol_pct: float | None = None,
    fixed_gasoline_pct: float | None = None,
) -> pd.DataFrame:
    """Simulate the canonical deep-network predictor across a valid blend grid."""

    candidates = generate_single_additive_candidates(
        selected_additive=selected_additive,
        min_gasoline_pct=min_gasoline_pct,
        min_ethanol_pct=min_ethanol_pct,
        max_ethanol_pct=max_ethanol_pct,
        min_additive_pct=min_additive_pct,
        max_additive_pct=max_additive_pct,
        grid_step_pct=grid_step_pct,
        fixed_ethanol_pct=fixed_ethanol_pct,
        fixed_gasoline_pct=fixed_gasoline_pct,
    )
    results = evaluate_candidates(
        bundle=bundle,
        candidates=candidates,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )
    results["within_validated_support"] = (
        results["support_distance"] <= float(bundle["support_distance_threshold"])
    )
    if selected_additive is not None and (
        fixed_ethanol_pct is not None or fixed_gasoline_pct is not None
    ):
        sort_columns = [selected_additive]
    else:
        sort_columns = ["ethanol_pct"]
        if selected_additive is not None:
            sort_columns.append(selected_additive)
    return results.sort_values(sort_columns).reset_index(drop=True)
