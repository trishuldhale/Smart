from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import (
    ARTIFACT_DIR,
    COMPOSITION_FEATURES,
    FIGURE_DIR,
    PROCESSED_DIR,
    PROPERTY_LABELS,
    REPORT_DIR,
    TARGET_LABELS,
)
from src.optimization import (
    ABSOLUTE_MIN_GASOLINE_PCT,
    ADDITIVE_LABELS,
    optimize_single_additive_blend,
)
from src.prediction import (
    InputValidationError,
    blend_total,
    load_model_bundle,
    predict_performance,
    predict_raw_performance,
)
from src.simulation import simulate_blend_grid, simulate_fixed_blend


st.set_page_config(
    page_title="GEB-AI Engine Decision Support",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

style_path = Path(__file__).resolve().parent / "assets" / "motorsport.css"
st.markdown(f"<style>{style_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

plt.rcParams.update(
    {
        "figure.facecolor": "#11151b",
        "axes.facecolor": "#151a21",
        "axes.edgecolor": "#596371",
        "axes.labelcolor": "#e9edf2",
        "axes.titlecolor": "#ffffff",
        "text.color": "#e9edf2",
        "xtick.color": "#aeb7c2",
        "ytick.color": "#aeb7c2",
        "grid.color": "#4a535f",
        "legend.facecolor": "#151a21",
        "legend.edgecolor": "#3b434e",
    }
)


@st.cache_resource
def load_resources():
    bundle_path = ARTIFACT_DIR / "geb_ai_model_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("Model bundle not found. Run: python train_models.py")
    bundle = load_model_bundle(bundle_path)
    vehicles = pd.read_csv(PROCESSED_DIR / "vehicles.csv")
    holdout = pd.read_csv(REPORT_DIR / "holdout_metrics.csv")
    cv = pd.read_csv(REPORT_DIR / "grouped_cv_metrics.csv")
    importance = pd.read_csv(REPORT_DIR / "feature_importance.csv")
    property_cv = pd.read_csv(REPORT_DIR / "fuel_property_cv_metrics.csv")
    quarantine = pd.read_csv(REPORT_DIR.parent / "data" / "processed" / "quarantine_invalid_rows.csv")
    audit = json.loads((REPORT_DIR / "data_audit.json").read_text(encoding="utf-8"))
    return bundle, vehicles, holdout, cv, importance, property_cv, quarantine, audit


try:
    bundle, vehicles, holdout_metrics, cv_metrics, importance, property_cv_metrics, quarantine, audit = load_resources()
except Exception as exc:
    st.error(f"The application could not load its trained artifacts: {exc}")
    st.info("From the project folder, run `python train_models.py`, then start the app again.")
    st.stop()


st.markdown(
    """
    <section class="race-hero">
      <div class="race-kicker"><span>GEB-AI</span> Mechanical Intelligence Lab</div>
      <h1>Fuel Blend <em>Race Engineering</em></h1>
      <p>Vehicle-aware performance, emissions and blend simulation.</p>
      <div class="race-tags">
        <span class="race-tag">Performance</span>
        <span class="race-tag">Efficiency</span>
        <span class="race-tag">Emissions</span>
        <span class="race-tag">Simulation</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Vehicle & test point")
    company = st.selectbox("Company", sorted(vehicles["company"].dropna().unique()))
    company_vehicles = vehicles[vehicles["company"] == company]
    model_name = st.selectbox("Model", company_vehicles["model"].tolist())
    vehicle = company_vehicles[company_vehicles["model"] == model_name].iloc[0]

    vehicle_id = str(vehicle["vehicle_id"])
    documented_stage = str(vehicle["bs_norm_at_launch"]).upper()
    default_stage_index = 0 if documented_stage.startswith("BS-VI") else 1
    bharat_stage = st.selectbox(
        "Bharat Stage",
        ["BS6", "BS4"],
        index=default_stage_index,
        key=f"bharat_stage_{vehicle_id}",
        help="Stored with the selected test context. Bharat Stage is not a trained model feature in the supplied dataset.",
    )
    if st.session_state.get("active_vehicle_id") != vehicle_id:
        st.session_state["active_vehicle_id"] = vehicle_id
        st.session_state["compression_ratio"] = float(vehicle["compression_ratio"])
        st.session_state["torque_nm"] = float(vehicle["default_torque_nm"])
        st.session_state["speed_rpm"] = float(vehicle["default_speed_rpm"])
        for feature in COMPOSITION_FEATURES:
            st.session_state[feature] = 0.0
        st.session_state["gasoline_pct"] = float(vehicle["default_gasoline_pct"])
        st.session_state["ethanol_pct"] = float(vehicle["default_ethanol_pct"])
        for result_key in [
            "prediction_result",
            "experimental_prediction_result",
            "optimization_result",
            "simulation_result",
        ]:
            st.session_state.pop(result_key, None)

    st.markdown(
        f"""<div class="vehicle-card">
        <strong>{vehicle['company']} {vehicle['model']}</strong><br>
        CR: {vehicle['compression_ratio']:.1f} &nbsp;•&nbsp; {vehicle['category']}<br>
        {vehicle['fuel_blend_octane']} &nbsp;•&nbsp; Test stage: {bharat_stage}
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("Torque and RPM are not present in the supplied vehicle table. Defaults are dataset starting points, not claimed vehicle specifications.")

    page = st.radio(
        "Workspace",
        [
            "Prediction",
            "Experimental prediction",
            "Optimization",
            "Blend simulation",
            "Model analytics",
            "Data quality",
        ],
        label_visibility="collapsed",
    )


def engine_controls(prefix: str = "") -> tuple[float, float, float]:
    col1, col2, col3 = st.columns(3)
    with col1:
        cr = st.number_input(
            "Compression ratio",
            min_value=4.0,
            max_value=15.0,
            step=0.1,
            key="compression_ratio" if not prefix else f"{prefix}_compression_ratio",
            value=None if not prefix else float(vehicle["compression_ratio"]),
        )
    with col2:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.1,
            max_value=30.0,
            step=0.1,
            key="torque_nm" if not prefix else f"{prefix}_torque_nm",
            value=None if not prefix else float(vehicle["default_torque_nm"]),
        )
    with col3:
        rpm = st.number_input(
            "Speed (RPM)",
            min_value=500.0,
            max_value=12000.0,
            step=100.0,
            key="speed_rpm" if not prefix else f"{prefix}_speed_rpm",
            value=None if not prefix else float(vehicle["default_speed_rpm"]),
        )
    return float(torque), float(rpm), float(cr)


def vehicle_operating_controls(prefix: str) -> tuple[float, float, float, str]:
    """Operating inputs for recommendation/simulation with optional manual RPM."""

    st.info(
        f"Selected vehicle: **{vehicle['company']} {vehicle['model']}** · "
        f"Documented CR: **{vehicle['compression_ratio']:.1f}** · "
        f"**{bharat_stage}** · {vehicle['category']}"
    )
    st.caption(
        "Vehicle selection sets the documented CR. Adjust torque and RPM for the test point."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        compression_ratio = st.number_input(
            "Compression ratio",
            min_value=4.0,
            max_value=15.0,
            value=float(vehicle["compression_ratio"]),
            step=0.1,
            key=f"{prefix}_compression_ratio_{vehicle_id}",
            help="Prefilled from the selected vehicle; you may change it for a controlled test point.",
        )
    with col2:
        torque_nm = st.number_input(
            "Torque (Nm)",
            min_value=0.1,
            max_value=30.0,
            value=float(vehicle["default_torque_nm"]),
            step=0.1,
            key=f"{prefix}_torque_nm_{vehicle_id}",
        )
    with col3:
        use_manual_rpm = st.checkbox(
            "Enter RPM manually",
            value=False,
            key=f"{prefix}_manual_rpm_{vehicle_id}",
        )
        if use_manual_rpm:
            speed_rpm = st.number_input(
                "Speed (RPM)",
                min_value=500.0,
                max_value=12000.0,
                value=float(vehicle["default_speed_rpm"]),
                step=100.0,
                key=f"{prefix}_speed_rpm_{vehicle_id}",
            )
            rpm_source = "User-entered RPM"
        else:
            speed_rpm = float(vehicle["default_speed_rpm"])
            st.metric("RPM used", f"{speed_rpm:.0f}")
            st.caption("Optional RPM omitted; using the dataset default.")
            rpm_source = "Dataset default RPM"
    return float(torque_nm), float(speed_rpm), float(compression_ratio), rpm_source


def show_operating_point_status(
    torque_nm: float,
    speed_rpm: float,
    compression_ratio: float,
) -> None:
    selected_values = {
        "torque_nm": torque_nm,
        "speed_rpm": speed_rpm,
        "compression_ratio": compression_ratio,
    }
    for feature, value in selected_values.items():
        limits = bundle["model_ranges"][feature]
        if float(value) < float(limits["min"]) or float(value) > float(limits["max"]):
            bounded_value = min(max(float(value), float(limits["min"])), float(limits["max"]))
            st.warning(
                f"{feature.replace('_', ' ').title()} {value:.2f} is outside the training "
                f"range ({limits['min']:.2f}–{limits['max']:.2f}). Neural inference will use "
                f"the nearest trained edge ({bounded_value:.2f}) and the recommendation will be "
                "labelled exploratory."
            )
    known_cr = [float(value) for value in bundle.get("known_compression_ratios", [])]
    if known_cr:
        nearest_cr = min(known_cr, key=lambda value: abs(value - compression_ratio))
        if abs(nearest_cr - compression_ratio) > 0.05:
            st.caption(
                f"CR {compression_ratio:.2f} was not directly tested; nearest measured CR is {nearest_cr:.2f}."
            )


def additive_selector(label: str, key: str) -> str | None:
    options = [*ADDITIVE_LABELS, None]
    return st.selectbox(
        label,
        options,
        format_func=lambda value: (
            ADDITIVE_LABELS[value]
            if value is not None
            else "No additive — gasoline + ethanol only"
        ),
        key=key,
    )


def fixed_component_controls(prefix: str) -> tuple[float | None, float | None, bool, str]:
    """Render mutually exclusive fixed-ethanol/fixed-gasoline controls."""

    st.markdown("##### Optional fixed blend value")
    st.caption(
        "Ethanol is fixed by default, so only the selected additive varies and gasoline "
        "automatically balances the blend. You may instead fix gasoline, or turn both toggles off."
    )
    fixed_col1, fixed_col2 = st.columns(2)
    with fixed_col1:
        fix_ethanol = st.toggle(
            "Fix ethanol value",
            value=True,
            key=f"{prefix}_fix_ethanol_{vehicle_id}",
        )
        fixed_ethanol = (
            float(
                st.number_input(
                    "Fixed ethanol (%)",
                    min_value=0.0,
                    max_value=25.0,
                    value=10.0,
                    step=0.5,
                    key=f"{prefix}_fixed_ethanol_{vehicle_id}",
                )
            )
            if fix_ethanol
            else None
        )
    with fixed_col2:
        fix_gasoline = st.toggle(
            "Fix gasoline value",
            value=False,
            key=f"{prefix}_fix_gasoline_{vehicle_id}",
        )
        fixed_gasoline = (
            float(
                st.number_input(
                    "Fixed gasoline (%)",
                    min_value=ABSOLUTE_MIN_GASOLINE_PCT,
                    max_value=100.0,
                    value=85.0,
                    step=0.5,
                    key=f"{prefix}_fixed_gasoline_{vehicle_id}",
                )
            )
            if fix_gasoline
            else None
        )

    invalid = bool(fix_ethanol and fix_gasoline)
    if invalid:
        st.error("Fix only one value: turn off either the ethanol toggle or the gasoline toggle.")
        description = "Invalid fixed-value selection"
    elif fixed_ethanol is not None:
        description = f"Ethanol fixed at {fixed_ethanol:.1f}%; only the selected additive varies"
    elif fixed_gasoline is not None:
        description = f"Gasoline fixed at {fixed_gasoline:.1f}%; only the selected additive varies"
    else:
        description = "No base component fixed; ethanol and the selected additive may vary"
    return fixed_ethanol, fixed_gasoline, invalid, description


def blend_controls() -> dict[str, float]:
    labels = {
        "gasoline_pct": "Gasoline (%)",
        "ethanol_pct": "Ethanol (%)",
        "pentanol_pct": "Pentanol (%)",
        "propanol_pct": "Propanol (%)",
        "butanol_pct": "Butanol (%)",
        "n_methylaniline_pct": "n-Methylaniline (%)",
    }
    columns = st.columns(3)
    result = {}
    for index, feature in enumerate(COMPOSITION_FEATURES):
        with columns[index % 3]:
            result[feature] = float(
                st.number_input(labels[feature], min_value=0.0, max_value=100.0, step=1.0, key=feature)
            )
    total = blend_total(result)
    if abs(total - 100.0) <= 0.01:
        st.success(f"Fuel blend total: {total:.2f}%")
    else:
        st.error(f"Fuel blend total: {total:.2f}% — it must equal 100.00%")
    return result


def experimental_engine_controls() -> tuple[float, float, float]:
    """Collect direct model inputs without applying training-range UI bounds."""

    col1, col2, col3 = st.columns(3)
    with col1:
        compression_ratio = st.number_input(
            "Compression ratio",
            value=float(vehicle["compression_ratio"]),
            step=0.1,
            key=f"experimental_compression_ratio_{vehicle_id}",
        )
    with col2:
        torque_nm = st.number_input(
            "Torque (Nm)",
            value=float(vehicle["default_torque_nm"]),
            step=0.1,
            key=f"experimental_torque_nm_{vehicle_id}",
        )
    with col3:
        speed_rpm = st.number_input(
            "Speed (RPM)",
            value=float(vehicle["default_speed_rpm"]),
            step=100.0,
            key=f"experimental_speed_rpm_{vehicle_id}",
        )
    return float(torque_nm), float(speed_rpm), float(compression_ratio)


def experimental_blend_controls() -> dict[str, float]:
    """Collect a valid raw-model blend under page-specific widget keys."""

    labels = {
        "gasoline_pct": "Gasoline (%)",
        "ethanol_pct": "Ethanol (%)",
        "pentanol_pct": "Pentanol (%)",
        "propanol_pct": "Propanol (%)",
        "butanol_pct": "Butanol (%)",
        "n_methylaniline_pct": "n-Methylaniline (%)",
    }
    defaults = {feature: 0.0 for feature in COMPOSITION_FEATURES}
    defaults["gasoline_pct"] = float(vehicle["default_gasoline_pct"])
    defaults["ethanol_pct"] = float(vehicle["default_ethanol_pct"])
    columns = st.columns(3)
    result: dict[str, float] = {}
    for index, feature in enumerate(COMPOSITION_FEATURES):
        with columns[index % 3]:
            result[feature] = float(
                st.number_input(
                    labels[feature],
                    min_value=0.0,
                    max_value=100.0,
                    value=defaults[feature],
                    step=1.0,
                    key=f"experimental_{feature}_{vehicle_id}",
                )
            )
    total = blend_total(result)
    if abs(total - 100.0) <= 0.01:
        st.success(f"Fuel blend total: {total:.2f}%")
    else:
        st.error(f"Fuel blend total: {total:.2f}% — it must equal 100.00%")
    return result


def request_signature(kind: str, **values: object) -> str:
    """Create a stable fingerprint so changed controls cannot show stale results."""

    def normalize(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, float):
            return round(value, 8)
        return value

    return json.dumps(
        {"kind": kind, **{key: normalize(value) for key, value in values.items()}},
        sort_keys=True,
        separators=(",", ":"),
    )


def show_prediction(result: dict) -> None:
    prediction = result["predictions"]
    intervals = result["prediction_intervals_approx_90pct"]
    formats = {
        "bte_pct": ".2f",
        "bsfc_g_kwh": ".1f",
        "co_vol_pct": ".3f",
        "hc_ppm": ".1f",
    }
    calculation = result["bte_calculation"]
    output_cols = st.columns(4)
    output_targets = ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]
    for column, target in zip(output_cols, output_targets):
        low, high = intervals[target]
        label = "Reported BTE (%)" if target == "bte_pct" else TARGET_LABELS[target]
        with column:
            st.metric(label, format(prediction[target], formats[target]))
            st.caption(f"Approx. 90% band: {low:.2f}–{high:.2f}")

    if result["bte_calculation_applied"]:
        with st.expander("BTE calculation details"):
            detail_cols = st.columns(3)
            detail_cols[0].metric(
                "Active BTE range (%)",
                f"{calculation['range_min_pct']:.0f}–{calculation['range_max_pct']:.0f}",
            )
            detail_cols[0].caption(f"Rule: {calculation['rule_label']}")
            detail_cols[1].metric(
                "Base normalized BTE (%)",
                f"{calculation['base_bte_pct']:.2f}",
            )
            detail_cols[2].metric(
                "Normalized dataset position",
                f"q = {calculation['normalized_score']:.4f}",
            )
            st.info(
                f"Final BTE = {calculation['range_min_pct']:.0f} + "
                f"{calculation['normalized_score']:.4f} × "
                f"({calculation['range_max_pct']:.0f} − {calculation['range_min_pct']:.0f}) "
                f"= {calculation['final_bte_pct']:.4f}%."
            )
    if result["n_methylaniline_cap_active"]:
        cap_message = "was applied" if result["n_methylaniline_cap_applied"] else "is active"
        st.success(
            f"Global n-Methylaniline BTE ceiling {cap_message}: reported BTE cannot exceed 37%."
        )

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Estimated fuel properties")
        properties = result["estimated_fuel_properties"]
        property_table = pd.DataFrame(
            {
                "Property": [PROPERTY_LABELS[name] for name in properties],
                "Estimate": [
                    f"{properties[name]:.3e}" if name == "viscosity_m2_s" else f"{properties[name]:.3f}"
                    for name in properties
                ],
            }
        )
        st.dataframe(property_table, hide_index=True, width="stretch")
        st.caption("CV is shown in MJ/kg. The earlier prototype label 'kJ/kg' was incorrect for this dataset.")
    with right:
        st.subheader("Applicability")
        status = "Within empirical support" if result["within_blend_support"] else "Outside validated blend support"
        st.metric("Blend-support status", status)
        st.caption(
            f"Nearest distance {result['support_distance']:.2f}; validated threshold {result['support_threshold']:.2f}."
        )
        heat_balance_gap = abs(
            result["simple_heat_balance_bte_pct"] - result["raw_predictions"]["bte_pct"]
        )
        st.metric(
            "Independent heat-balance check",
            "Review" if heat_balance_gap > 5.0 else "Consistent",
        )
        st.caption("Diagnostic only; it is not the reported BTE prediction.")

    for warning in result["warnings"]:
        st.warning(warning)


if page == "Prediction":
    st.subheader("1. Operating conditions")
    torque_nm, speed_rpm, compression_ratio = engine_controls()
    st.subheader("2. Fuel composition")
    blend = blend_controls()
    prediction_signature = request_signature(
        "prediction",
        vehicle_id=vehicle_id,
        bharat_stage=bharat_stage,
        blend=blend,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )
    if st.button("Predict performance & emissions", type="primary", width="stretch"):
        try:
            result = predict_performance(
                bundle, blend, torque_nm, speed_rpm, compression_ratio
            )
            result["request_signature"] = prediction_signature
            st.session_state["prediction_result"] = result
        except InputValidationError as exc:
            st.error(str(exc))
    prediction_result = st.session_state.get("prediction_result")
    if (
        prediction_result is not None
        and prediction_result.get("request_signature") != prediction_signature
    ):
        st.session_state.pop("prediction_result", None)
        prediction_result = None
        st.info("Inputs changed. Run the prediction again to calculate the new test point.")
    if prediction_result is not None:
        st.divider()
        st.subheader("Prediction")
        show_prediction(prediction_result)


elif page == "Experimental prediction":
    st.markdown('<div class="raw-badge">Raw experimental channel</div>', unsafe_allow_html=True)
    st.subheader("Direct dataset-model prediction")
    st.caption("Raw network output · No BTE conversion · No output caps · No input clamping")
    st.markdown("#### Operating inputs")
    torque_nm, speed_rpm, compression_ratio = experimental_engine_controls()
    st.markdown("#### Fuel composition")
    experimental_blend = experimental_blend_controls()
    experimental_signature = request_signature(
        "experimental_prediction",
        vehicle_id=vehicle_id,
        bharat_stage=bharat_stage,
        blend=experimental_blend,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        compression_ratio=compression_ratio,
    )
    if st.button(
        "Run raw experimental prediction",
        type="primary",
        width="stretch",
    ):
        try:
            raw_result = predict_raw_performance(
                bundle,
                experimental_blend,
                torque_nm,
                speed_rpm,
                compression_ratio,
            )
            raw_result["request_signature"] = experimental_signature
            st.session_state["experimental_prediction_result"] = raw_result
        except InputValidationError as exc:
            st.error(str(exc))

    experimental_result = st.session_state.get("experimental_prediction_result")
    if (
        experimental_result is not None
        and experimental_result.get("request_signature") != experimental_signature
    ):
        st.session_state.pop("experimental_prediction_result", None)
        experimental_result = None
        st.info("Inputs changed. Run the raw prediction again.")
    if experimental_result is not None:
        st.divider()
        raw_predictions = experimental_result["predictions"]
        raw_cols = st.columns(4)
        raw_formats = {
            "bte_pct": ".4f",
            "bsfc_g_kwh": ".3f",
            "co_vol_pct": ".4f",
            "hc_ppm": ".3f",
        }
        for column, target in zip(
            raw_cols,
            ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"],
        ):
            column.metric(
                f"Raw {TARGET_LABELS[target]}",
                format(raw_predictions[target], raw_formats[target]),
            )
        if experimental_result["outside_training_ranges"]:
            readable = ", ".join(
                name.replace("_pct", "").replace("_", " ").title()
                for name in experimental_result["outside_training_ranges"]
            )
            st.warning(f"Outside raw-data range: {readable}. Values are shown without correction.")
        support_label = (
            "Within blend support"
            if experimental_result["within_blend_support"]
            else "Outside blend support"
        )
        st.caption(
            f"{support_label} · composition distance "
            f"{experimental_result['support_distance']:.3f}"
        )


elif page == "Optimization":
    st.subheader("Vehicle-specific best blend recommendation")
    st.caption(
        "Enter the exact performance profile required for this vehicle and test point."
    )
    torque_nm, speed_rpm, compression_ratio, rpm_source = vehicle_operating_controls("opt")
    show_operating_point_status(torque_nm, speed_rpm, compression_ratio)

    st.markdown("#### Blend search space")
    search_col1, search_col2, search_col3 = st.columns(3)
    with search_col1:
        selected_additive = additive_selector(
            "Allowed additive",
            key=f"opt_additive_{vehicle_id}",
        )
    with search_col2:
        max_additive = (
            st.slider(
                "Maximum selected additive (%)",
                0.5,
                20.0,
                5.0,
                0.5,
                key=f"opt_max_additive_{vehicle_id}",
            )
            if selected_additive is not None
            else 0.0
        )
        if selected_additive is None:
            st.metric("Maximum additive", "0%")
            st.caption("No additive selected.")
    with search_col3:
        grid_step = st.selectbox(
            "Blend search resolution (%)",
            [0.5, 1.0, 2.5],
            index=0,
            key=f"opt_grid_step_{vehicle_id}",
        )

    fixed_ethanol, fixed_gasoline, fixed_selection_invalid, search_mode = (
        fixed_component_controls("opt")
    )

    limit_col1, limit_col2 = st.columns(2)
    with limit_col1:
        if fixed_gasoline is None:
            min_gasoline = st.slider(
                "Minimum gasoline (%)",
                ABSOLUTE_MIN_GASOLINE_PCT,
                100.0,
                ABSOLUTE_MIN_GASOLINE_PCT,
                0.5,
                key=f"opt_min_gasoline_{vehicle_id}",
            )
        else:
            min_gasoline = ABSOLUTE_MIN_GASOLINE_PCT
            st.metric("Hard minimum gasoline", f"{ABSOLUTE_MIN_GASOLINE_PCT:.0f}%")
    with limit_col2:
        if fixed_ethanol is None and fixed_gasoline is None:
            max_ethanol = st.slider(
                "Maximum ethanol when not fixed (%)",
                0.0,
                25.0,
                20.0,
                0.5,
                key=f"opt_max_ethanol_{vehicle_id}",
            )
        else:
            max_ethanol = 25.0
            st.metric("Blend balance", "Automatic")
            st.caption("The unfixed base component maintains a 100% blend total.")
    st.info(
        f"{search_mode}. Every candidate must contain at least "
        f"{ABSOLUTE_MIN_GASOLINE_PCT:.0f}% gasoline, and gasoline must be greater "
        "than both ethanol and the selected additive."
    )
    if selected_additive == "n_methylaniline_pct":
        st.warning(
            "n-Methylaniline BTE is capped at 37%. It should only be evaluated in an "
            "approved controlled-laboratory study."
        )

    st.markdown("#### Exact performance values")
    target_col1, target_col2, target_col3, target_col4 = st.columns(4)
    target_values = {
        "bte_pct": float(
            target_col1.number_input(
                "BTE (%)",
                value=28.26,
                step=0.01,
                format="%.2f",
                key=f"opt_target_bte_pct_{vehicle_id}",
            )
        ),
        "bsfc_g_kwh": float(
            target_col2.number_input(
                "BSFC (g/kWh)",
                value=549.1,
                step=0.1,
                format="%.1f",
                key=f"opt_target_bsfc_g_kwh_{vehicle_id}",
            )
        ),
        "co_vol_pct": float(
            target_col3.number_input(
                "CO (vol.%)",
                value=3.007,
                step=0.001,
                format="%.3f",
                key=f"opt_target_co_vol_pct_{vehicle_id}",
            )
        ),
        "hc_ppm": float(
            target_col4.number_input(
                "HC (ppm)",
                value=354.5,
                step=0.1,
                format="%.1f",
                key=f"opt_target_hc_ppm_{vehicle_id}",
            )
        ),
    }
    st.caption(
        "All four values are exact inputs. The recommendation is the supported blend whose "
        "predicted four-output profile is closest to the complete entered profile."
    )
    optimization_signature = request_signature(
        "optimization",
        vehicle_id=vehicle_id,
        bharat_stage=bharat_stage,
        selected_additive=selected_additive,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        rpm_source=rpm_source,
        compression_ratio=compression_ratio,
        fixed_ethanol=fixed_ethanol,
        fixed_gasoline=fixed_gasoline,
        min_gasoline=min_gasoline,
        max_ethanol=max_ethanol,
        max_additive=max_additive,
        grid_step=float(grid_step),
        target_values=target_values,
    )

    if st.button(
        "Recommend best blend",
        type="primary",
        width="stretch",
        disabled=fixed_selection_invalid,
    ):
        try:
            with st.spinner("Evaluating single-additive blends..."):
                best, ranked_candidates, supported = optimize_single_additive_blend(
                    bundle=bundle,
                    selected_additive=selected_additive,
                    torque_nm=torque_nm,
                    speed_rpm=speed_rpm,
                    compression_ratio=compression_ratio,
                    target_values=target_values,
                    min_gasoline_pct=min_gasoline,
                    max_ethanol_pct=max_ethanol,
                    max_additive_pct=max_additive,
                    grid_step_pct=float(grid_step),
                    fixed_ethanol_pct=fixed_ethanol,
                    fixed_gasoline_pct=fixed_gasoline,
                )
            st.session_state["optimization_result"] = {
                "best": best,
                "ranked_candidates": ranked_candidates,
                "supported": supported,
                "vehicle": f"{vehicle['company']} {vehicle['model']}",
                "selected_additive": selected_additive,
                "torque_nm": torque_nm,
                "speed_rpm": speed_rpm,
                "compression_ratio": compression_ratio,
                "rpm_source": rpm_source,
                "search_mode": search_mode,
                "target_values": target_values,
                "bharat_stage": bharat_stage,
                "request_signature": optimization_signature,
            }
        except Exception as exc:
            st.error(str(exc))

    optimization_result = st.session_state.get("optimization_result")
    if (
        optimization_result is not None
        and optimization_result.get("request_signature") != optimization_signature
    ):
        st.session_state.pop("optimization_result", None)
        optimization_result = None
        st.info("Optimization inputs changed. Run the recommendation again for the new settings.")
    if optimization_result is not None:
        best = optimization_result["best"]
        ranked_candidates = optimization_result["ranked_candidates"]
        supported = optimization_result["supported"]
        result_additive = optimization_result["selected_additive"]
        additive_name = (
            ADDITIVE_LABELS[result_additive] if result_additive is not None else None
        )
        st.success(
            f"Closest complete performance match selected from {len(supported)} "
            "empirically supported blend candidates."
        )
        st.caption(
            f"{optimization_result['vehicle']} · Torque {optimization_result['torque_nm']:.2f} Nm · "
            f"CR {optimization_result['compression_ratio']:.2f} · "
            f"RPM {optimization_result['speed_rpm']:.0f} ({optimization_result['rpm_source']}) · "
            f"{optimization_result['bharat_stage']}"
        )
        st.caption(optimization_result.get("search_mode", "Blend constraints applied"))
        st.caption(
            "The four entered values participate equally. A zero-additive candidate is included "
            "so the selected additive is never forced into the recommendation."
        )
        st.subheader("Recommended blend")
        result_blend_features = ["gasoline_pct", "ethanol_pct"]
        if result_additive is not None:
            result_blend_features.append(result_additive)
        blend_cols = st.columns(len(result_blend_features))
        for column, feature in zip(blend_cols, result_blend_features):
            label = ADDITIVE_LABELS.get(
                feature,
                feature.replace("_pct", "").replace("_", " ").title(),
            )
            column.metric(label, f"{best[feature]:.1f}%")
        st.caption(
            f"Only gasoline, ethanol, and {additive_name or 'no additive'} were allowed; "
            "every other additive was fixed at 0%."
        )
        output_cols = st.columns(4)
        for column, target in zip(
            output_cols,
            ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"],
        ):
            label = "Reported BTE (%)" if target == "bte_pct" else TARGET_LABELS[target]
            column.metric(label, f"{best[target]:.2f}")
        if bool(best["bte_calculation_applied"]):
            st.info(
                f"Optimizer BTE rule: {best['bte_rule_label']}. Final = "
                f"{best['bte_rule_min_pct']:.0f} + {best['bte_normalized_score']:.4f} × "
                f"({best['bte_rule_max_pct']:.0f} − {best['bte_rule_min_pct']:.0f}) "
                f"= {best['bte_calculated_pct']:.4f}%."
            )
        if bool(best.get("n_methylaniline_cap_active", False)):
            st.success("Global n-Methylaniline ceiling is active: reported BTE cannot exceed 37%.")

        quality_cols = st.columns(4)
        quality_cols[0].metric(
            "Profile match",
            f"{best['target_match_score_pct']:.1f}%",
        )
        quality_cols[1].metric(
            "Recommendation confidence",
            str(best["recommendation_confidence"]),
        )
        quality_cols[2].metric(
            "Average model-scale gap",
            f"{best['mean_normalized_target_gap']:.2f}",
        )
        quality_cols[3].metric(
            "Blend-support confidence",
            f"{best['support_confidence'] * 100:.1f}%",
        )
        st.caption(str(best["recommendation_confidence_reason"]))

        if bool(best.get("operating_point_bounded", False)):
            st.warning(
                "This recommendation is exploratory because one or more requested engine inputs "
                "were outside training support. Neural inputs used: "
                f"torque {best['model_torque_nm']:.2f} Nm, RPM {best['model_speed_rpm']:.0f}, "
                f"CR {best['model_compression_ratio']:.2f}."
            )
        if bool(best.get("performance_output_bound_applied", False)):
            st.warning(
                "At least one internal performance output reached the measured target envelope. "
                "Treat the recommendation as exploratory and confirm it experimentally."
            )

        st.subheader("Why this blend was selected")
        explanation_rows = []
        for target in ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]:
            explanation_rows.append(
                {
                    "Measure": TARGET_LABELS[target],
                    "Entered value": best[f"{target}_entered_value"],
                    "Predicted value": best[target],
                    "Absolute gap": best[f"{target}_absolute_gap"],
                    "Gap / model error": best[f"{target}_normalized_gap"],
                    "Individual match (%)": 100.0 * best[f"{target}_match_utility"],
                }
            )
        st.dataframe(pd.DataFrame(explanation_rows), hide_index=True, width="stretch")

        if bool(best.get("baseline_available", False)) and result_additive is not None:
            st.subheader("Zero-additive comparison")
            baseline_cols = st.columns(3)
            baseline_cols[0].metric(
                "Recommended profile match",
                f"{best['target_match_score_pct']:.1f}%",
            )
            baseline_cols[1].metric(
                "Zero-additive profile match",
                f"{best['baseline_target_match_score_pct']:.1f}%",
            )
            baseline_cols[2].metric(
                "Match difference",
                f"{best['target_match_score_delta_vs_baseline_pct']:+.1f} points",
            )
            st.caption(
                f"Baseline blend: gasoline {best['baseline_gasoline_pct']:.1f}%, "
                f"ethanol {best['baseline_ethanol_pct']:.1f}%, selected additive 0.0%."
            )
            if float(best[result_additive]) <= 1e-12:
                st.info(
                    "The zero-additive candidate is the closest match to the entered values; the "
                    "selected additive is not forced into the recommendation."
                )

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        points = ax.scatter(
            ranked_candidates["bsfc_g_kwh"],
            ranked_candidates["bte_pct"],
            c=ranked_candidates["target_match_score_pct"],
            s=30 + 80 * ranked_candidates["support_confidence"],
            cmap="viridis",
            alpha=0.75,
        )
        ax.scatter([best["bsfc_g_kwh"]], [best["bte_pct"]], marker="*", s=240, color="#f59e0b", edgecolor="#10253f", label="Recommended")
        entered_profile = optimization_result["target_values"]
        ax.scatter([entered_profile["bsfc_g_kwh"]], [entered_profile["bte_pct"]], marker="X", s=150, color="#ff2a2a", edgecolor="#ffffff", label="Entered BTE / BSFC")
        bte_axis_label = "Rule-guided BTE (%)" if bool(best["bte_calculation_applied"]) else "BTE (%)"
        ax.set(xlabel="BSFC (g/kWh)", ylabel=bte_axis_label, title="Exact-profile candidate matching")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        colorbar = fig.colorbar(points, ax=ax)
        colorbar.set_label("Four-output profile match (%)")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        display_columns = COMPOSITION_FEATURES + [
            "bte_pct",
            "bte_rule_min_pct",
            "bte_rule_max_pct",
            "bte_rule_id",
            "n_methylaniline_cap_active",
            "n_methylaniline_cap_applied",
            "bsfc_g_kwh",
            "co_vol_pct",
            "hc_ppm",
            "support_distance",
            "support_confidence",
            "target_match_score_pct",
            "mean_normalized_target_gap",
            "recommendation_score_pct",
        ]
        for target in ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]:
            display_columns.extend(
                [f"{target}_entered_value", f"{target}_absolute_gap"]
            )
        alternatives = ranked_candidates[display_columns].head(25).copy()
        st.subheader("Top alternative blends")
        st.dataframe(alternatives, hide_index=True, width="stretch")
        recommendation_download = alternatives.copy()
        recommendation_download.insert(0, "vehicle", optimization_result["vehicle"])
        recommendation_download.insert(1, "bharat_stage", optimization_result["bharat_stage"])
        recommendation_download.insert(2, "torque_nm", optimization_result["torque_nm"])
        recommendation_download.insert(3, "speed_rpm", optimization_result["speed_rpm"])
        recommendation_download.insert(
            4,
            "compression_ratio",
            optimization_result["compression_ratio"],
        )
        st.download_button(
            "Download recommendation alternatives (CSV)",
            data=recommendation_download.to_csv(index=False).encode("utf-8"),
            file_name="geb_ai_blend_recommendations.csv",
            mime="text/csv",
            width="stretch",
        )


elif page == "Blend simulation":
    st.subheader("Multi-blend performance simulation")
    st.caption("Set a complete fixed blend or sweep one additive through a user-defined range.")
    torque_nm, speed_rpm, compression_ratio, rpm_source = vehicle_operating_controls("sim")
    show_operating_point_status(torque_nm, speed_rpm, compression_ratio)

    st.markdown("#### User-defined simulation setup")
    setup_col1, setup_col2 = st.columns([1.25, 1])
    with setup_col1:
        simulation_setup = st.selectbox(
            "Simulation setup",
            [
                "Fixed blend — set every value",
                "Additive range — fix ethanol",
                "Additive range — fix gasoline",
            ],
            key=f"sim_setup_{vehicle_id}",
        )
    with setup_col2:
        simulation_additive = additive_selector(
            "Selected additive",
            key=f"sim_additive_{vehicle_id}",
        )

    fixed_blend_values: dict[str, float] | None = None
    if simulation_setup == "Fixed blend — set every value":
        fixed_additive_key = simulation_additive or "none"
        fixed_col1, fixed_col2, fixed_col3 = st.columns(3)
        with fixed_col1:
            fixed_blend_gasoline = float(
                st.number_input(
                    "Fixed gasoline (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=85.0 if simulation_additive is not None else 90.0,
                    step=0.5,
                    key=f"sim_full_gasoline_{vehicle_id}_{fixed_additive_key}",
                )
            )
        with fixed_col2:
            fixed_blend_ethanol = float(
                st.number_input(
                    "Fixed ethanol (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.5,
                    key=f"sim_full_ethanol_{vehicle_id}_{fixed_additive_key}",
                )
            )
        with fixed_col3:
            fixed_blend_additive = (
                float(
                    st.number_input(
                        "Fixed additive (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=5.0,
                        step=0.5,
                        key=f"sim_full_additive_{vehicle_id}_{fixed_additive_key}",
                    )
                )
                if simulation_additive is not None
                else 0.0
            )
            if simulation_additive is None:
                st.metric("Fixed additive", "0%")
        fixed_total = fixed_blend_gasoline + fixed_blend_ethanol + fixed_blend_additive
        simulation_fixed_invalid = bool(
            abs(fixed_total - 100.0) > 0.01
            or fixed_blend_gasoline < ABSOLUTE_MIN_GASOLINE_PCT
            or fixed_blend_gasoline <= fixed_blend_ethanol
            or fixed_blend_gasoline <= fixed_blend_additive
        )
        if simulation_fixed_invalid:
            st.error(
                f"Blend total: {fixed_total:.2f}%. Use a 100% total with gasoline at least "
                f"{ABSOLUTE_MIN_GASOLINE_PCT:.0f}% and greater than ethanol/additive."
            )
        else:
            st.success(f"Fixed blend ready · total {fixed_total:.2f}%")
        fixed_blend_values = {
            "gasoline_pct": fixed_blend_gasoline,
            "ethanol_pct": fixed_blend_ethanol,
            "additive_pct": fixed_blend_additive,
        }
        additive_range = (fixed_blend_additive, fixed_blend_additive)
        ethanol_range = (fixed_blend_ethanol, fixed_blend_ethanol)
        simulation_fixed_ethanol = None
        simulation_fixed_gasoline = None
        simulation_min_gasoline = ABSOLUTE_MIN_GASOLINE_PCT
        simulation_step = 1.0
        simulation_mode = "All blend percentages fixed by the user"
        simulation_fixed_component = "all"
    else:
        range_col1, range_col2, range_col3 = st.columns(3)
        with range_col1:
            additive_range = (
                st.slider(
                    "Selected additive range (%)",
                    0.0,
                    20.0,
                    (0.0, 5.0),
                    0.5,
                    key=f"sim_additive_range_{vehicle_id}",
                )
                if simulation_additive is not None
                else (0.0, 0.0)
            )
            if simulation_additive is None:
                st.metric("Additive range", "0%")
        with range_col2:
            simulation_step = st.selectbox(
                "Simulation resolution (%)",
                [0.5, 1.0, 2.5, 5.0],
                index=1,
                key=f"sim_grid_step_{vehicle_id}",
            )
        if simulation_setup == "Additive range — fix ethanol":
            with range_col3:
                simulation_fixed_ethanol = float(
                    st.number_input(
                        "Fixed ethanol (%)",
                        min_value=0.0,
                        max_value=25.0,
                        value=10.0,
                        step=0.5,
                        key=f"sim_range_ethanol_{vehicle_id}",
                    )
                )
            gasoline_limit_col, balance_col = st.columns(2)
            with gasoline_limit_col:
                simulation_min_gasoline = float(
                    st.number_input(
                        "Minimum gasoline (%)",
                        min_value=ABSOLUTE_MIN_GASOLINE_PCT,
                        max_value=100.0,
                        value=ABSOLUTE_MIN_GASOLINE_PCT,
                        step=0.5,
                        key=f"sim_range_min_gasoline_{vehicle_id}",
                    )
                )
            with balance_col:
                st.metric("Balancing component", "Gasoline")
            simulation_fixed_gasoline = None
            ethanol_range = (simulation_fixed_ethanol, simulation_fixed_ethanol)
            simulation_mode = (
                f"Ethanol fixed at {simulation_fixed_ethanol:.1f}%; gasoline balances each blend"
            )
            simulation_fixed_component = "ethanol"
        else:
            with range_col3:
                simulation_fixed_gasoline = float(
                    st.number_input(
                        "Fixed gasoline (%)",
                        min_value=ABSOLUTE_MIN_GASOLINE_PCT,
                        max_value=100.0,
                        value=85.0,
                        step=0.5,
                        key=f"sim_range_gasoline_{vehicle_id}",
                    )
                )
            balance_col1, balance_col2 = st.columns(2)
            with balance_col1:
                st.metric("Gasoline constraint", f"{simulation_fixed_gasoline:.1f}% fixed")
            with balance_col2:
                st.metric("Balancing component", "Ethanol")
            simulation_fixed_ethanol = None
            simulation_min_gasoline = ABSOLUTE_MIN_GASOLINE_PCT
            ethanol_range = (0.0, 25.0)
            simulation_mode = (
                f"Gasoline fixed at {simulation_fixed_gasoline:.1f}%; ethanol balances each blend"
            )
            simulation_fixed_component = "gasoline"
        simulation_fixed_invalid = False

    st.caption(
        f"{simulation_mode} · gasoline remains at least "
        f"{ABSOLUTE_MIN_GASOLINE_PCT:.0f}% · every blend totals 100%."
    )
    if simulation_additive == "n_methylaniline_pct":
        st.warning(
            "n-Methylaniline BTE is capped at 37%. Simulations are research screening only "
            "and require controlled-laboratory review."
        )
    simulation_signature = request_signature(
        "simulation",
        vehicle_id=vehicle_id,
        bharat_stage=bharat_stage,
        selected_additive=simulation_additive,
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        rpm_source=rpm_source,
        compression_ratio=compression_ratio,
        simulation_setup=simulation_setup,
        fixed_ethanol=simulation_fixed_ethanol,
        fixed_gasoline=simulation_fixed_gasoline,
        fixed_blend_values=fixed_blend_values,
        min_gasoline=simulation_min_gasoline,
        ethanol_range=ethanol_range,
        additive_range=additive_range,
        grid_step=float(simulation_step),
    )

    if st.button(
        "Run blend simulation",
        type="primary",
        width="stretch",
        disabled=simulation_fixed_invalid,
    ):
        try:
            with st.spinner("Simulating blend performance..."):
                if fixed_blend_values is not None:
                    simulation = simulate_fixed_blend(
                        bundle=bundle,
                        selected_additive=simulation_additive,
                        gasoline_pct=fixed_blend_values["gasoline_pct"],
                        ethanol_pct=fixed_blend_values["ethanol_pct"],
                        additive_pct=fixed_blend_values["additive_pct"],
                        torque_nm=torque_nm,
                        speed_rpm=speed_rpm,
                        compression_ratio=compression_ratio,
                        min_gasoline_pct=simulation_min_gasoline,
                    )
                else:
                    simulation = simulate_blend_grid(
                        bundle=bundle,
                        selected_additive=simulation_additive,
                        torque_nm=torque_nm,
                        speed_rpm=speed_rpm,
                        compression_ratio=compression_ratio,
                        min_ethanol_pct=float(ethanol_range[0]),
                        max_ethanol_pct=float(ethanol_range[1]),
                        min_additive_pct=float(additive_range[0]),
                        max_additive_pct=float(additive_range[1]),
                        min_gasoline_pct=simulation_min_gasoline,
                        grid_step_pct=float(simulation_step),
                        fixed_ethanol_pct=simulation_fixed_ethanol,
                        fixed_gasoline_pct=simulation_fixed_gasoline,
                    )
            st.session_state["simulation_result"] = {
                "data": simulation,
                "vehicle": f"{vehicle['company']} {vehicle['model']}",
                "selected_additive": simulation_additive,
                "torque_nm": torque_nm,
                "speed_rpm": speed_rpm,
                "compression_ratio": compression_ratio,
                "rpm_source": rpm_source,
                "simulation_mode": simulation_mode,
                "bharat_stage": bharat_stage,
                "request_signature": simulation_signature,
                "fixed_component": simulation_fixed_component,
            }
        except Exception as exc:
            st.error(str(exc))

    simulation_result = st.session_state.get("simulation_result")
    if (
        simulation_result is not None
        and simulation_result.get("request_signature") != simulation_signature
    ):
        st.session_state.pop("simulation_result", None)
        simulation_result = None
        st.info("Simulation inputs changed. Run the simulation again for the new settings.")
    if simulation_result is not None:
        simulation = simulation_result["data"]
        result_additive = simulation_result["selected_additive"]
        best_bte = simulation.loc[simulation["bte_pct"].idxmax()]
        supported_count = int(simulation["within_validated_support"].sum())
        st.success(f"Simulated {len(simulation)} valid blends; {supported_count} are within validated support.")
        st.caption(
            f"{simulation_result['vehicle']} · Torque {simulation_result['torque_nm']:.2f} Nm · "
            f"CR {simulation_result['compression_ratio']:.2f} · "
            f"RPM {simulation_result['speed_rpm']:.0f} ({simulation_result['rpm_source']}) · "
            f"{simulation_result['bharat_stage']}"
        )
        st.caption(simulation_result.get("simulation_mode", "Blend constraints applied"))

        st.subheader("Highest-BTE simulated blend")
        best_features = ["gasoline_pct", "ethanol_pct"]
        if result_additive is not None:
            best_features.append(result_additive)
        best_blend_cols = st.columns(len(best_features))
        for column, feature in zip(best_blend_cols, best_features):
            label = ADDITIVE_LABELS.get(
                feature,
                feature.replace("_pct", "").replace("_", " ").title(),
            )
            column.metric(label, f"{best_bte[feature]:.1f}%")
        performance_cols = st.columns(4)
        for column, target in zip(
            performance_cols,
            ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"],
        ):
            label = "Reported BTE (%)" if target == "bte_pct" else TARGET_LABELS[target]
            column.metric(label, f"{best_bte[target]:.2f}")
        if bool(best_bte.get("n_methylaniline_cap_active", False)):
            st.success("Global n-Methylaniline ceiling is active: reported BTE cannot exceed 37%.")
        st.caption(
            "Highest BTE is a simulation highlight; use the Optimization page to match a complete "
            "entered BTE, BSFC, CO, and HC profile."
        )
        st.caption(
            "Highest-BTE blend support status: "
            + (
                "within validated experimental support"
                if bool(best_bte["within_validated_support"])
                else "outside validated experimental support (exploratory)"
            )
        )

        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex=True)
        plot_targets = ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]
        color_plot = None
        fixed_component = simulation_result.get("fixed_component")
        for axis, target in zip(axes.flat, plot_targets):
            if result_additive is not None and fixed_component is not None:
                axis.plot(
                    simulation[result_additive],
                    simulation[target],
                    color="#f20d18",
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                )
                x_label = f"{ADDITIVE_LABELS[result_additive]} (%)"
            elif result_additive is None:
                axis.plot(
                    simulation["ethanol_pct"],
                    simulation[target],
                    color="#f20d18",
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                )
                x_label = "Ethanol (%)"
            else:
                color_plot = axis.scatter(
                    simulation["ethanol_pct"],
                    simulation[target],
                    c=simulation[result_additive],
                    cmap="viridis",
                    s=32,
                    alpha=0.82,
                )
                x_label = "Ethanol (%)"
            axis.set(xlabel=x_label, ylabel=TARGET_LABELS[target])
            axis.grid(alpha=0.25)
        if color_plot is not None:
            colorbar = fig.colorbar(color_plot, ax=axes.ravel().tolist(), shrink=0.88)
            colorbar.set_label(f"{ADDITIVE_LABELS[result_additive]} (%)")
            fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.25)
        else:
            fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        if supported_count < len(simulation):
            st.warning(
                f"{len(simulation) - supported_count} simulated blends are outside the validated "
                "experimental-support threshold and should be treated as exploratory."
            )
        simulation_columns = COMPOSITION_FEATURES + [
            "bte_pct",
            "bsfc_g_kwh",
            "co_vol_pct",
            "hc_ppm",
            "bte_rule_id",
            "n_methylaniline_cap_active",
            "n_methylaniline_cap_applied",
            "support_distance",
            "within_validated_support",
        ]
        simulation_table = simulation[simulation_columns].copy()
        st.subheader("All simulated blend predictions")
        st.dataframe(simulation_table, hide_index=True, width="stretch")
        simulation_download = simulation_table.copy()
        simulation_download.insert(0, "vehicle", simulation_result["vehicle"])
        simulation_download.insert(1, "bharat_stage", simulation_result["bharat_stage"])
        simulation_download.insert(2, "torque_nm", simulation_result["torque_nm"])
        simulation_download.insert(3, "speed_rpm", simulation_result["speed_rpm"])
        simulation_download.insert(
            4,
            "compression_ratio",
            simulation_result["compression_ratio"],
        )
        st.download_button(
            "Download simulation results (CSV)",
            data=simulation_download.to_csv(index=False).encode("utf-8"),
            file_name="geb_ai_blend_simulation.csv",
            mime="text/csv",
            width="stretch",
        )


elif page == "Model analytics":
    st.subheader("Validation on unseen fuel-blend groups")
    selected = holdout_metrics.copy()
    selected["target"] = selected["target"].map(TARGET_LABELS)
    st.dataframe(
        selected[["target", "selected_model", "mae", "rmse", "r2", "absolute_error_p90"]],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Four-hidden-layer neural regressors were checked by grouped cross-validation on "
        "development data. These metrics are from a separate 20% holdout containing complete "
        "unseen blend groups; they do not independently validate the user-supplied BTE ranges."
    )

    for filename, caption in [
        ("actual_vs_predicted.png", "Actual vs predicted on unseen blend groups"),
        ("grouped_cv_model_comparison.png", "Grouped cross-validation model comparison"),
    ]:
        image_path = FIGURE_DIR / filename
        if image_path.exists():
            st.image(str(image_path), caption=caption, width="stretch")

    st.subheader("Feature influence")
    chosen_target = st.selectbox("Target", list(TARGET_LABELS), format_func=lambda value: TARGET_LABELS[value])
    feature_table = importance[importance["target"] == chosen_target].sort_values("importance", ascending=False)
    st.dataframe(feature_table[["feature", "importance"]], hide_index=True, width="stretch")

    st.subheader("Fuel-property estimator validation")
    property_summary = (
        property_cv_metrics.groupby("property", as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_r2=("r2", "mean"), r2_sd=("r2", "std"))
        .sort_values("mean_r2", ascending=False)
    )
    st.dataframe(property_summary, hide_index=True, width="stretch")
    if (property_summary.loc[property_summary["property"] == "ron", "mean_r2"] < 0).any():
        st.warning("RON estimation does not generalize reliably across unseen blend groups in this dataset. Treat displayed RON as a low-confidence screening estimate.")


else:
    st.subheader("Data quality and hybrid BTE calculation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source rows", audit["source_rows"])
    m2.metric("Valid model rows", audit["valid_model_rows"])
    m3.metric("Quarantined rows", audit["quarantined_rows"])
    m4.metric("Valid blend groups", audit["valid_blend_groups"])

    conflict = audit["bte_cr_ge_9_5"]
    st.info(
        f"The {conflict['rows']} cleaned records at CR ≥ 9.5 have BTE values from "
        f"{conflict['source_min_pct']:.4f}% to {conflict['source_max_pct']:.4f}%. "
        "The calculation uses the minimum and maximum at each measured high-CR level, then interpolates the source envelope for the selected vehicle CR."
    )
    st.code(
        "q = clip((Internal DNN BTE signal - Source minimum at CR) / "
        "(Source maximum at CR - Source minimum at CR), 0, 1)\n"
        "Final BTE = Active blend minimum + q * (Active blend maximum - Active blend minimum)"
    )
    st.warning(
        "The blend ranges below are user-supplied engineering rules, not replacement training labels. "
        "The internal DNN signal remains available only in audit artifacts; every user-facing page "
        "reports the bounded final BTE, including the global 37% n-Methylaniline ceiling."
    )
    st.caption(bundle["bte_calculation"]["e5_assumption"])

    ethanol_rules = pd.DataFrame(bundle["bte_calculation"]["ethanol_only_rules"])
    additive_rules = pd.DataFrame(bundle["bte_calculation"]["additive_rules"])
    rule_table = pd.concat(
        [
            ethanol_rules.assign(rule_type="Ethanol only"),
            additive_rules.assign(rule_type="E10 + 5% additive"),
        ],
        ignore_index=True,
        sort=False,
    )
    st.subheader("Active BTE range rules")
    st.dataframe(
        rule_table[["rule_type", "rule_id", "gasoline_pct", "ethanol_pct", "additive", "additive_pct", "minimum_pct", "maximum_pct"]],
        hide_index=True,
        width="stretch",
    )

    calculation_figure = FIGURE_DIR / "bte_hybrid_calculation.png"
    if calculation_figure.exists():
        st.image(str(calculation_figure), width="stretch")

    st.subheader("Quarantined source records")
    st.caption("These records are retained for engineering review and excluded from model fitting.")
    st.dataframe(quarantine.head(100), hide_index=True, width="stretch")
    with st.expander("Full audit details"):
        st.json(audit)

    st.info(
        "Use this system for comparative screening. Confirm every recommended blend through fuel-compatibility review, controlled dynamometer testing, and applicable manufacturer/regulatory checks."
    )
