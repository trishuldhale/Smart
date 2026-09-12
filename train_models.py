from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bte_calculation import ADDITIVE_RULES, ETHANOL_ONLY_RULES
from src.config import (
    ARTIFACT_DIR,
    BTE_CALCULATION_CR_THRESHOLD,
    FIGURE_DIR,
    PERFORMANCE_TARGETS,
    PROCESSED_DIR,
    PROPERTY_TARGETS,
    RAW_DATA_PATH,
    RAW_VEHICLE_PATH,
    REPORT_DIR,
    TARGET_LABELS,
)
from src.data_pipeline import extract_vehicle_database, load_and_clean_experimental_data, write_audit
from src.modeling import save_bundle, train_models


PALETTE = {
    "navy": "#10253F",
    "teal": "#0E8A83",
    "cyan": "#36C2B4",
    "orange": "#F59E0B",
    "red": "#DC5A5A",
    "grid": "#D8E1E8",
}


def _prepare_directories() -> None:
    for directory in [PROCESSED_DIR, ARTIFACT_DIR, REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _plot_bte_calculation(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    jitter = np.random.default_rng(42).normal(0, 0.025, len(data))
    high = data["compression_ratio"] >= BTE_CALCULATION_CR_THRESHOLD
    colors = np.where(high, PALETTE["teal"], "#9AA8B4")
    ax.scatter(data["compression_ratio"] + jitter, data["bte_pct"], c=colors, s=15, alpha=0.52, edgecolors="none")
    envelope = (
        data.loc[high]
        .groupby("compression_ratio")["bte_pct"]
        .agg(["min", "max"])
        .reset_index()
    )
    ax.plot(envelope["compression_ratio"], envelope["min"], color=PALETTE["orange"], marker="o", label="CR source minimum")
    ax.plot(envelope["compression_ratio"], envelope["max"], color=PALETTE["red"], marker="o", label="CR source maximum")
    ax.axvline(BTE_CALCULATION_CR_THRESHOLD, color=PALETTE["navy"], linestyle="--", linewidth=1.2)
    ax.set(
        title="DNN normalization source envelope by compression ratio",
        xlabel="Compression ratio",
        ylabel="Measured BTE (%)",
    )
    ax.grid(color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    rules = [*ETHANOL_ONLY_RULES, *ADDITIVE_RULES]
    x = np.arange(len(rules))
    minima = np.asarray([rule["minimum_pct"] for rule in rules], dtype=float)
    maxima = np.asarray([rule["maximum_pct"] for rule in rules], dtype=float)
    rule_colors = [PALETTE["orange"] if rule["rule_id"] == "E10_P5" else PALETTE["teal"] for rule in rules]
    ax.vlines(x, minima, maxima, color=rule_colors, linewidth=5, alpha=0.85)
    ax.scatter(x, minima, color=rule_colors, s=34, zorder=3)
    ax.scatter(x, maxima, color=rule_colors, s=34, zorder=3)
    ax.set_xticks(x, [rule["rule_id"] for rule in rules], rotation=35, ha="right")
    ax.set(
        title="User-specified final BTE ranges",
        xlabel="Fuel-blend rule",
        ylabel="Rule-guided BTE range (%)",
        ylim=(22.0, 38.0),
    )
    ax.grid(color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    fig.suptitle("Deep neural prediction + engineering-rule BTE calculation", fontsize=14, color=PALETTE["navy"], fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bte_hybrid_calculation.png", dpi=180)
    plt.close(fig)


def _plot_actual_vs_predicted(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    for axis, target in zip(axes.flat, PERFORMANCE_TARGETS):
        subset = predictions[predictions["target"] == target]
        axis.scatter(subset["actual"], subset["predicted_raw"], s=22, alpha=0.55, color=PALETTE["teal"], edgecolors="none")
        lower = min(subset["actual"].min(), subset["predicted_raw"].min())
        upper = max(subset["actual"].max(), subset["predicted_raw"].max())
        axis.plot([lower, upper], [lower, upper], color=PALETTE["orange"], linewidth=1.6)
        axis.set(title=TARGET_LABELS[target], xlabel="Actual", ylabel="Predicted")
        axis.grid(color=PALETTE["grid"], linewidth=0.55, alpha=0.7)
    fig.suptitle(
        "Internal DNN holdout performance — before BTE rules",
        fontsize=14,
        color=PALETTE["navy"],
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)


def _plot_cv_comparison(cv: pd.DataFrame) -> None:
    summary = cv.groupby(["target", "model"], as_index=False).agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"))
    models = list(summary["model"].drop_duplicates())
    targets = PERFORMANCE_TARGETS
    x = np.arange(len(targets), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(models))
    colors = [PALETTE["navy"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"]]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for offset, color, model in zip(offsets, colors, models):
        values = summary[summary["model"] == model].set_index("target").reindex(targets)
        ax.errorbar(x + offset, values["mean_r2"], yerr=values["std_r2"], fmt="o", capsize=3, label=model, color=color, markersize=6)
    ax.axhline(0, color="#7A8793", linewidth=0.8)
    ax.set_xticks(x, [TARGET_LABELS[t] for t in targets])
    ax.set_ylabel("Grouped CV R² (mean ± SD)")
    ax.set_title("Deep-network validation across unseen blend groups (128-128-64-32)")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    if len(models) > 1:
        ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "grouped_cv_model_comparison.png", dpi=180)
    plt.close(fig)


def _write_model_card(
    audit: dict,
    holdout: pd.DataFrame,
    cv: pd.DataFrame,
    property_cv: pd.DataFrame,
    bte_calculation: dict,
) -> None:
    selected_lines = []
    for row in holdout.itertuples(index=False):
        selected_lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.selected_model} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |"
        )
    property_summary = property_cv.groupby("property", as_index=False).agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"))
    property_lines = [
        f"| {row.property} | {row.mean_r2:.4f} ± {row.std_r2:.4f} |"
        for row in property_summary.itertuples(index=False)
    ]
    envelope_lines = [
        f"| {row['compression_ratio']:.2f} | {row['source_rows']} | {row['source_min_pct']:.6f}% | {row['source_max_pct']:.6f}% |"
        for row in bte_calculation["source_envelopes_by_cr"]
    ]
    text = f"""# GEB-AI Model Card

## Intended use

Screening-level deep-neural prediction and empirical-support optimization of gasoline/alcohol blends for the supplied SI-engine dataset. Outputs are decision support, not a substitute for dynamometer testing, fuel compatibility testing, safety review, or manufacturer approval.

## Data used

- Valid training records: {audit['valid_model_rows']}
- Unique valid blend groups: {audit['valid_blend_groups']}
- Quarantined records: {audit['quarantined_rows']}
- Split rule: whole fuel-composition groups are held out so the same blend cannot leak into training and test sets.

## Unseen-blend holdout results

| Target | Selected model | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
{chr(10).join(selected_lines)}

## Fuel-property estimator CV

| Property | Mean R² ± SD |
|---|---:|
{chr(10).join(property_lines)}

## Hybrid BTE calculation requested for this project

For CR >= 9.5, the app first locates the internal DNN BTE signal inside the cleaned-data minimum/maximum envelope at that compression ratio:

| Measured CR | Rows | Source minimum | Source maximum |
|---:|---:|---:|---:|
{chr(10).join(envelope_lines)}

`q = clip((DNN BTE signal - BTE minimum at CR) / (BTE maximum at CR - BTE minimum at CR), 0, 1)`

It then maps that dataset-driven score into the active blend range:

`Final BTE = Blend minimum + q * (Blend maximum - Blend minimum)`

| Blend rule | Final BTE range |
|---|---:|
| E0: G100 | 25-30% |
| E5: G95 + E5 | 27-32% |
| E10: G90 + E10 | 31-35% |
| E15: G85 + E15 | 26-31% |
| E20: G80 + E20 | 23-30% |
| E10 + P5: G85 + E10 + pentanol 5 | 32-37% |
| E10 + propanol/butanol 5 | 31-35% |
| E10 + n-methylaniline 5 | 31-37% (37% maximum) |

The prompt's invalid “95% gasoline + 27% ethanol” combination is treated as E5 (95% gasoline + 5% ethanol), with 27% as the lower BTE bound. Whenever n-methylaniline is present, a final global ceiling of 37% is enforced at every compression ratio, including CR below 9.5. Prediction, Optimization, and Simulation use the same canonical inference function.

The user-supplied ranges are engineering priors and conflict with some measured source labels. Therefore, the raw DNN holdout metrics above do not constitute independent validation of the final rule-guided BTE; controlled engine testing remains required.

## Recommendation method

The recommendation engine generates gasoline/ethanol blends containing at most the one user-selected additive and includes a zero-additive baseline. It rejects candidates beyond the validated composition-support threshold, then compares every supported blend with exact user-entered BTE, BSFC, CO, and HC values.

Each absolute prediction gap is divided by its target's unseen-blend P90 error so all four outputs participate equally despite their different units. Match utilities are averaged, and a support-distance penalty of up to 0.03 favors candidates nearer experimental compositions when matches are similar. Deterministic tie-breaking provides reproducible ranking. Confidence reflects the average and largest model-error-scaled gaps; any operating-input or target-output bound changes the label to Exploratory.

## Important limitations

- The neural training envelope is torque 1.92-8.08 Nm, RPM 1700-3300, and CR up to 10.5. Requests outside it use the nearest trained edge for neural inference and are flagged Exploratory; the requested CR remains active in the BTE engineering rule.
- Vehicle CR values that are not directly represented by a measured training level are also flagged with the nearest tested CR.
- The supplied vehicle table does not contain engine torque or RPM. The interface uses clearly labelled dataset starting points that must be changed to the intended test condition.
- Fuel-property predictions for new blends are estimated from the median property values of known blend groups and bounded to observed ranges.
- The four performance targets use separate four-hidden-layer deep neural regressors (128-128-64-32 neurons). The auxiliary fuel-property estimator remains separate and is not used as a performance target model.
- BSFC, CO, and HC are bounded to their measured target envelopes after neural inference, preventing implausible out-of-range network outputs from being rewarded by optimization.
- The separate Experimental prediction page deliberately bypasses the BTE conversion, n-Methylaniline ceiling, operating-input clamping, and all output bounds. Its direct network values may be negative or implausible outside the dataset and must not be treated as recommendations.
- Candidate search combines convex mixtures of experimentally supported blends with the explicit user-rule anchors, then rejects distant candidates. It does not establish fuel-system, material, legal, or road-use compatibility.
- Bharat Stage is retained as BS6/BS4 test metadata only; it is not a trained feature in the supplied experimental dataset.
- Recommendation and standard simulation enforce at least 75% gasoline, require gasoline to exceed ethanol and the selected additive, and never permit more than one additive. Simulation supports a fully fixed blend or an additive range with either ethanol or gasoline fixed.
- BTE, BSFC, CO, and HC are independently learned targets. The app reports a heat-balance consistency warning when needed.
"""
    (REPORT_DIR / "model_card.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate the GEB-AI model bundle.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--vehicles", type=Path, default=RAW_VEHICLE_PATH)
    args = parser.parse_args()

    _prepare_directories()
    data, quarantine, audit = load_and_clean_experimental_data(args.data)
    vehicles = extract_vehicle_database(args.vehicles)

    data.to_csv(PROCESSED_DIR / "experimental_clean.csv", index=False)
    quarantine.to_csv(PROCESSED_DIR / "quarantine_invalid_rows.csv", index=False)
    vehicles.to_csv(PROCESSED_DIR / "vehicles.csv", index=False)
    write_audit(audit, REPORT_DIR / "data_audit.json")

    bundle, reports = train_models(data)
    save_bundle(bundle, ARTIFACT_DIR / "geb_ai_model_bundle.joblib")
    reports["cv_metrics"].to_csv(REPORT_DIR / "grouped_cv_metrics.csv", index=False)
    reports["holdout_metrics"].to_csv(REPORT_DIR / "holdout_metrics.csv", index=False)
    reports["holdout_predictions"].to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)
    reports["feature_importance"].to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    reports["property_cv_metrics"].to_csv(REPORT_DIR / "fuel_property_cv_metrics.csv", index=False)

    model_metadata = {
        "bundle_version": bundle["version"],
        "model_family": bundle["model_family"],
        "deep_hidden_layers": bundle["deep_hidden_layers"],
        "holdout_random_state": bundle["holdout_random_state"],
        "training_rows": bundle["training_rows"],
        "training_blend_groups": bundle["training_blend_groups"],
        "model_features": bundle["model_features"],
        "performance_ranges": bundle["performance_ranges"],
        "selected_models": dict(
            zip(
                reports["holdout_metrics"]["target"],
                reports["holdout_metrics"]["selected_model"],
            )
        ),
        "bte_calculation": bundle["bte_calculation"],
        "support_distance_threshold": bundle["support_distance_threshold"],
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")

    rule_rows = [
        {"rule_type": "ethanol_only", **rule}
        for rule in bundle["bte_calculation"]["ethanol_only_rules"]
    ] + [
        {"rule_type": "e10_additive", **rule}
        for rule in bundle["bte_calculation"]["additive_rules"]
    ]
    pd.DataFrame(rule_rows).to_csv(REPORT_DIR / "bte_rule_ranges.csv", index=False)

    _plot_bte_calculation(data)
    _plot_actual_vs_predicted(reports["holdout_predictions"])
    _plot_cv_comparison(reports["cv_metrics"])
    _write_model_card(
        audit,
        reports["holdout_metrics"],
        reports["cv_metrics"],
        reports["property_cv_metrics"],
        bundle["bte_calculation"],
    )

    print("Training complete")
    print(reports["holdout_metrics"].to_string(index=False))
    print(f"Model bundle: {ARTIFACT_DIR / 'geb_ai_model_bundle.joblib'}")


if __name__ == "__main__":
    main()
