from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.xlsx"
RAW_VEHICLE_PATH = PROJECT_ROOT / "data" / "raw" / "Vehical comparision specific (1).docx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
BLEND_TOLERANCE_LOW = 99.99
BLEND_TOLERANCE_HIGH = 100.01

COMPOSITION_FEATURES = [
    "gasoline_pct",
    "ethanol_pct",
    "pentanol_pct",
    "propanol_pct",
    "butanol_pct",
    "n_methylaniline_pct",
]

ENGINE_FEATURES = ["torque_nm", "speed_rpm", "compression_ratio"]
MODEL_FEATURES = COMPOSITION_FEATURES + ENGINE_FEATURES

PROPERTY_TARGETS = [
    "cv_mj_kg",
    "air_fuel_ratio",
    "viscosity_m2_s",
    "ron",
    "oxygen_pct",
]

PERFORMANCE_TARGETS = ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]

TARGET_LABELS = {
    "bte_pct": "BTE (%)",
    "bsfc_g_kwh": "BSFC (g/kWh)",
    "co_vol_pct": "CO (vol.%)",
    "hc_ppm": "HC (ppm)",
}

PROPERTY_LABELS = {
    "cv_mj_kg": "Calorific value (MJ/kg)",
    "air_fuel_ratio": "Air-fuel ratio",
    "viscosity_m2_s": "Viscosity (m²/s)",
    "ron": "Research octane number",
    "oxygen_pct": "Oxygen (%)",
}

# Hybrid BTE calculation: source envelopes are learned from the cleaned data,
# while the final range is selected from the user's engineering blend rules.
BTE_CALCULATION_CR_THRESHOLD = 9.5
BTE_BASE_RANGE_MIN = 26.0
BTE_BASE_RANGE_MAX = 36.0
