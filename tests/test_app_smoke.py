from __future__ import annotations

import runpy
import sys
import types
import unittest
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
VEHICLE_PATH = PROJECT_ROOT / "data" / "processed" / "vehicles.csv"


class _Context:
    def __init__(self, owner: "FakeStreamlit") -> None:
        self.owner = owner

    def __enter__(self) -> "_Context":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)


class FakeStreamlit(types.ModuleType):
    """Small Streamlit surface used to execute every top-level app branch."""

    def __init__(self, page: str, run_actions: bool = True) -> None:
        super().__init__("streamlit")
        vehicles = pd.read_csv(VEHICLE_PATH)
        company = sorted(vehicles["company"].dropna().unique())[0]
        vehicle = vehicles.loc[vehicles["company"] == company].iloc[0]
        self.page = page
        self.run_actions = run_actions
        self.errors: list[str] = []
        self.metrics: list[tuple[str, str]] = []
        self.selectbox_options: dict[str, list[Any]] = {}
        self.number_input_labels: list[str] = []
        self.session_state: dict[str, Any] = {
            "active_vehicle_id": str(vehicle["vehicle_id"]),
            "compression_ratio": 9.49,
            "torque_nm": 5.51,
            "speed_rpm": 2500.0,
            "gasoline_pct": 85.0,
            "ethanol_pct": 10.0,
            "pentanol_pct": 0.0,
            "propanol_pct": 0.0,
            "butanol_pct": 0.0,
            "n_methylaniline_pct": 5.0,
        }
        self.sidebar = _Context(self)

    def cache_resource(self, function=None, **_kwargs):
        return function if function is not None else (lambda wrapped: wrapped)

    def columns(self, specification, **_kwargs):
        count = specification if isinstance(specification, int) else len(specification)
        return [_Context(self) for _ in range(count)]

    def radio(self, _label, _options, **_kwargs):
        return self.page

    def selectbox(self, label, options, index=0, key=None, **_kwargs):
        values = list(options)
        self.selectbox_options[str(label)] = values
        if key is not None and key in self.session_state:
            value = self.session_state[key]
        elif label in {"Allowed additive", "Simulated additive", "Selected additive"}:
            value = "n_methylaniline_pct"
        else:
            value = values[index]
        if key is not None:
            self.session_state[key] = value
        return value

    def number_input(self, _label, *args, key=None, value=None, min_value=None, **_kwargs):
        self.number_input_labels.append(str(_label))
        if key is not None and key in self.session_state:
            return self.session_state[key]
        if value is None:
            value = min_value if min_value is not None else (args[0] if args else 0.0)
        if key is not None:
            self.session_state[key] = value
        return value

    def slider(self, _label, *args, key=None, value=None, **_kwargs):
        if key is not None and key in self.session_state:
            return self.session_state[key]
        if value is None:
            value = args[2] if len(args) >= 3 else args[0]
        if key is not None:
            self.session_state[key] = value
        return value

    def checkbox(self, _label, value=False, key=None, **_kwargs):
        if key is not None and key in self.session_state:
            return self.session_state[key]
        if key is not None:
            self.session_state[key] = value
        return value

    def toggle(self, _label, value=False, key=None, **_kwargs):
        return self.checkbox(_label, value=value, key=key)

    def button(self, label, **_kwargs):
        return self.run_actions and label in {
            "Predict performance & emissions",
            "Run raw experimental prediction",
            "Recommend best blend",
            "Run blend simulation",
        }

    def spinner(self, *_args, **_kwargs):
        return _Context(self)

    def expander(self, *_args, **_kwargs):
        return _Context(self)

    def metric(self, label, value, **_kwargs):
        self.metrics.append((str(label), str(value)))

    def error(self, message, **_kwargs):
        self.errors.append(str(message))

    def stop(self):
        raise RuntimeError("Streamlit stop() was called during the smoke test")

    def set_page_config(self, **_kwargs):
        return None

    def download_button(self, *_args, **_kwargs):
        return False

    def __getattr__(self, name: str):
        if name in {
            "caption",
            "code",
            "dataframe",
            "divider",
            "header",
            "image",
            "info",
            "json",
            "markdown",
            "pyplot",
            "subheader",
            "success",
            "warning",
        }:
            return lambda *_args, **_kwargs: None
        raise AttributeError(name)


def run_app(fake: FakeStreamlit) -> None:
    sys.modules["streamlit"] = fake
    runpy.run_path(str(APP_PATH), run_name=f"__smoke_{fake.page.replace(' ', '_')}")


class StreamlitSmokeTests(unittest.TestCase):
    def test_motorsport_theme_is_loaded_and_old_banner_is_removed(self):
        app_source = APP_PATH.read_text(encoding="utf-8")
        theme_source = (PROJECT_ROOT / "assets" / "motorsport.css").read_text(
            encoding="utf-8"
        )
        self.assertIn("assets\" / \"motorsport.css", app_source)
        self.assertNotIn("Deep-learning BTE calculation:", app_source)
        self.assertNotIn("The four-output performance engine", app_source)
        self.assertIn(".race-hero", theme_source)
        self.assertIn("@keyframes gear-spin", theme_source)
        self.assertIn("prefers-reduced-motion", theme_source)

    def test_optimization_uses_exact_inputs_and_two_bharat_stage_choices(self):
        app_source = APP_PATH.read_text(encoding="utf-8")
        self.assertNotIn("Objective priorities", app_source)
        self.assertNotIn("Priority profile", app_source)
        self.assertNotIn("Maximize BTE", app_source)
        self.assertNotIn("Minimize BSFC", app_source)

        fake = FakeStreamlit("Optimization")
        run_app(fake)
        self.assertEqual(fake.selectbox_options["Bharat Stage"], ["BS6", "BS4"])
        for label in ["BTE (%)", "BSFC (g/kWh)", "CO (vol.%)", "HC (ppm)"]:
            self.assertIn(label, fake.number_input_labels)
        result = fake.session_state["optimization_result"]
        self.assertIn(result["bharat_stage"], {"BS6", "BS4"})
        self.assertTrue(bool(result["best"]["target_matching_active"]))
        self.assertEqual(
            set(result["target_values"]),
            {"bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"},
        )

    def test_changed_exact_value_or_bharat_stage_clears_recommendation(self):
        for changed_key, changed_value in [
            ("opt_target_bte_pct", 31.75),
            ("bharat_stage", None),
        ]:
            with self.subTest(changed_key=changed_key):
                fake = FakeStreamlit("Optimization", run_actions=True)
                run_app(fake)
                self.assertIn("optimization_result", fake.session_state)
                fake.run_actions = False
                vehicle_id = fake.session_state["active_vehicle_id"]
                if changed_key == "bharat_stage":
                    current = fake.session_state[f"bharat_stage_{vehicle_id}"]
                    changed_value = "BS6" if current == "BS4" else "BS4"
                fake.session_state[f"{changed_key}_{vehicle_id}"] = changed_value
                run_app(fake)
                self.assertNotIn("optimization_result", fake.session_state)

    def test_every_page_executes_without_ui_errors(self):
        expected_result = {
            "Prediction": "prediction_result",
            "Experimental prediction": "experimental_prediction_result",
            "Optimization": "optimization_result",
            "Blend simulation": "simulation_result",
        }
        for page in [
            "Prediction",
            "Experimental prediction",
            "Optimization",
            "Blend simulation",
            "Model analytics",
            "Data quality",
        ]:
            with self.subTest(page=page):
                fake = FakeStreamlit(page)
                run_app(fake)
                self.assertEqual(fake.errors, [])
                if page in expected_result:
                    self.assertIn(expected_result[page], fake.session_state)
                self.assertFalse(
                    any("Raw ML BTE" in label for label, _value in fake.metrics)
                )

    def test_experimental_page_displays_direct_raw_model_output(self):
        fake = FakeStreamlit("Experimental prediction")
        run_app(fake)
        result = fake.session_state["experimental_prediction_result"]
        self.assertFalse(result["post_processing_applied"])
        self.assertTrue(any(label == "Raw BTE (%)" for label, _value in fake.metrics))

    def test_nma_outputs_and_simulation_are_capped(self):
        prediction_ui = FakeStreamlit("Prediction")
        run_app(prediction_ui)
        predicted = prediction_ui.session_state["prediction_result"]
        self.assertLessEqual(predicted["predictions"]["bte_pct"], 37.0)

        simulation_ui = FakeStreamlit("Blend simulation")
        run_app(simulation_ui)
        simulation = simulation_ui.session_state["simulation_result"]["data"]
        nma_rows = simulation[simulation["n_methylaniline_pct"] > 0.0]
        self.assertGreater(len(nma_rows), 0)
        self.assertTrue((nma_rows["bte_pct"] <= 37.0).all())

    def test_simulator_user_defined_range_modes_are_wired(self):
        for setup, fixed_column, expected_value in [
            ("Additive range — fix ethanol", "ethanol_pct", 10.0),
            ("Additive range — fix gasoline", "gasoline_pct", 85.0),
        ]:
            with self.subTest(setup=setup):
                fake = FakeStreamlit("Blend simulation")
                vehicle_id = fake.session_state["active_vehicle_id"]
                fake.session_state[f"sim_setup_{vehicle_id}"] = setup
                run_app(fake)
                self.assertEqual(fake.errors, [])
                simulation = fake.session_state["simulation_result"]["data"]
                self.assertGreater(len(simulation), 1)
                self.assertEqual(simulation[fixed_column].nunique(), 1)
                self.assertAlmostEqual(float(simulation[fixed_column].iloc[0]), expected_value)

    def test_changed_prediction_inputs_clear_stale_result(self):
        fake = FakeStreamlit("Prediction", run_actions=True)
        run_app(fake)
        self.assertIn("prediction_result", fake.session_state)
        fake.run_actions = False
        fake.session_state["gasoline_pct"] = 80.0
        fake.session_state["n_methylaniline_pct"] = 10.0
        run_app(fake)
        self.assertNotIn("prediction_result", fake.session_state)

    def test_changed_optimizer_and_simulator_inputs_clear_stale_results(self):
        for page, prefix, result_key in [
            ("Optimization", "opt", "optimization_result"),
            ("Blend simulation", "sim", "simulation_result"),
        ]:
            with self.subTest(page=page):
                fake = FakeStreamlit(page, run_actions=True)
                run_app(fake)
                self.assertIn(result_key, fake.session_state)
                fake.run_actions = False
                vehicle_id = fake.session_state["active_vehicle_id"]
                fake.session_state[f"{prefix}_torque_nm_{vehicle_id}"] = 7.25
                run_app(fake)
                self.assertNotIn(result_key, fake.session_state)


if __name__ == "__main__":
    unittest.main()
