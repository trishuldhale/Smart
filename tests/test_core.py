from __future__ import annotations

import itertools
import unittest

import numpy as np
import pandas as pd

from src.bte_calculation import calculate_bte, source_envelope_at_cr
from src.config import ARTIFACT_DIR, COMPOSITION_FEATURES, MODEL_FEATURES, PROCESSED_DIR
from src.optimization import (
    ADDITIVE_LABELS,
    evaluate_candidates,
    generate_candidates,
    generate_single_additive_candidates,
    optimize_single_additive_blend,
)
from src.prediction import (
    InputValidationError,
    load_model_bundle,
    predict_performance,
    predict_raw_performance,
    validate_blend,
)
from src.simulation import simulate_blend_grid, simulate_fixed_blend


def blend(gasoline, ethanol, pentanol=0, propanol=0, butanol=0, nma=0):
    return {
        "gasoline_pct": gasoline,
        "ethanol_pct": ethanol,
        "pentanol_pct": pentanol,
        "propanol_pct": propanol,
        "butanol_pct": butanol,
        "n_methylaniline_pct": nma,
    }


class CoreSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle(ARTIFACT_DIR / "geb_ai_model_bundle.joblib")

    def test_valid_blend(self):
        validate_blend(blend(90, 10))

    def test_invalid_blend_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_blend(blend(90, 20))

    def test_e10_prediction_uses_31_to_35_range(self):
        result = predict_performance(
            self.bundle,
            blend(90, 10),
            torque_nm=5.5,
            speed_rpm=2500,
            compression_ratio=9.7,
        )
        self.assertGreaterEqual(result["predictions"]["bte_pct"], 31.0)
        self.assertLessEqual(result["predictions"]["bte_pct"], 35.0)
        self.assertEqual(result["bte_calculation"]["rule_id"], "E10")
        self.assertTrue(result["bte_calculation_applied"])

    def test_base_mapping_uses_26_to_36_range(self):
        calculation = self.bundle["bte_calculation"]
        source_min, source_max = source_envelope_at_cr(calculation, 9.7)
        generic_blend = blend(70, 10, pentanol=20)
        outputs = [
            calculate_bte(value, 9.7, generic_blend, calculation)["final_bte_pct"]
            for value in [source_min, (source_min + source_max) / 2.0, source_max]
        ]
        np.testing.assert_allclose(outputs, [26.0, 31.0, 36.0], atol=1e-12)

    def test_all_ethanol_anchor_ranges(self):
        calculation = self.bundle["bte_calculation"]
        source_min, source_max = source_envelope_at_cr(calculation, 9.7)
        raw_midpoint = (source_min + source_max) / 2.0
        cases = [
            (blend(100, 0), "E0", 25.0, 30.0),
            (blend(95, 5), "E5", 27.0, 32.0),
            (blend(90, 10), "E10", 31.0, 35.0),
            (blend(85, 15), "E15", 26.0, 31.0),
            (blend(80, 20), "E20", 23.0, 30.0),
        ]
        for composition, rule_id, lower, upper in cases:
            with self.subTest(rule_id=rule_id):
                trace = calculate_bte(raw_midpoint, 9.7, composition, calculation)
                self.assertEqual(trace["rule_id"], rule_id)
                self.assertEqual(trace["range_min_pct"], lower)
                self.assertEqual(trace["range_max_pct"], upper)
                self.assertAlmostEqual(trace["final_bte_pct"], (lower + upper) / 2.0)

    def test_pentanol_has_requested_32_to_37_range(self):
        calculation = self.bundle["bte_calculation"]
        source_min, source_max = source_envelope_at_cr(calculation, 9.7)
        raw_midpoint = (source_min + source_max) / 2.0
        pentanol = calculate_bte(raw_midpoint, 9.7, blend(85, 10, pentanol=5), calculation)
        n_methylaniline = calculate_bte(
            raw_midpoint, 9.7, blend(85, 10, nma=5), calculation
        )
        propanol = calculate_bte(raw_midpoint, 9.7, blend(85, 10, propanol=5), calculation)
        butanol = calculate_bte(raw_midpoint, 9.7, blend(85, 10, butanol=5), calculation)
        self.assertEqual((pentanol["range_min_pct"], pentanol["range_max_pct"]), (32.0, 37.0))
        self.assertEqual(
            (n_methylaniline["range_min_pct"], n_methylaniline["range_max_pct"]),
            (31.0, 37.0),
        )
        self.assertEqual((propanol["range_min_pct"], propanol["range_max_pct"]), (31.0, 35.0))
        self.assertEqual((butanol["range_min_pct"], butanol["range_max_pct"]), (31.0, 35.0))
        self.assertGreater(pentanol["final_bte_pct"], butanol["final_bte_pct"])

    def test_n_methylaniline_bte_never_exceeds_37_at_high_cr(self):
        calculation = self.bundle["bte_calculation"]
        _, source_max = source_envelope_at_cr(calculation, 9.7)
        trace = calculate_bte(source_max + 100.0, 9.7, blend(85, 10, nma=5), calculation)
        self.assertEqual(trace["rule_id"], "E10_NMA5")
        self.assertEqual(trace["range_max_pct"], 37.0)
        self.assertEqual(trace["final_bte_pct"], 37.0)
        metadata_rule = next(
            rule
            for rule in calculation["additive_rules"]
            if rule["additive"] == "n_methylaniline_pct"
        )
        self.assertEqual(metadata_rule["maximum_pct"], 37.0)

    def test_n_methylaniline_global_cap_applies_at_every_compression_ratio(self):
        calculation = self.bundle["bte_calculation"]
        for compression_ratio in [4.0, 4.67, 6.5, 9.49, 9.5, 9.7, 10.5, 11.0, 15.0]:
            with self.subTest(compression_ratio=compression_ratio):
                trace = calculate_bte(
                    50.0,
                    compression_ratio,
                    blend(75, 15, nma=10),
                    calculation,
                )
                self.assertLessEqual(trace["final_bte_pct"], 37.0)
                self.assertTrue(trace["n_methylaniline_cap_active"])
                self.assertEqual(trace["n_methylaniline_cap_pct"], 37.0)
        below_threshold = calculate_bte(50.0, 6.5, blend(75, 15, nma=10), calculation)
        self.assertTrue(below_threshold["n_methylaniline_cap_applied"])
        self.assertEqual(below_threshold["rule_id"], "nma_global_cap_37")

    def test_nma_prediction_interval_is_also_capped_below_threshold(self):
        result = predict_performance(
            self.bundle,
            blend(75, 15, nma=10),
            torque_nm=8.08,
            speed_rpm=2500.0,
            compression_ratio=9.49,
        )
        self.assertLessEqual(result["predictions"]["bte_pct"], 37.0)
        self.assertLessEqual(
            result["prediction_intervals_approx_90pct"]["bte_pct"][1],
            37.0,
        )
        self.assertTrue(result["n_methylaniline_cap_active"])

    def test_rules_hold_from_cr_9_5_to_11(self):
        for compression_ratio in [9.5, 10.0, 10.5, 11.0]:
            with self.subTest(compression_ratio=compression_ratio, rule="E10"):
                e10 = predict_performance(
                    self.bundle,
                    blend(90, 10),
                    torque_nm=5.5,
                    speed_rpm=2500,
                    compression_ratio=compression_ratio,
                )
                self.assertTrue(31.0 <= e10["predictions"]["bte_pct"] <= 35.0)
            with self.subTest(compression_ratio=compression_ratio, rule="E10_P5"):
                pentanol = predict_performance(
                    self.bundle,
                    blend(85, 10, pentanol=5),
                    torque_nm=5.5,
                    speed_rpm=2500,
                    compression_ratio=compression_ratio,
                )
                self.assertTrue(32.0 <= pentanol["predictions"]["bte_pct"] <= 37.0)

    def test_non_bte_outputs_remain_raw_model_outputs(self):
        result = predict_performance(
            self.bundle,
            blend(85, 10, pentanol=5),
            torque_nm=5.5,
            speed_rpm=2500,
            compression_ratio=9.7,
        )
        for target in ["bsfc_g_kwh", "co_vol_pct", "hc_ppm"]:
            self.assertAlmostEqual(
                result["predictions"][target],
                max(0.0, result["raw_predictions"][target]),
                places=12,
            )

    def test_prediction_optimizer_and_simulator_use_identical_inference(self):
        composition = blend(85, 10, nma=5)
        operating_point = {
            "torque_nm": 5.51,
            "speed_rpm": 2500.0,
            "compression_ratio": 9.7,
        }
        prediction = predict_performance(self.bundle, composition, **operating_point)
        evaluated = evaluate_candidates(
            self.bundle,
            pd.DataFrame([composition], columns=COMPOSITION_FEATURES),
            **operating_point,
        ).iloc[0]
        simulated = simulate_blend_grid(
            self.bundle,
            selected_additive="n_methylaniline_pct",
            min_additive_pct=5.0,
            max_additive_pct=5.0,
            min_gasoline_pct=75.0,
            grid_step_pct=1.0,
            fixed_ethanol_pct=10.0,
            **operating_point,
        ).iloc[0]
        for target in ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]:
            with self.subTest(target=target):
                self.assertAlmostEqual(
                    prediction["predictions"][target],
                    evaluated[target],
                    places=12,
                )
                self.assertAlmostEqual(evaluated[target], simulated[target], places=12)

    def test_raw_experimental_prediction_is_direct_and_unbounded(self):
        composition = blend(85, 10, nma=5)
        operating_point = {
            "torque_nm": 30.0,
            "speed_rpm": 12000.0,
            "compression_ratio": 15.0,
        }
        result = predict_raw_performance(
            self.bundle,
            composition,
            **operating_point,
        )
        model_input = pd.DataFrame(
            [{**composition, **operating_point}],
            columns=MODEL_FEATURES,
        )
        for target, model in self.bundle["performance_models"].items():
            with self.subTest(target=target):
                expected = float(model.predict(model_input)[0])
                self.assertAlmostEqual(result["predictions"][target], expected, places=12)
        self.assertFalse(result["post_processing_applied"])
        self.assertIn("torque_nm", result["outside_training_ranges"])
        self.assertIn("speed_rpm", result["outside_training_ranges"])
        self.assertIn("compression_ratio", result["outside_training_ranges"])
        self.assertGreater(result["predictions"]["bte_pct"], 37.0)
        self.assertLess(result["predictions"]["bsfc_g_kwh"], 0.0)

    def test_fully_fixed_user_simulation_matches_standard_prediction(self):
        composition = blend(85, 10, pentanol=5)
        operating_point = {
            "torque_nm": 5.51,
            "speed_rpm": 2500.0,
            "compression_ratio": 9.7,
        }
        fixed = simulate_fixed_blend(
            self.bundle,
            selected_additive="pentanol_pct",
            gasoline_pct=85.0,
            ethanol_pct=10.0,
            additive_pct=5.0,
            **operating_point,
        )
        self.assertEqual(len(fixed), 1)
        np.testing.assert_allclose(
            fixed.iloc[0][COMPOSITION_FEATURES].to_numpy(dtype=float),
            [composition[feature] for feature in COMPOSITION_FEATURES],
        )
        standard = predict_performance(self.bundle, composition, **operating_point)
        for target in ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]:
            self.assertAlmostEqual(
                float(fixed.iloc[0][target]),
                standard["predictions"][target],
                places=12,
            )
        with self.assertRaisesRegex(ValueError, "totals"):
            simulate_fixed_blend(
                self.bundle,
                selected_additive="pentanol_pct",
                gasoline_pct=85.0,
                ethanol_pct=10.0,
                additive_pct=4.0,
                **operating_point,
            )

    def test_bundle_contains_four_hidden_layer_deep_networks(self):
        self.assertEqual(self.bundle["version"], "2.1.0")
        self.assertEqual(self.bundle["model_family"], "Deep Neural Network")
        self.assertEqual(self.bundle["deep_hidden_layers"], [128, 128, 64, 32])
        for target, model in self.bundle["performance_models"].items():
            with self.subTest(target=target):
                network = model.regressor_.named_steps["network"]
                self.assertEqual(network.hidden_layer_sizes, (128, 128, 64, 32))

    def test_unsupported_operating_inputs_are_bounded_without_zero_outputs(self):
        result = predict_performance(
            self.bundle,
            blend(85, 10, pentanol=5),
            torque_nm=30.0,
            speed_rpm=12000.0,
            compression_ratio=15.0,
        )
        self.assertTrue(result["operating_point_bounded"])
        for feature, expected in {
            "torque_nm": 8.08,
            "speed_rpm": 3300.0,
            "compression_ratio": 10.5,
        }.items():
            self.assertAlmostEqual(result["model_operating_inputs"][feature], expected)
        for target in ["bsfc_g_kwh", "co_vol_pct", "hc_ppm"]:
            limits = self.bundle["performance_ranges"][target]
            self.assertGreaterEqual(result["predictions"][target], limits["min"])
            self.assertLessEqual(result["predictions"][target], limits["max"])

    def test_deep_network_unseen_blend_accuracy(self):
        metrics = pd.read_csv(PROCESSED_DIR.parent.parent / "reports" / "holdout_metrics.csv")
        minimum_r2 = {
            "bte_pct": 0.99,
            "bsfc_g_kwh": 0.97,
            "co_vol_pct": 0.94,
            "hc_ppm": 0.97,
        }
        for target, threshold in minimum_r2.items():
            with self.subTest(target=target):
                row = metrics.loc[metrics["target"] == target].iloc[0]
                self.assertIn("Deep Neural Network", row["selected_model"])
                self.assertGreaterEqual(row["r2"], threshold)

    def test_bundle_source_envelopes_match_cleaned_dataset(self):
        data = pd.read_csv(PROCESSED_DIR / "experimental_clean.csv")
        calculation = self.bundle["bte_calculation"]
        for envelope in calculation["source_envelopes_by_cr"]:
            subset = data.loc[
                data["compression_ratio"] == envelope["compression_ratio"],
                "bte_pct",
            ]
            self.assertAlmostEqual(envelope["source_min_pct"], subset.min(), places=10)
            self.assertAlmostEqual(envelope["source_max_pct"], subset.max(), places=10)

    def test_optimizer_candidates_sum_to_100(self):
        candidates = generate_candidates(self.bundle, n_candidates=250)
        totals = candidates[COMPOSITION_FEATURES].sum(axis=1)
        self.assertTrue(((totals - 100.0).abs() < 1e-6).all())
        self.assertTrue((candidates["gasoline_pct"] >= 75.0).all())
        self.assertTrue((candidates["gasoline_pct"] > candidates["ethanol_pct"]).all())
        for additive in ADDITIVE_LABELS:
            self.assertTrue((candidates["gasoline_pct"] > candidates[additive]).all())
        e10_p5 = (
            candidates[COMPOSITION_FEATURES]
            .sub(pd.Series(blend(85, 10, pentanol=5)))
            .abs()
            .max(axis=1)
            < 1e-9
        )
        self.assertTrue(e10_p5.any())

    def test_single_additive_candidate_grid_never_mixes_additives(self):
        for selected_additive in ADDITIVE_LABELS:
            with self.subTest(selected_additive=selected_additive):
                candidates = generate_single_additive_candidates(
                    selected_additive,
                    min_gasoline_pct=75.0,
                    max_ethanol_pct=20.0,
                    max_additive_pct=5.0,
                    grid_step_pct=1.0,
                )
                totals = candidates[COMPOSITION_FEATURES].sum(axis=1)
                self.assertTrue(((totals - 100.0).abs() < 1e-9).all())
                self.assertTrue((candidates[selected_additive] > 0.0).all())
                self.assertTrue((candidates["gasoline_pct"] >= 75.0).all())
                self.assertTrue((candidates["gasoline_pct"] > candidates["ethanol_pct"]).all())
                self.assertTrue((candidates["gasoline_pct"] > candidates[selected_additive]).all())
                for other_additive in ADDITIVE_LABELS:
                    if other_additive != selected_additive:
                        self.assertTrue((candidates[other_additive] == 0.0).all())

    def test_gasoline_ethanol_only_grid_has_no_additives(self):
        candidates = generate_single_additive_candidates(
            None,
            min_gasoline_pct=80.0,
            max_ethanol_pct=20.0,
            grid_step_pct=1.0,
        )
        self.assertTrue((candidates[list(ADDITIVE_LABELS)] == 0.0).all().all())
        self.assertTrue(
            ((candidates[COMPOSITION_FEATURES].sum(axis=1) - 100.0).abs() < 1e-9).all()
        )

    def test_vehicle_recommendation_uses_only_selected_additive(self):
        requested_profile = predict_performance(
            self.bundle,
            blend(85, 10, pentanol=5),
            torque_nm=5.51,
            speed_rpm=2500.0,
            compression_ratio=10.0,
        )["predictions"]
        best, ranked, supported = optimize_single_additive_blend(
            self.bundle,
            selected_additive="pentanol_pct",
            target_values=requested_profile,
            torque_nm=5.51,
            speed_rpm=2500.0,
            compression_ratio=10.0,
            min_gasoline_pct=75.0,
            max_ethanol_pct=20.0,
            max_additive_pct=5.0,
            grid_step_pct=1.0,
            fixed_ethanol_pct=10.0,
        )
        self.assertGreater(len(ranked), 0)
        self.assertGreater(len(supported), 0)
        self.assertGreater(best["pentanol_pct"], 0.0)
        for other_additive in ["propanol_pct", "butanol_pct", "n_methylaniline_pct"]:
            self.assertEqual(best[other_additive], 0.0)
        self.assertAlmostEqual(best[COMPOSITION_FEATURES].sum(), 100.0, places=9)
        self.assertTrue(best["baseline_available"])
        self.assertTrue((supported["pentanol_pct"] == 0.0).any())
        self.assertIn(best["recommendation_confidence"], {"High", "Moderate", "Low"})
        self.assertTrue(0.0 <= best["recommendation_score_pct"] <= 100.0)
        for target in ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]:
            self.assertTrue(
                (ranked[f"{target}_match_utility"].between(0.0, 1.0)).all()
            )

    def test_recommendation_is_deterministic_and_extreme_case_is_exploratory(self):
        kwargs = {
            "selected_additive": "n_methylaniline_pct",
            "torque_nm": 30.0,
            "speed_rpm": 12000.0,
            "compression_ratio": 15.0,
            "min_gasoline_pct": 75.0,
            "max_ethanol_pct": 20.0,
            "max_additive_pct": 5.0,
            "grid_step_pct": 0.5,
            "fixed_ethanol_pct": 10.0,
            "target_values": {
                "bte_pct": 34.0,
                "bsfc_g_kwh": 400.0,
                "co_vol_pct": 2.0,
                "hc_ppm": 250.0,
            },
        }
        first, _, _ = optimize_single_additive_blend(self.bundle, **kwargs)
        second, _, _ = optimize_single_additive_blend(self.bundle, **kwargs)
        for feature in COMPOSITION_FEATURES:
            self.assertAlmostEqual(first[feature], second[feature], places=12)
        self.assertAlmostEqual(first["weighted_score"], second["weighted_score"], places=12)
        self.assertEqual(first["recommendation_confidence"], "Exploratory")
        self.assertTrue(first["operating_point_bounded"])
        self.assertGreaterEqual(first["bsfc_g_kwh"], self.bundle["performance_ranges"]["bsfc_g_kwh"]["min"])
        self.assertGreaterEqual(first["hc_ppm"], self.bundle["performance_ranges"]["hc_ppm"]["min"])
        self.assertLessEqual(first["bte_pct"], 37.0)

    def test_exact_performance_profile_is_required(self):
        with self.assertRaisesRegex(ValueError, "exact values"):
            optimize_single_additive_blend(
                self.bundle,
                selected_additive="pentanol_pct",
                torque_nm=5.51,
                speed_rpm=2500.0,
                compression_ratio=9.7,
                fixed_ethanol_pct=10.0,
            )

    def test_exact_performance_profile_selects_its_matching_blend(self):
        operating_point = {
            "torque_nm": 5.51,
            "speed_rpm": 2500.0,
            "compression_ratio": 10.0,
        }
        candidates = generate_single_additive_candidates(
            "pentanol_pct",
            min_gasoline_pct=75.0,
            max_additive_pct=5.0,
            grid_step_pct=1.0,
            fixed_ethanol_pct=10.0,
            min_additive_pct=0.0,
        )
        evaluated = evaluate_candidates(self.bundle, candidates, **operating_point)
        selected_doses = []
        for row_index in [0, len(evaluated) - 1]:
            expected = evaluated.iloc[row_index]
            exact_values = {
                target: float(expected[target])
                for target in ["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]
            }
            best, ranked, supported = optimize_single_additive_blend(
                self.bundle,
                selected_additive="pentanol_pct",
                target_values=exact_values,
                min_gasoline_pct=75.0,
                max_additive_pct=5.0,
                grid_step_pct=1.0,
                fixed_ethanol_pct=10.0,
                **operating_point,
            )
            selected_doses.append(float(best["pentanol_pct"]))
            self.assertAlmostEqual(
                best["pentanol_pct"], expected["pentanol_pct"], places=12
            )
            self.assertAlmostEqual(best["target_match_score_pct"], 100.0, places=10)
            self.assertEqual(len(ranked), len(supported))
            self.assertTrue(bool(best["target_matching_active"]))
            for target, entered in exact_values.items():
                self.assertAlmostEqual(best[f"{target}_entered_value"], entered)
                self.assertAlmostEqual(best[f"{target}_absolute_gap"], 0.0, places=10)
        self.assertNotEqual(selected_doses[0], selected_doses[1])

    def test_invalid_exact_performance_profiles_are_rejected(self):
        common = {
            "selected_additive": "pentanol_pct",
            "torque_nm": 5.51,
            "speed_rpm": 2500.0,
            "compression_ratio": 9.7,
            "fixed_ethanol_pct": 10.0,
        }
        valid = {
            "bte_pct": 31.0,
            "bsfc_g_kwh": 400.0,
            "co_vol_pct": 2.0,
            "hc_ppm": 250.0,
        }
        invalid_profiles = [
            {key: value for key, value in valid.items() if key != "hc_ppm"},
            {**valid, "unknown_target": 1.0},
            {**valid, "bte_pct": float("nan")},
            {**valid, "co_vol_pct": -0.1},
        ]
        for invalid in invalid_profiles:
            with self.subTest(target_values=invalid):
                with self.assertRaises(ValueError):
                    optimize_single_additive_blend(
                        self.bundle,
                        target_values=invalid,
                        **common,
                    )

    def test_fixed_ethanol_varies_only_selected_additive(self):
        candidates = generate_single_additive_candidates(
            "pentanol_pct",
            min_gasoline_pct=75.0,
            min_additive_pct=0.0,
            max_additive_pct=5.0,
            grid_step_pct=1.0,
            fixed_ethanol_pct=10.0,
        )
        self.assertEqual(len(candidates), 6)
        self.assertEqual(candidates["ethanol_pct"].nunique(), 1)
        self.assertEqual(candidates["ethanol_pct"].iloc[0], 10.0)
        np.testing.assert_allclose(
            candidates["gasoline_pct"], 90.0 - candidates["pentanol_pct"]
        )

    def test_fixed_gasoline_varies_only_selected_additive(self):
        candidates = generate_single_additive_candidates(
            "butanol_pct",
            min_gasoline_pct=75.0,
            min_additive_pct=0.0,
            max_additive_pct=5.0,
            grid_step_pct=1.0,
            fixed_gasoline_pct=85.0,
        )
        self.assertEqual(len(candidates), 6)
        self.assertEqual(candidates["gasoline_pct"].nunique(), 1)
        self.assertEqual(candidates["gasoline_pct"].iloc[0], 85.0)
        np.testing.assert_allclose(
            candidates["ethanol_pct"], 15.0 - candidates["butanol_pct"]
        )

    def test_fixed_component_and_minimum_gasoline_validation(self):
        with self.assertRaisesRegex(ValueError, "either gasoline or ethanol"):
            generate_single_additive_candidates(
                "pentanol_pct",
                fixed_ethanol_pct=10.0,
                fixed_gasoline_pct=85.0,
            )
        with self.assertRaisesRegex(ValueError, "between 75 and 100"):
            generate_single_additive_candidates(
                "pentanol_pct",
                min_gasoline_pct=74.5,
            )

    def test_simulation_grid_predicts_every_valid_blend(self):
        results = simulate_blend_grid(
            self.bundle,
            selected_additive="butanol_pct",
            torque_nm=5.51,
            speed_rpm=2500.0,
            compression_ratio=10.0,
            min_ethanol_pct=0.0,
            max_ethanol_pct=10.0,
            min_additive_pct=0.0,
            max_additive_pct=5.0,
            min_gasoline_pct=75.0,
            grid_step_pct=1.0,
            fixed_ethanol_pct=None,
        )
        self.assertEqual(len(results), 66)
        self.assertTrue((results[["pentanol_pct", "propanol_pct", "n_methylaniline_pct"]] == 0.0).all().all())
        self.assertTrue(
            ((results[COMPOSITION_FEATURES].sum(axis=1) - 100.0).abs() < 1e-9).all()
        )
        self.assertTrue((results[["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]] >= 0.0).all().all())
        self.assertIn("within_validated_support", results)

    def test_simulation_fixed_ethanol_is_additive_only_sweep(self):
        results = simulate_blend_grid(
            self.bundle,
            selected_additive="n_methylaniline_pct",
            torque_nm=5.51,
            speed_rpm=2500.0,
            compression_ratio=10.0,
            min_additive_pct=0.0,
            max_additive_pct=5.0,
            min_gasoline_pct=75.0,
            grid_step_pct=1.0,
            fixed_ethanol_pct=10.0,
        )
        self.assertEqual(len(results), 6)
        self.assertEqual(results["ethanol_pct"].nunique(), 1)
        self.assertTrue((results["bte_pct"] <= 37.0 + 1e-9).all())
        self.assertTrue((results["gasoline_pct"] >= 75.0).all())

    def test_simulation_all_additives_and_constraint_modes(self):
        modes = [
            {"fixed_ethanol_pct": 10.0},
            {"fixed_gasoline_pct": 85.0},
            {},
        ]
        for selected_additive, mode in itertools.product(
            [None, *ADDITIVE_LABELS], modes
        ):
            with self.subTest(selected_additive=selected_additive, mode=mode):
                results = simulate_blend_grid(
                    self.bundle,
                    selected_additive=selected_additive,
                    torque_nm=5.51,
                    speed_rpm=2500.0,
                    compression_ratio=9.49,
                    min_ethanol_pct=0.0,
                    max_ethanol_pct=10.0,
                    min_additive_pct=0.0,
                    max_additive_pct=5.0,
                    min_gasoline_pct=75.0,
                    grid_step_pct=2.5,
                    **mode,
                )
                self.assertGreater(len(results), 0)
                np.testing.assert_allclose(
                    results[COMPOSITION_FEATURES].sum(axis=1),
                    100.0,
                    atol=1e-9,
                )
                self.assertTrue((results["gasoline_pct"] >= 75.0).all())
                self.assertTrue(
                    (results["gasoline_pct"] > results["ethanol_pct"]).all()
                )
                for additive in ADDITIVE_LABELS:
                    if additive != selected_additive:
                        self.assertTrue((results[additive] == 0.0).all())
                self.assertTrue(
                    np.isfinite(
                        results[["bte_pct", "bsfc_g_kwh", "co_vol_pct", "hc_ppm"]]
                    ).all().all()
                )
                if selected_additive == "n_methylaniline_pct":
                    nma_rows = results[results[selected_additive] > 0.0]
                    self.assertTrue((nma_rows["bte_pct"] <= 37.0).all())

    def test_clean_dataset_has_only_valid_compositions(self):
        data = pd.read_csv(PROCESSED_DIR / "experimental_clean.csv")
        totals = data[COMPOSITION_FEATURES].sum(axis=1)
        self.assertTrue(((totals - 100.0).abs() < 1e-6).all())


if __name__ == "__main__":
    unittest.main()
