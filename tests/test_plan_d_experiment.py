import unittest

import numpy as np
import pandas as pd


class PlanDExperimentTests(unittest.TestCase):
    def test_true_coefficients_match_structural_models(self) -> None:
        from nonlinear_lab.plan_d_experiment import true_coefficients_for_case

        self.assertEqual(true_coefficients_for_case("base", {"a": 2.2}), {"X_n": -2.2})
        self.assertEqual(true_coefficients_for_case("delay", {"g": 0.8}), {"Lag_1": -0.8})
        self.assertEqual(
            true_coefficients_for_case("mixed", {"q": 2.8, "gamma": 0.5}),
            {"X_n": -2.8, "Lag_1": -1.4},
        )

    def test_active_features_respects_threshold(self) -> None:
        from nonlinear_lab.plan_d_experiment import active_features_from_coefficients

        coeffs = pd.Series({"X_n": 0.12, "Lag_1": 0.03, "Lag_2": -0.08})
        active = active_features_from_coefficients(coeffs, threshold=0.05)
        self.assertEqual(active, ["X_n", "Lag_2"])

    def test_run_plan_d_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.plan_d_experiment import run_plan_d_experiment

        results = run_plan_d_experiment(
            steps=90,
            window=20,
            lags=3,
            active_threshold=0.05,
            cases=[
                {
                    "model": "base",
                    "case": "base_a_2_2",
                    "regime": "затухающие колебания",
                    "params": {"a": 2.2, "k": 1.0},
                    "true_predictors": {"X_n"},
                },
                {
                    "model": "mixed",
                    "case": "mixed_q_1_5_gamma_0_5",
                    "regime": "смешанная память, умеренный режим",
                    "params": {"q": 1.5, "gamma": 0.5},
                    "true_predictors": {"X_n", "Lag_1"},
                },
            ],
        )

        self.assertIn("window_level", results)
        self.assertIn("case_summary", results)
        self.assertIn("overall_summary", results)
        self.assertIn("sign_summary", results)

        window_level = results["window_level"]
        self.assertFalse(window_level.empty)
        self.assertTrue(
            {
                "method",
                "case",
                "model",
                "false_lag_count",
                "false_lag_rate",
                "hit_rate",
                "support_size",
                "coef_mae",
                "sign_correct_share",
                "validation_r2",
            }.issubset(window_level.columns)
        )
        self.assertTrue({"enter", "stepwise", "ridge", "lasso", "elastic_net"}.issubset(set(window_level["method"])))

    def test_support_scoring_distinguishes_true_and_false_predictors(self) -> None:
        from nonlinear_lab.plan_d_experiment import score_support

        score = score_support(
            selected=["X_n", "Lag_1", "Lag_4"],
            true_predictors={"X_n", "Lag_1"},
        )
        self.assertAlmostEqual(score["hit_rate"], 1.0)
        self.assertEqual(score["false_lag_count"], 1)
        self.assertAlmostEqual(score["false_lag_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(score["true_lag_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
