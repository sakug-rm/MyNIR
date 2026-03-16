import unittest

import numpy as np
import pandas as pd


class PlanHExperimentTests(unittest.TestCase):
    def test_compute_condition_number_is_reduced_by_standardization(self) -> None:
        from nonlinear_lab.plan_h_experiment import compute_condition_number

        X = pd.DataFrame(
            {
                "X_n": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Lag_1": [1000.0, 1999.0, 3001.0, 3998.0, 5002.0],
                "Lag_2": [0.01, 0.02, 0.031, 0.041, 0.051],
            }
        )

        raw = compute_condition_number(X, standardize=False)
        scaled = compute_condition_number(X, standardize=True)

        self.assertGreater(raw, scaled)

    def test_compute_vif_detects_collinearity(self) -> None:
        from nonlinear_lab.plan_h_experiment import compute_vif

        X = pd.DataFrame(
            {
                "X_n": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Lag_1": [2.0, 4.0, 6.0, 8.0, 10.0],
                "Lag_2": [1.0, 1.5, 2.0, 2.5, 3.0],
            }
        )

        vif = compute_vif(X)
        self.assertTrue(np.isinf(vif["X_n"]) or vif["X_n"] > 100.0)
        self.assertTrue(np.isinf(vif["Lag_1"]) or vif["Lag_1"] > 100.0)

    def test_run_plan_h_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.plan_h_experiment import run_plan_h_experiment

        results = run_plan_h_experiment(steps=90, window=25, lags=10)

        self.assertIn("window_level", results)
        self.assertIn("case_summary", results)
        self.assertIn("overall_summary", results)
        self.assertIn("risk_grid", results)
        self.assertIn("threshold_summary", results)

        window_level = results["window_level"]
        self.assertFalse(window_level.empty)
        self.assertTrue(
            {
                "var_x",
                "var_omega",
                "cond_raw",
                "cond_scaled",
                "max_vif",
                "false_lag_rate",
                "failure_type",
                "degraded_window",
            }.issubset(window_level.columns)
        )
        self.assertIn("low_dispersion", set(window_level["failure_type"]))


if __name__ == "__main__":
    unittest.main()
