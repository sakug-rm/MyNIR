import unittest

import numpy as np


class PlanBExperimentTests(unittest.TestCase):
    def test_estimate_characteristic_period_finds_repeating_cycle(self) -> None:
        from nonlinear_lab.plan_b_experiment import estimate_characteristic_period

        cycle = np.array([1.0, 2.0, 3.0, 2.0], dtype=float)
        series = np.tile(cycle, 20)
        period = estimate_characteristic_period(series, min_lag=2, max_lag=20)
        self.assertEqual(period, 4)

    def test_score_selected_features_separates_true_and_false_lags(self) -> None:
        from nonlinear_lab.plan_b_experiment import score_selected_features

        score = score_selected_features(selected=["X_n", "Lag_1", "Lag_4"], true_predictors={"X_n", "Lag_1"})
        self.assertAlmostEqual(score["hit_rate"], 1.0)
        self.assertEqual(score["false_lag_count"], 1)
        self.assertAlmostEqual(score["false_lag_rate"], 1.0 / 3.0)

    def test_run_plan_b_experiment_emits_expected_columns(self) -> None:
        from nonlinear_lab.plan_b_experiment import run_plan_b_experiment

        results = run_plan_b_experiment(
            steps=80,
            window_sizes=[10, 15],
            lag_options=[3],
            threshold_in=0.01,
            threshold_out=0.05,
        )

        self.assertIn("window_level", results)
        self.assertIn("case_summary", results)
        self.assertIn("overall_summary", results)
        self.assertIn("period_summary", results)

        window_level = results["window_level"]
        self.assertFalse(window_level.empty)
        self.assertTrue(
            {
                "model",
                "case",
                "window",
                "lags",
                "selected_count",
                "false_lag_count",
                "false_lag_rate",
                "hit_rate",
                "rho",
                "period",
            }.issubset(window_level.columns)
        )

    def test_summarize_plan_b_results_builds_rho_bands(self) -> None:
        import pandas as pd

        from nonlinear_lab.plan_b_experiment import summarize_plan_b_results

        frame = pd.DataFrame(
            {
                "model": ["base", "base", "delay"],
                "window": [10, 20, 30],
                "lags": [3, 3, 5],
                "rho": [0.8, 1.4, 2.2],
                "false_lag_rate": [0.6, 0.3, 0.1],
                "hit_rate": [0.5, 0.8, 1.0],
                "no_model": [0, 0, 1],
                "b_sum": [0.0, 0.1, -0.8],
                "period": [12, 12, 10],
                "selected_count": [2, 1, 0],
                "false_lag_count": [1, 0, 0],
            }
        )
        summary = summarize_plan_b_results(frame)
        self.assertIn("rho_band", summary.columns)
        self.assertEqual(set(summary["rho_band"]), {"rho<1", "1<=rho<2", "rho>=2"})


if __name__ == "__main__":
    unittest.main()
