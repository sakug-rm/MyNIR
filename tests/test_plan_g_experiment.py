import unittest

import numpy as np


class PlanGExperimentTests(unittest.TestCase):
    def test_extract_window_diagnostics_returns_expected_features(self) -> None:
        from nonlinear_lab.plan_g_experiment import extract_window_diagnostics

        window = np.array([0.1, 0.2, 0.35, 0.5, 0.62, 0.7, 0.76, 0.8], dtype=float)
        features = extract_window_diagnostics(window)
        self.assertTrue(
            {
                "mean_x",
                "std_x",
                "rel_range",
                "omega_std",
                "turning_rate",
                "lag1_autocorr",
                "dominant_period",
                "dominant_acf",
                "corr_omega_x",
                "corr_omega_lag1",
            }.issubset(features.keys())
        )

    def test_oracle_regime_label_flags_plateau_and_collapse(self) -> None:
        from nonlinear_lab.plan_g_experiment import oracle_regime_label

        plateau = np.array([0.99, 1.0, 1.0, 1.0, 0.995, 1.0], dtype=float)
        collapse = np.array([1.2, 1.1, 0.9, 0.4, 0.1, 0.02], dtype=float)

        self.assertEqual(
            oracle_regime_label(plateau, regime_family="growth_no_memory", degenerate=True),
            "plateau_degenerate",
        )
        self.assertEqual(
            oracle_regime_label(collapse, regime_family="chaotic_informative", degenerate=False),
            "collapse",
        )

    def test_predict_regime_label_distinguishes_simple_patterns(self) -> None:
        from nonlinear_lab.plan_g_experiment import extract_window_diagnostics, predict_regime_label

        growth = np.array([0.05, 0.08, 0.12, 0.18, 0.27, 0.38, 0.5, 0.61, 0.7], dtype=float)
        cycle = np.tile(np.array([0.2, 0.8, 0.25, 0.75], dtype=float), 4)
        plateau = np.array([0.995, 1.0, 1.0, 0.999, 1.0, 1.0, 0.998, 1.0], dtype=float)

        self.assertEqual(predict_regime_label(extract_window_diagnostics(plateau)), "plateau_degenerate")
        self.assertIn(predict_regime_label(extract_window_diagnostics(growth)), {"growth_no_memory", "growth_with_memory"})
        self.assertIn(predict_regime_label(extract_window_diagnostics(cycle)), {"stable_cycle", "chaotic_informative"})

    def test_run_plan_g_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.plan_g_experiment import run_plan_g_experiment

        results = run_plan_g_experiment(steps=90, window=20, lags=5)
        self.assertIn("window_diagnostics", results)
        self.assertIn("classification_summary", results)
        self.assertIn("downstream_summary", results)
        self.assertIn("confusion_matrix", results)

        window_level = results["window_diagnostics"]
        self.assertFalse(window_level.empty)
        self.assertTrue(
            {
                "model",
                "case",
                "true_label",
                "predicted_label",
                "recommended_tool",
                "false_lag_rate",
                "hit_rate",
                "no_model",
            }.issubset(window_level.columns)
        )


if __name__ == "__main__":
    unittest.main()
