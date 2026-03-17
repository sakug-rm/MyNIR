import unittest


class PlanEExperimentTests(unittest.TestCase):
    def test_interpretability_label_uses_variance_first(self) -> None:
        from nonlinear_lab.plan_e_experiment import classify_interpretability_window

        self.assertEqual(
            classify_interpretability_window(var_omega=1e-10, cond_scaled=1e5),
            "low_dispersion",
        )
        self.assertEqual(
            classify_interpretability_window(var_omega=1e-5, cond_scaled=1e18),
            "collinearity_heavy",
        )
        self.assertEqual(
            classify_interpretability_window(var_omega=1e-5, cond_scaled=1e5),
            "interpretable",
        )

    def test_alpha_configs_are_monotone(self) -> None:
        from nonlinear_lab.plan_e_experiment import alpha_configs

        configs = alpha_configs([0.01, 0.05, 0.10])
        self.assertEqual([cfg["alpha"] for cfg in configs], [0.01, 0.05, 0.10])
        self.assertTrue(all(cfg["threshold_in"] <= cfg["threshold_out"] for cfg in configs))

    def test_run_plan_e_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.plan_e_experiment import run_plan_e_experiment

        results = run_plan_e_experiment(
            steps=90,
            window=20,
            lags=3,
            alphas=(0.01, 0.05),
            cases=[
                {
                    "model": "base",
                    "case": "base_a_2_2",
                    "regime": "затухающие колебания",
                    "params": {"a": 2.2, "k": 1.0},
                    "true_predictors": {"X_n"},
                },
                {
                    "model": "delay",
                    "case": "delay_g_0_8",
                    "regime": "затухающие колебания",
                    "params": {"g": 0.8},
                    "true_predictors": {"Lag_1"},
                },
            ],
        )

        self.assertTrue({"window_level", "alpha_summary", "case_summary", "interpretability_summary"}.issubset(results))

        window_level = results["window_level"]
        self.assertFalse(window_level.empty)
        self.assertTrue(
            {
                "alpha",
                "case",
                "false_lag_rate",
                "miss_rate",
                "no_model",
                "exact_recovery",
                "interpretability",
                "selected_count",
            }.issubset(window_level.columns)
        )
        self.assertEqual(set(window_level["alpha"]), {0.01, 0.05})


if __name__ == "__main__":
    unittest.main()
