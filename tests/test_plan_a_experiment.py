import unittest

from nonlinear_lab.models import (
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
)


class PlanAExperimentTests(unittest.TestCase):
    def test_structural_regression_recovers_base_parameters_on_clean_window(self) -> None:
        from nonlinear_lab.plan_a_experiment import fit_structural_regression

        series = generate_base_process(a=0.8, k=1.0, x0=0.1, steps=40, clip_max=None)
        estimate = fit_structural_regression(series[:25], model_name="base")
        self.assertTrue(estimate["valid"])
        self.assertAlmostEqual(estimate["a"], 0.8, places=6)
        self.assertAlmostEqual(estimate["k"], 1.0, places=6)

    def test_structural_regression_recovers_delay_parameters_on_clean_window(self) -> None:
        from nonlinear_lab.plan_a_experiment import fit_structural_regression

        series = generate_delay_process(g=0.8, x0=0.1, steps=50, clip_max=None)
        estimate = fit_structural_regression(series[:25], model_name="delay")
        self.assertTrue(estimate["valid"])
        self.assertAlmostEqual(estimate["g"], 0.8, places=6)
        self.assertAlmostEqual(estimate["k"], 1.0, places=6)

    def test_structural_regression_recovers_mixed_parameters_on_clean_window(self) -> None:
        from nonlinear_lab.plan_a_experiment import fit_structural_regression

        series = generate_mixed_process(q=1.5, gamma=0.5, x0=0.1, steps=60, clip_max=None)
        estimate = fit_structural_regression(series[:25], model_name="mixed")
        self.assertTrue(estimate["valid"])
        self.assertAlmostEqual(estimate["q"], 1.5, places=6)
        self.assertAlmostEqual(estimate["gamma"], 0.5, places=6)
        self.assertAlmostEqual(estimate["k"], 1.0, places=6)

    def test_detect_degenerate_window_flags_near_constant_series(self) -> None:
        from nonlinear_lab.plan_a_experiment import detect_degenerate_window

        self.assertTrue(detect_degenerate_window([1.0, 1.0, 1.0, 1.0]))
        self.assertFalse(detect_degenerate_window([0.1, 0.2, 0.35, 0.5]))

    def test_identify_best_model_finds_true_family_on_clean_window(self) -> None:
        from nonlinear_lab.plan_a_experiment import identify_best_model

        base = generate_base_process(a=0.8, k=1.0, x0=0.1, steps=40, clip_max=None)
        delay = generate_delay_process(g=0.8, x0=0.1, steps=40, clip_max=None)
        mixed = generate_mixed_process(q=1.5, gamma=0.5, x0=0.1, steps=40, clip_max=None)

        self.assertEqual(identify_best_model(base[:25], estimator="regression")["best_model"], "base")
        self.assertEqual(identify_best_model(delay[:25], estimator="regression")["best_model"], "delay")
        self.assertEqual(identify_best_model(mixed[:25], estimator="regression")["best_model"], "mixed")

    def test_noise_sweep_reports_requested_levels(self) -> None:
        from nonlinear_lab.plan_a_experiment import run_plan_a_noise_sweep

        sweep = run_plan_a_noise_sweep(noise_levels=[0.0, 0.01], steps=60, regression_window=15, noise_seed=3)
        self.assertEqual(sorted(sweep["noise_level"].unique().tolist()), [0.0, 0.01])
        self.assertIn("valid_share", sweep.columns)


if __name__ == "__main__":
    unittest.main()
