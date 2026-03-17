import unittest

import numpy as np


class PlanFExperimentTests(unittest.TestCase):
    def test_choose_component_count_respects_rho(self) -> None:
        from nonlinear_lab.plan_f_experiment import choose_component_count

        explained = np.array([0.6, 0.2, 0.15, 0.05])
        self.assertEqual(choose_component_count(explained, rho=0.8), 2)
        self.assertEqual(choose_component_count(explained, rho=0.9), 3)
        self.assertEqual(choose_component_count(explained, rho=0.99), 4)

    def test_run_plan_f_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.plan_f_experiment import run_plan_f_experiment

        results = run_plan_f_experiment(
            steps=90,
            window=20,
            lags=3,
            rho=0.9,
            rho_grid=[0.8, 0.9],
        )

        self.assertIn("window_level", results)
        self.assertIn("case_summary", results)
        self.assertIn("overall_summary", results)
        self.assertIn("rho_summary", results)
        self.assertIn("component_summary", results)

        window_level = results["window_level"]
        self.assertFalse(window_level.empty)
        self.assertTrue(
            {
                "case",
                "model",
                "cond_original",
                "cond_pcr",
                "retained_components",
                "enter_true_mass_share",
                "pcr_true_mass_share",
                "component_dominance",
            }.issubset(window_level.columns)
        )
        self.assertTrue((window_level["retained_components"] >= 1).all())

    def test_component_dominance_is_bounded(self) -> None:
        from nonlinear_lab.plan_f_experiment import _component_dominance

        loadings = np.array([[0.5, 0.5], [0.9, 0.1]])
        score = _component_dominance(loadings)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
