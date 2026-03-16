import unittest

import pandas as pd


class PlanCExperimentTests(unittest.TestCase):
    def test_compute_ranking_metrics_scores_true_predictors(self) -> None:
        from nonlinear_lab.plan_c_experiment import compute_ranking_metrics

        scores = pd.Series({"X_n": 0.8, "Lag_1": 0.6, "Lag_2": 0.2, "Lag_3": 0.1})
        metrics = compute_ranking_metrics(scores, true_predictors={"X_n", "Lag_1"})

        self.assertEqual(metrics["top_predictor"], "X_n")
        self.assertEqual(metrics["correct_top"], 1.0)
        self.assertEqual(metrics["topk_recall"], 1.0)
        self.assertEqual(metrics["pairwise_score"], 1.0)
        self.assertGreater(metrics["margin"], 0.0)

    def test_run_plan_c_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.plan_c_experiment import run_plan_c_experiment

        results = run_plan_c_experiment(steps=90, window=25, lags=10)

        self.assertIn("window_variable_level", results)
        self.assertIn("window_summary", results)
        self.assertIn("case_summary", results)
        self.assertIn("overall_summary", results)
        self.assertIn("frequency_summary", results)
        self.assertIn("example_windows", results)

        window_summary = results["window_summary"]
        self.assertFalse(window_summary.empty)
        self.assertTrue(
            {
                "model",
                "case",
                "start",
                "problem_window",
                "correct_top_b",
                "correct_top_beta",
                "pairwise_b",
                "pairwise_beta",
                "top_b",
                "top_beta",
            }.issubset(window_summary.columns)
        )

        case_summary = results["case_summary"]
        base_case = case_summary[case_summary["case"] == "base_a_2_2"].iloc[0]
        self.assertGreater(base_case["problem_correct_top_beta"], base_case["problem_correct_top_b"])


if __name__ == "__main__":
    unittest.main()
