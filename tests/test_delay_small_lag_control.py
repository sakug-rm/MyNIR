import unittest


class DelaySmallLagControlTests(unittest.TestCase):
    def test_control_experiment_builds_expected_variants(self) -> None:
        from nonlinear_lab.delay_small_lag_control import build_delay_small_lag_control

        summary, paths, extra = build_delay_small_lag_control(steps=150, horizon=20)

        self.assertEqual(
            set(summary["variant"]),
            {"lag1_only", "stepwise_full", "thresholded"},
        )
        self.assertIn("true_nonlinear", set(paths["variant"]))
        self.assertIn("Lag_1", extra["meta"]["selected_features"])
        self.assertIn("Lag_3", extra["meta"]["selected_features"])
        self.assertIn("Lag_9", extra["meta"]["selected_features"])


if __name__ == "__main__":
    unittest.main()
