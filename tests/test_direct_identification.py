import unittest

import numpy as np

from nonlinear_lab.models import (
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
)


class DirectIdentificationTests(unittest.TestCase):
    def test_base_triplet_recovers_parameters(self) -> None:
        from nonlinear_lab.direct_identification import estimate_base_from_triplet

        series = generate_base_process(a=0.8, k=1.0, x0=0.1, steps=5, clip_max=None)
        estimate = estimate_base_from_triplet(series[:3])
        self.assertAlmostEqual(estimate["a"], 0.8, places=8)
        self.assertAlmostEqual(estimate["k"], 1.0, places=8)
        self.assertTrue(estimate["valid"])

    def test_delay_quadruplet_recovers_parameters(self) -> None:
        from nonlinear_lab.direct_identification import estimate_delay_from_quadruplet

        series = generate_delay_process(g=0.8, x0=0.1, steps=6, clip_max=None)
        estimate = estimate_delay_from_quadruplet(series[:4])
        self.assertAlmostEqual(estimate["g"], 0.8, places=8)
        self.assertAlmostEqual(estimate["k"], 1.0, places=8)
        self.assertTrue(estimate["valid"])

    def test_mixed_quintet_recovers_parameters(self) -> None:
        from nonlinear_lab.direct_identification import estimate_mixed_from_quintet

        series = generate_mixed_process(q=1.5, gamma=0.5, x0=0.1, steps=7, clip_max=None)
        estimate = estimate_mixed_from_quintet(series[:5])
        self.assertAlmostEqual(estimate["q"], 1.5, places=8)
        self.assertAlmostEqual(estimate["gamma"], 0.5, places=8)
        self.assertAlmostEqual(estimate["k"], 1.0, places=8)
        self.assertTrue(estimate["valid"])

    def test_rolling_direct_identification_returns_windowed_estimates(self) -> None:
        from nonlinear_lab.direct_identification import rolling_direct_identification

        series = generate_base_process(a=0.8, k=1.0, x0=0.1, steps=8, clip_max=None)
        estimates = rolling_direct_identification(series, model_name="base")
        self.assertEqual(len(estimates), len(series) - 2)
        self.assertTrue({"start", "end", "valid", "a", "k"}.issubset(estimates.columns))
        self.assertTrue(estimates["valid"].all())

    def test_summarize_parameter_errors_reports_bias_and_valid_share(self) -> None:
        from nonlinear_lab.direct_identification import summarize_parameter_errors

        estimates = [
            {"valid": True, "a": 0.75, "k": 1.0},
            {"valid": True, "a": 0.85, "k": 1.0},
            {"valid": False, "a": np.nan, "k": np.nan},
        ]
        summary = summarize_parameter_errors(estimates, truth={"a": 0.8, "k": 1.0})
        self.assertAlmostEqual(summary["valid_share"], 2.0 / 3.0, places=8)
        self.assertAlmostEqual(summary["a_mae"], 0.05, places=8)
        self.assertAlmostEqual(summary["k_mae"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
