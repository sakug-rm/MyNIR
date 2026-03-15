import unittest

import numpy as np

from nonlinear_lab.models import (
    fixed_point,
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
    theoretical_coeffs,
)


class ModelGenerationTests(unittest.TestCase):
    def test_base_generator_produces_expected_first_steps(self) -> None:
        series = generate_base_process(a=0.8, k=1.0, x0=0.1, steps=4, clip_max=None)
        expected = np.array([0.1, 0.172, 0.2859328, 0.44927299])
        np.testing.assert_allclose(series, expected, rtol=1e-7, atol=1e-9)

    def test_delay_generator_uses_previous_state(self) -> None:
        series = generate_delay_process(g=0.2, x0=0.1, steps=4, clip_max=None)
        expected = np.array([0.1, 0.118, 0.13924, 0.16380194])
        np.testing.assert_allclose(series, expected, rtol=1e-7, atol=1e-9)

    def test_mixed_generator_uses_current_and_previous_state(self) -> None:
        series = generate_mixed_process(q=1.5, gamma=0.5, x0=0.1, steps=4, clip_max=None)
        expected = np.array([0.1, 0.2275, 0.47405328, 0.76715795])
        np.testing.assert_allclose(series, expected, rtol=1e-6, atol=1e-9)

    def test_fixed_point_matches_theory(self) -> None:
        self.assertAlmostEqual(fixed_point(-0.2), 1.25)
        self.assertTrue(np.isnan(fixed_point(-1.0)))

    def test_theoretical_coeffs_match_mixed_growth_equation(self) -> None:
        coeffs = theoretical_coeffs(q=2.8, gamma=0.5)
        self.assertEqual(coeffs["B_X_n_theory"], -2.8)
        self.assertEqual(coeffs["B_Lag1_theory"], -1.4)


if __name__ == "__main__":
    unittest.main()
