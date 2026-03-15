import unittest

import numpy as np

from nonlinear_lab.features import growth_rate, make_regression_df
from nonlinear_lab.models import generate_base_process
from nonlinear_lab.regression import (
    fit_enter_with_beta,
    rolling_window_regression,
    stepwise_frequency,
    stepwise_selection,
)


class FeatureEngineeringTests(unittest.TestCase):
    def test_growth_rate_matches_base_model_identity(self) -> None:
        series = generate_base_process(a=0.8, k=1.0, x0=0.1, steps=6, clip_max=None)
        omega = growth_rate(series)
        expected = 0.8 * (1.0 - series[:-1])
        np.testing.assert_allclose(omega, expected, rtol=1e-7, atol=1e-9)

    def test_make_regression_df_builds_current_and_lagged_features(self) -> None:
        series = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=float)
        df = make_regression_df(series, lags=2)
        self.assertEqual(list(df.columns), ["omega", "X_n", "Lag_1", "Lag_2"])
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.loc[0, "X_n"], 4.0)
        self.assertAlmostEqual(df.loc[0, "Lag_1"], 2.0)
        self.assertAlmostEqual(df.loc[0, "Lag_2"], 1.0)
        self.assertAlmostEqual(df.loc[0, "omega"], 1.0)


class RegressionTests(unittest.TestCase):
    def test_stepwise_selection_finds_true_signal(self) -> None:
        rng = np.random.default_rng(0)
        x_signal = np.linspace(-1.0, 1.0, 80)
        x_noise = rng.normal(size=80)
        y = 2.0 * x_signal + rng.normal(scale=0.01, size=80)

        import pandas as pd

        X = pd.DataFrame({"signal": x_signal, "noise": x_noise})

        selected = stepwise_selection(X, y, threshold_in=1e-4, threshold_out=0.01, max_steps=50)
        self.assertEqual(selected, ["signal"])

    def test_fit_enter_with_beta_recovers_base_slope(self) -> None:
        series = generate_base_process(a=0.8, k=1.0, x0=0.1, steps=80, clip_max=None)
        df = make_regression_df(series, lags=2)
        model, beta = fit_enter_with_beta(df.drop(columns=["omega"]), df["omega"])
        self.assertAlmostEqual(model.params["X_n"], -0.8, places=6)
        self.assertIn("X_n", beta.index)

    def test_rolling_window_regression_returns_expected_columns(self) -> None:
        series = generate_base_process(a=0.1, k=1.0, x0=0.001, steps=80, clip_max=None)
        windows = rolling_window_regression(series, window=25, lags=5, method="enter")
        self.assertFalse(windows.empty)
        self.assertTrue({"start", "end", "R2", "selected", "B_X_n"}.issubset(windows.columns))
        self.assertTrue((windows["R2"] > 0.99).all())

    def test_stepwise_frequency_counts_variable_occurrences(self) -> None:
        import pandas as pd

        roll_step = pd.DataFrame(
            {
                "selected": [
                    ["X_n", "Lag_1"],
                    ["Lag_1"],
                    ["X_n"],
                ]
            }
        )
        freq = stepwise_frequency(roll_step, lags=2).set_index("var")
        self.assertEqual(int(freq.loc["X_n", "count"]), 2)
        self.assertEqual(int(freq.loc["Lag_1", "count"]), 2)
        self.assertEqual(int(freq.loc["Lag_2", "count"]), 0)


if __name__ == "__main__":
    unittest.main()
