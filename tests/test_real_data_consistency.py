import unittest

import pandas as pd


class RealDataConsistencyTests(unittest.TestCase):
    def test_run_ipp_consistency_analysis_emits_expected_outputs(self) -> None:
        from nonlinear_lab.real_data_consistency import run_ipp_consistency_analysis

        routing = pd.DataFrame(
            {
                "series_code": ["10", "10", "10", "21", "21", "21"],
                "variant": ["adj_smoothed", "adj_unsmoothed", "raw"] * 2,
                "window": [24] * 6,
                "regime_label": ["plateau_degenerate", "turbulent_informative", "turbulent_informative", "turbulent_informative", "turbulent_informative", "turbulent_informative"],
                "interpretability_label": ["plateau_degenerate", "interpretable", "interpretable", "interpretable", "interpretable", "collinearity_heavy"],
                "preferred_tool": ["do_not_read_regression", "phase_short_forecast", "phase_short_forecast", "phase_short_forecast", "phase_short_forecast", "do_not_read_regression"],
            }
        )
        structural = pd.DataFrame(
            {
                "series_code": ["10", "10", "21", "21"],
                "variant": ["adj_unsmoothed", "raw", "adj_unsmoothed", "raw"],
                "window": [24, 24, 24, 24],
                "reading_mode": ["phase_caution", "phase_caution", "phase_caution", "beta_bsum_caution"],
                "top_beta_predictor": ["X_n", "X_n", "Lag_1", "Lag_1"],
                "top_b_predictor": ["X_n", "Lag_1", "Lag_1", "Lag_1"],
                "interpretability_label": ["interpretable", "interpretable", "interpretable", "collinearity_heavy"],
            }
        )

        results = run_ipp_consistency_analysis(routing, structural)

        self.assertIn("variant_consistency", results)
        self.assertIn("series_consistency", results)
        self.assertIn("overall_summary", results)
        self.assertFalse(results["series_consistency"].empty)
        self.assertTrue(
            {
                "series_code",
                "window",
                "regime_consistency",
                "interpretability_consistency",
                "top_beta_consistency",
            }.issubset(results["series_consistency"].columns)
        )


if __name__ == "__main__":
    unittest.main()
