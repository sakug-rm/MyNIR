import unittest

import pandas as pd


class RealDataStructuralTests(unittest.TestCase):
    def test_run_ipp_structural_reading_emits_expected_outputs(self) -> None:
        from nonlinear_lab.real_data_routing import run_ipp_routing_experiment
        from nonlinear_lab.real_data_structural import run_ipp_structural_reading

        dates = pd.date_range("2018-01-01", periods=36, freq="MS")
        rows = []
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "series_code": "10",
                    "series_name": "Пищевые продукты",
                    "variant": "adj_unsmoothed",
                    "index_value": 90.0 + 0.8 * idx + (0.5 if idx % 3 == 0 else 0.0),
                }
            )

        ipp_long = pd.DataFrame(rows)
        routing = run_ipp_routing_experiment(ipp_long, windows=[24], lags=3, variant_filter=["adj_unsmoothed"])
        results = run_ipp_structural_reading(ipp_long, routing["window_features"], lags=3)

        self.assertIn("window_structures", results)
        self.assertIn("series_summary", results)
        self.assertIn("mode_summary", results)
        self.assertFalse(results["window_structures"].empty)
        self.assertTrue(
            {
                "series_code",
                "reading_mode",
                "top_beta_predictor",
                "b_sum",
                "selected_count",
            }.issubset(results["window_structures"].columns)
        )


if __name__ == "__main__":
    unittest.main()
