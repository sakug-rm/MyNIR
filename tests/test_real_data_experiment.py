import unittest

import pandas as pd


class RealDataExperimentTests(unittest.TestCase):
    def test_run_ipp_variant_qc_emits_expected_columns(self) -> None:
        from nonlinear_lab.real_data_experiment import run_ipp_variant_qc

        dates = pd.date_range("2018-01-01", periods=18, freq="MS")
        rows = []
        for variant, scale in [("adj_smoothed", 1.00), ("adj_unsmoothed", 1.02), ("raw", 0.98)]:
            for idx, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "series_code": "B",
                        "series_name": "Добыча полезных ископаемых",
                        "variant": variant,
                        "index_value": 100.0 + scale * idx + (0.2 if variant == "raw" and idx % 2 else 0.0),
                    }
                )

        ipp_long = pd.DataFrame(rows)
        results = run_ipp_variant_qc(ipp_long, windows=[12], lag_options=[3])

        self.assertIn("series_summary", results)
        self.assertIn("window_summary", results)
        self.assertIn("variant_summary", results)
        self.assertFalse(results["series_summary"].empty)
        self.assertTrue(
            {
                "series_code",
                "variant",
                "window",
                "lags",
                "var_omega",
                "acf1",
                "no_model_share",
                "mean_cond",
                "sign_stability_x",
            }.issubset(results["series_summary"].columns)
        )


if __name__ == "__main__":
    unittest.main()
