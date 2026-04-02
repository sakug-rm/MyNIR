import unittest

import pandas as pd


class RealDataRoutingTests(unittest.TestCase):
    def test_run_ipp_routing_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.real_data_routing import run_ipp_routing_experiment

        dates = pd.date_range("2018-01-01", periods=36, freq="MS")
        rows = []
        for series_code, series_name, base in [
            ("B", "Добыча полезных ископаемых", 100.0),
            ("10", "Пищевые продукты", 90.0),
        ]:
            for variant, noise in [("adj_unsmoothed", 0.5), ("raw", 2.0)]:
                for idx, date in enumerate(dates):
                    rows.append(
                        {
                            "date": date,
                            "series_code": series_code,
                            "series_name": series_name,
                            "variant": variant,
                            "index_value": base + 0.7 * idx + (noise if idx % 2 else 0.0),
                        }
                    )

        ipp_long = pd.DataFrame(rows)
        results = run_ipp_routing_experiment(ipp_long, windows=[24], lags=3)

        self.assertIn("window_features", results)
        self.assertIn("regime_summary", results)
        self.assertIn("interpretability_summary", results)
        self.assertIn("routing_summary", results)
        self.assertFalse(results["window_features"].empty)
        self.assertTrue(
            {
                "series_code",
                "variant",
                "window",
                "regime_label",
                "interpretability_label",
                "preferred_tool",
                "var_omega",
                "cond_scaled",
            }.issubset(results["window_features"].columns)
        )


if __name__ == "__main__":
    unittest.main()
