import tempfile
import unittest
from pathlib import Path

import pandas as pd


class RealDataCasesTests(unittest.TestCase):
    def test_run_ipp_case_cards_produces_expected_outputs(self) -> None:
        from nonlinear_lab.real_data_cases import run_ipp_case_cards

        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        ipp_long = pd.DataFrame(
            {
                "date": list(dates) * 2 * 3,
                "series_code": ["10"] * 36 + ["C"] * 36,
                "series_name": ["Food"] * 36 + ["Manufacturing"] * 36,
                "variant": ["adj_smoothed"] * 12 + ["adj_unsmoothed"] * 12 + ["raw"] * 12 + ["adj_smoothed"] * 12 + ["adj_unsmoothed"] * 12 + ["raw"] * 12,
                "index_value": list(range(100, 112)) * 6,
            }
        )
        hierarchy = pd.DataFrame(
            {
                "series_code": ["10"],
                "series_name": ["Food"],
                "parent_code": ["C"],
                "parent_name": ["Manufacturing"],
            }
        )
        routing = pd.DataFrame(
            {
                "series_code": ["10", "10", "C", "C"],
                "series_name": ["Food", "Food", "Manufacturing", "Manufacturing"],
                "variant": ["adj_unsmoothed"] * 4,
                "window": [24, 24, 24, 24],
                "start": [0, 1, 0, 1],
                "end": [10, 11, 10, 11],
                "regime_label": ["turbulent_informative", "turbulent_informative", "plateau_degenerate", "plateau_degenerate"],
                "interpretability_label": ["interpretable", "interpretable", "plateau_degenerate", "plateau_degenerate"],
                "preferred_tool": ["phase_short_forecast", "phase_short_forecast", "do_not_read_regression", "do_not_read_regression"],
                "center_date": pd.to_datetime(["2020-06-01", "2020-07-01", "2020-06-01", "2020-07-01"]),
            }
        )
        structural = pd.DataFrame(
            {
                "series_code": ["10", "10", "C", "C"],
                "series_name": ["Food", "Food", "Manufacturing", "Manufacturing"],
                "variant": ["adj_unsmoothed"] * 4,
                "window": [24, 24, 24, 24],
                "start": [0, 1, 0, 1],
                "end": [10, 11, 10, 11],
                "regime_label": ["turbulent_informative"] * 4,
                "interpretability_label": ["interpretable", "interpretable", "plateau_degenerate", "plateau_degenerate"],
                "reading_mode": ["phase_caution", "phase_caution", "do_not_read", "do_not_read"],
                "R2_enter": [0.4, 0.5, 0.9, 0.9],
                "top_beta_predictor": ["X_n", "Lag_1", "X_n", "X_n"],
                "top_b_predictor": ["Lag_1", "Lag_1", "X_n", "X_n"],
                "b_sum": [0.01, 0.02, 0.0, 0.0],
                "selected": ["[]", "['Lag_1']", "[]", "[]"],
                "selected_count": [0.0, 1.0, 0.0, 0.0],
                "beta_x": [0.5, 0.4, 0.2, 0.2],
                "beta_lag1": [0.1, 0.3, 0.0, 0.0],
                "b_x": [0.2, 0.1, 0.0, 0.0],
                "b_lag1": [0.05, 0.06, 0.0, 0.0],
                "center_date": pd.to_datetime(["2020-06-01", "2020-07-01", "2020-06-01", "2020-07-01"]),
            }
        )

        results = run_ipp_case_cards(
            ipp_long,
            hierarchy,
            routing,
            structural,
            case_codes=("10",),
        )

        self.assertIn("case_summary", results)
        self.assertIn("shock_summary", results)
        self.assertIn("hierarchy_summary", results)
        self.assertEqual(results["case_summary"]["series_code"].tolist(), ["10"])
        self.assertTrue({"dominant_mode", "dominant_top_beta", "shock_2020_shock_regime"}.issubset(results["case_summary"].columns))

    def test_save_ipp_case_card_results_writes_files(self) -> None:
        from nonlinear_lab.real_data_cases import run_ipp_case_cards, save_ipp_case_card_results

        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        ipp_long = pd.DataFrame(
            {
                "date": list(dates) * 2 * 3,
                "series_code": ["10"] * 36 + ["C"] * 36,
                "series_name": ["Food"] * 36 + ["Manufacturing"] * 36,
                "variant": ["adj_smoothed"] * 12 + ["adj_unsmoothed"] * 12 + ["raw"] * 12 + ["adj_smoothed"] * 12 + ["adj_unsmoothed"] * 12 + ["raw"] * 12,
                "index_value": list(range(100, 112)) * 6,
            }
        )
        hierarchy = pd.DataFrame(
            {
                "series_code": ["10"],
                "series_name": ["Food"],
                "parent_code": ["C"],
                "parent_name": ["Manufacturing"],
            }
        )
        routing = pd.DataFrame(
            {
                "series_code": ["10", "10", "C", "C"],
                "series_name": ["Food", "Food", "Manufacturing", "Manufacturing"],
                "variant": ["adj_unsmoothed"] * 4,
                "window": [24, 24, 24, 24],
                "start": [0, 1, 0, 1],
                "end": [10, 11, 10, 11],
                "regime_label": ["turbulent_informative", "turbulent_informative", "plateau_degenerate", "plateau_degenerate"],
                "interpretability_label": ["interpretable", "interpretable", "plateau_degenerate", "plateau_degenerate"],
                "preferred_tool": ["phase_short_forecast", "phase_short_forecast", "do_not_read_regression", "do_not_read_regression"],
                "center_date": pd.to_datetime(["2020-06-01", "2020-07-01", "2020-06-01", "2020-07-01"]),
            }
        )
        structural = pd.DataFrame(
            {
                "series_code": ["10", "10", "C", "C"],
                "series_name": ["Food", "Food", "Manufacturing", "Manufacturing"],
                "variant": ["adj_unsmoothed"] * 4,
                "window": [24, 24, 24, 24],
                "start": [0, 1, 0, 1],
                "end": [10, 11, 10, 11],
                "regime_label": ["turbulent_informative"] * 4,
                "interpretability_label": ["interpretable", "interpretable", "plateau_degenerate", "plateau_degenerate"],
                "reading_mode": ["phase_caution", "phase_caution", "do_not_read", "do_not_read"],
                "R2_enter": [0.4, 0.5, 0.9, 0.9],
                "top_beta_predictor": ["X_n", "Lag_1", "X_n", "X_n"],
                "top_b_predictor": ["Lag_1", "Lag_1", "X_n", "X_n"],
                "b_sum": [0.01, 0.02, 0.0, 0.0],
                "selected": ["[]", "['Lag_1']", "[]", "[]"],
                "selected_count": [0.0, 1.0, 0.0, 0.0],
                "beta_x": [0.5, 0.4, 0.2, 0.2],
                "beta_lag1": [0.1, 0.3, 0.0, 0.0],
                "b_x": [0.2, 0.1, 0.0, 0.0],
                "b_lag1": [0.05, 0.06, 0.0, 0.0],
                "center_date": pd.to_datetime(["2020-06-01", "2020-07-01", "2020-06-01", "2020-07-01"]),
            }
        )
        results = run_ipp_case_cards(ipp_long, hierarchy, routing, structural, case_codes=("10",))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = save_ipp_case_card_results(results, Path(tmpdir))
            self.assertTrue((output_dir / "case_summary.csv").exists())
            self.assertTrue((output_dir / "case_cards.md").exists())
            self.assertTrue((output_dir / "card_10.png").exists())


if __name__ == "__main__":
    unittest.main()
