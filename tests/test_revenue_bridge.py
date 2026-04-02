import tempfile
import unittest
from pathlib import Path

import pandas as pd


class RevenueBridgeTests(unittest.TestCase):
    def test_run_bridge_case_emits_expected_outputs(self) -> None:
        from nonlinear_lab.revenue_bridge import run_bridge_case

        dates = pd.date_range("2014-01-01", periods=9 * 12, freq="MS")
        ipp_long = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "series_code": ["10"] * len(dates) + ["36"] * len(dates),
                "series_name": ["Food IPP"] * len(dates) + ["Water IPP"] * len(dates),
                "variant": ["adj_unsmoothed"] * (2 * len(dates)),
                "index_value": list(range(100, 100 + len(dates))) + list(range(200, 200 + len(dates))),
            }
        )
        b2_yearly = pd.DataFrame(
            {
                "sector_2d_code": ["15"] * 9 + ["41"] * 9,
                "sector_label": ["15 — Food"] * 9 + ["41 — Water"] * 9,
                "year": list(range(2014, 2023)) * 2,
                "median_revenue": [100 + i * 5 for i in range(9)] + [200 + i * 2 for i in range(9)],
                "firms": [100] * 18,
            }
        )
        bridge_map = (
            {
                "bridge_id": "food",
                "revenue_sector_2d": "15",
                "revenue_label_hint": "food",
                "ipp_series_code": "10",
                "ipp_label_hint": "food",
                "mapping_quality": "high",
                "note": "test",
            },
            {
                "bridge_id": "water",
                "revenue_sector_2d": "41",
                "revenue_label_hint": "water",
                "ipp_series_code": "36",
                "ipp_label_hint": "water",
                "mapping_quality": "medium",
                "note": "test",
            },
        )

        results = run_bridge_case(ipp_long, b2_yearly, bridge_map=bridge_map)

        self.assertIn("bridge_comparison", results)
        self.assertIn("shock_comparison", results)
        self.assertEqual(set(results["bridge_mapping"]["bridge_id"]), {"food", "water"})
        self.assertFalse(results["bridge_comparison"].empty)

    def test_save_bridge_results_writes_expected_files(self) -> None:
        from nonlinear_lab.revenue_bridge import run_bridge_case, save_bridge_results

        dates = pd.date_range("2014-01-01", periods=9 * 12, freq="MS")
        ipp_long = pd.DataFrame(
            {
                "date": list(dates),
                "series_code": ["10"] * len(dates),
                "series_name": ["Food IPP"] * len(dates),
                "variant": ["adj_unsmoothed"] * len(dates),
                "index_value": list(range(100, 100 + len(dates))),
            }
        )
        b2_yearly = pd.DataFrame(
            {
                "sector_2d_code": ["15"] * 9,
                "sector_label": ["15 — Food"] * 9,
                "year": list(range(2014, 2023)),
                "median_revenue": [100 + i * 5 for i in range(9)],
                "firms": [100] * 9,
            }
        )
        bridge_map = (
            {
                "bridge_id": "food",
                "revenue_sector_2d": "15",
                "revenue_label_hint": "food",
                "ipp_series_code": "10",
                "ipp_label_hint": "food",
                "mapping_quality": "high",
                "note": "test",
            },
        )
        results = run_bridge_case(ipp_long, b2_yearly, bridge_map=bridge_map)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_bridge_results(results, Path(tmpdir))
            self.assertTrue((out / "bridge_mapping.csv").exists())
            self.assertTrue((out / "bridge_trajectories.png").exists())
            self.assertTrue((out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
