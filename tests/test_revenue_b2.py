import tempfile
import unittest
from pathlib import Path

import pandas as pd


class RevenueB2Tests(unittest.TestCase):
    def _make_wide(self) -> pd.DataFrame:
        rows = []
        sector_codes = ["15.11", "24.42", "40.30"]
        for sector_idx, sector in enumerate(sector_codes):
            for firm_idx in range(6):
                row = {
                    "inn": f"{sector_idx}{firm_idx:03d}",
                    "company_name": f"Firm {sector}-{firm_idx}",
                    "okved_old": sector,
                    "industry_text": f"Industry {sector}",
                    "is_summary_row": False,
                }
                for year in range(2003, 2023):
                    base = 1000 + sector_idx * 100 + firm_idx * 10
                    slope = (sector_idx + 1) * 15
                    row[f"revenue_{year}"] = float(base + slope * (year - 2003))
                rows.append(row)
        return pd.DataFrame(rows)

    def _make_long(self, wide: pd.DataFrame) -> pd.DataFrame:
        value_columns = [column for column in wide.columns if column.startswith("revenue_")]
        long = wide.melt(
            id_vars=["inn", "company_name", "okved_old", "industry_text", "is_summary_row"],
            value_vars=value_columns,
            var_name="year_col",
            value_name="revenue",
        )
        long["year"] = long["year_col"].str.replace("revenue_", "", regex=False).astype(int)
        long["address"] = ""
        long["region"] = ""
        long["rank"] = 1
        long["okved_missing"] = False
        return long[
            [
                "rank",
                "company_name",
                "address",
                "inn",
                "region",
                "okved_old",
                "industry_text",
                "is_summary_row",
                "revenue",
                "year",
                "okved_missing",
            ]
        ]

    def test_run_b2_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.revenue_b2 import run_b2_experiment

        wide = self._make_wide()
        long = self._make_long(wide)
        results = run_b2_experiment(wide, long, min_firms=5)

        self.assertIn("sector_summary", results)
        self.assertIn("interval_summary", results)
        self.assertIn("window_summary", results)
        self.assertTrue({"sector_2d", "included"}.issubset(results["sector_summary"].columns))
        self.assertFalse(results["interval_summary"].empty)

    def test_save_b2_results_writes_expected_files(self) -> None:
        from nonlinear_lab.revenue_b2 import run_b2_experiment, save_b2_results

        wide = self._make_wide()
        long = self._make_long(wide)
        results = run_b2_experiment(wide, long, min_firms=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_b2_results(results, Path(tmpdir))
            self.assertTrue((out / "sector_summary.csv").exists())
            self.assertTrue((out / "sector_trajectories.png").exists())
            self.assertTrue((out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
