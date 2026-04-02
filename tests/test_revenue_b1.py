import tempfile
import unittest
from pathlib import Path

import pandas as pd


class RevenueB1Tests(unittest.TestCase):
    def _make_wide(self) -> pd.DataFrame:
        rows = []
        for idx in range(30):
            base = 100 + idx
            if idx < 10:
                growth = 0.2
            elif idx < 20:
                growth = 1.0
            else:
                growth = 3.0
            row = {
                "inn": f"{idx:04d}",
                "company_name": f"Firm {idx}",
                "is_summary_row": False,
            }
            for year in range(2003, 2023):
                if year <= 2010:
                    value = base + (year - 2003) * 5
                elif year <= 2014:
                    progress = (year - 2010) / 4
                    value = (base + 35) * (1 + growth * progress)
                else:
                    value = (base + 35) * (1 + growth) * (1 + 0.1 * (year - 2014))
                row[f"revenue_{year}"] = float(value)
            rows.append(row)
        return pd.DataFrame(rows)

    def _make_long(self, wide: pd.DataFrame) -> pd.DataFrame:
        value_columns = [column for column in wide.columns if column.startswith("revenue_")]
        long = wide.melt(
            id_vars=["inn", "company_name", "is_summary_row"],
            value_vars=value_columns,
            var_name="year_col",
            value_name="revenue",
        )
        long["year"] = long["year_col"].str.replace("revenue_", "", regex=False).astype(int)
        long["address"] = ""
        long["region"] = ""
        long["okved_old"] = ""
        long["industry_text"] = ""
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

    def test_run_b1_experiment_emits_expected_outputs(self) -> None:
        from nonlinear_lab.revenue_b1 import run_b1_experiment

        wide = self._make_wide()
        long = self._make_long(wide)
        results = run_b1_experiment(wide, long)

        self.assertIn("group_summary", results)
        self.assertIn("interval_summary", results)
        self.assertIn("window_summary", results)
        self.assertEqual(set(results["group_summary"]["b1_group"]), {"high_90_50", "low_10_50", "middle_45_55_50"})
        self.assertTrue({"interval", "best_spec", "dominant_tool"}.issubset(results["interval_summary"].columns))

    def test_save_b1_results_writes_expected_files(self) -> None:
        from nonlinear_lab.revenue_b1 import run_b1_experiment, save_b1_results

        wide = self._make_wide()
        long = self._make_long(wide)
        results = run_b1_experiment(wide, long)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_b1_results(results, Path(tmpdir))
            self.assertTrue((out / "group_summary.csv").exists())
            self.assertTrue((out / "group_trajectories.png").exists())
            self.assertTrue((out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
