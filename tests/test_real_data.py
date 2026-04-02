import unittest

import numpy as np
import pandas as pd


class RealDataPreparationTests(unittest.TestCase):
    def test_reshape_ipp_wide_to_long_assigns_triplet_variants(self) -> None:
        from nonlinear_lab.real_data import reshape_ipp_wide_to_long

        df = pd.DataFrame(
            {
                "label": [
                    "B Добыча полезных ископаемых",
                    "B Добыча полезных ископаемых",
                    "B Добыча полезных ископаемых",
                    "05 Добыча угля",
                    "05 Добыча угля",
                    "05 Добыча угля",
                    np.nan,
                    "Примечание. служебная строка",
                ],
                pd.Timestamp("2013-01-01"): [100, 101, 102, 200, 201, 202, np.nan, np.nan],
                pd.Timestamp("2013-02-01"): [110, 111, 112, 210, 211, 212, np.nan, np.nan],
            }
        )

        long_df = reshape_ipp_wide_to_long(df)

        self.assertEqual(set(long_df["variant"]), {"adj_smoothed", "adj_unsmoothed", "raw"})
        self.assertEqual(long_df["series_label"].nunique(), 2)
        self.assertEqual(len(long_df), 12)

        first = long_df[
            (long_df["series_code"] == "B")
            & (long_df["variant"] == "adj_smoothed")
            & (long_df["date"] == pd.Timestamp("2013-01-01"))
        ]
        self.assertEqual(float(first["index_value"].iloc[0]), 100.0)

    def test_build_ipp_hierarchy_links_dotted_and_section_rows(self) -> None:
        from nonlinear_lab.real_data import build_ipp_hierarchy

        series_df = pd.DataFrame(
            {
                "series_code": ["ПРОМ", "B", "06", "06.1", "06.2", "C", "10"],
                "series_name": [
                    "Промышленность - всего",
                    "Добыча полезных ископаемых",
                    "Добыча сырой нефти и природного газа",
                    "Добыча сырой нефти",
                    "Добыча природного газа",
                    "Обрабатывающие производства",
                    "Пищевые продукты",
                ],
            }
        )

        hierarchy = build_ipp_hierarchy(series_df)
        parent_map = dict(zip(hierarchy["series_code"], hierarchy["parent_code"]))

        self.assertIsNone(parent_map["ПРОМ"])
        self.assertEqual(parent_map["B"], "ПРОМ")
        self.assertEqual(parent_map["06"], "B")
        self.assertEqual(parent_map["06.1"], "06")
        self.assertEqual(parent_map["10"], "C")

    def test_normalize_revenue_sheet_uses_third_row_header(self) -> None:
        from nonlinear_lab.real_data import normalize_revenue_sheet, reshape_revenue_wide_to_long

        raw = pd.DataFrame(
            [
                [np.nan] * 9,
                [np.nan] * 9,
                [np.nan, "Название", "Адрес", "ИНН", "Субъект РФ", "ОКВЭД", np.nan, "Выручка (нетто) от продажи, тыс руб 2022, год", "Выручка (нетто) от продажи, тыс руб 2021, год"],
                [1, "Компания 1", "Адрес 1", "123", "Москва", "10.10", "Пищевые продукты", 1000, 900],
                [2, "Медиана", np.nan, np.nan, np.nan, np.nan, np.nan, 500, 450],
            ]
        )

        wide = normalize_revenue_sheet(raw)

        self.assertIn("company_name", wide.columns)
        self.assertIn("okved_old", wide.columns)
        self.assertIn("revenue_2022", wide.columns)
        self.assertTrue(wide["is_summary_row"].iloc[1])

        long_df = reshape_revenue_wide_to_long(wide)
        self.assertEqual(set(long_df["year"]), {2021, 2022})
        self.assertEqual(long_df["company_name"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
