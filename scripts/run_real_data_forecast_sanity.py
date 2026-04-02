from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def build_forecast_sanity(output_dir: Path) -> None:
    windows = pd.read_csv("outputs/real_data/ipp_routing_adj_unsmoothed/window_features.csv")
    series = pd.read_csv("real_data/processed/ipp_long.csv")
    series = series[series["variant"] == "adj_unsmoothed"].copy()
    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values(["series_code", "date"])
    series_map = {
        code: frame["index_value"].to_numpy() for code, frame in series.groupby("series_code")
    }

    rows = []
    for row in windows.itertuples(index=False):
        values = series_map.get(row.series_code)
        if values is None or row.end >= len(values) - 1 or row.start < 1:
            continue

        start = int(row.start)
        end = int(row.end)
        window = values[start : end + 1]
        if len(window) < 5:
            continue

        y = (window[2:] - window[1:-1]) / window[1:-1]
        x_current = window[1:-1]
        x_lag1 = window[:-2]
        design = sm.add_constant(pd.DataFrame({"x": x_current, "lag1": x_lag1}), has_constant="add")

        try:
            fitted = sm.OLS(y, design).fit()
            omega_hat = fitted.predict([1.0, window[-1], window[-2]])[0]
            pred_model = window[-1] * (1.0 + omega_hat)
        except Exception:
            continue

        actual_next = values[end + 1]
        pred_naive = window[-1]
        ae_model = abs(actual_next - pred_model)
        ae_naive = abs(actual_next - pred_naive)

        rows.append(
            {
                "series_code": row.series_code,
                "start": start,
                "end": end,
                "preferred_tool": row.preferred_tool,
                "interpretability_label": row.interpretability_label,
                "actual_next": actual_next,
                "pred_model": pred_model,
                "pred_naive": pred_naive,
                "ae_model": ae_model,
                "ae_naive": ae_naive,
                "improves": float(ae_model < ae_naive),
            }
        )

    window_level = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_level.to_csv(output_dir / "window_level.csv", index=False)

    by_interpretability = (
        window_level.groupby("interpretability_label")[["ae_model", "ae_naive", "improves"]]
        .mean()
        .reset_index()
    )
    by_interpretability["grouping"] = "interpretability"
    by_interpretability = by_interpretability.rename(columns={"interpretability_label": "label"})

    by_tool = (
        window_level.groupby("preferred_tool")[["ae_model", "ae_naive", "improves"]]
        .mean()
        .reset_index()
    )
    by_tool["grouping"] = "preferred_tool"
    by_tool = by_tool.rename(columns={"preferred_tool": "label"})

    summary = pd.concat([by_interpretability, by_tool], ignore_index=True)
    summary.to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/real_data/forecast_sanity",
    )
    args = parser.parse_args()
    build_forecast_sanity(Path(args.output_dir))


if __name__ == "__main__":
    main()
