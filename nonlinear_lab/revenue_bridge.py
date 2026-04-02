from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nonlinear_lab.plotting import save_figure
from nonlinear_lab.revenue_b1 import run_b1_interval_models, run_b1_short_windows, summarize_b1_shocks


DEFAULT_BRIDGE_MAP = (
    {
        "bridge_id": "food",
        "revenue_sector_2d": "15",
        "revenue_label_hint": "food_old_okved15",
        "ipp_series_code": "10",
        "ipp_label_hint": "food_ipp10",
        "mapping_quality": "high",
        "note": "Food manufacturing is the cleanest old-to-IPP match.",
    },
    {
        "bridge_id": "pharma",
        "revenue_sector_2d": "24",
        "revenue_label_hint": "pharma_old_okved24",
        "ipp_series_code": "21",
        "ipp_label_hint": "pharma_ipp21",
        "mapping_quality": "medium",
        "note": "Old sector 24 mixes chemistry and pharma; use as partial pharmaceutical proxy.",
    },
    {
        "bridge_id": "energy_heat",
        "revenue_sector_2d": "40",
        "revenue_label_hint": "energy_old_okved40",
        "ipp_series_code": "35.1+35.3 Производство, передача и",
        "ipp_label_hint": "energy_ipp35.1+35.3",
        "mapping_quality": "high",
        "note": "Electricity/heat sector is the strongest energy bridge.",
    },
    {
        "bridge_id": "water",
        "revenue_sector_2d": "41",
        "revenue_label_hint": "water_old_okved41",
        "ipp_series_code": "36",
        "ipp_label_hint": "water_ipp36",
        "mapping_quality": "medium",
        "note": "Water distribution is a close but not perfect match.",
    },
)


def build_bridge_mapping(bridge_map: tuple[dict[str, str], ...] | list[dict[str, str]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(list(DEFAULT_BRIDGE_MAP if bridge_map is None else bridge_map))


def build_ipp_annual_series(
    ipp_long: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    variant: str = "adj_unsmoothed",
) -> pd.DataFrame:
    frame = ipp_long.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    frame = frame[frame["variant"] == variant].copy()
    frame = frame.rename(columns={"series_code": "ipp_series_code", "series_name": "ipp_series_name"})
    frame["ipp_series_code"] = frame["ipp_series_code"].astype(str)
    mapping = mapping.copy()
    mapping["ipp_series_code"] = mapping["ipp_series_code"].astype(str)
    merged = frame.merge(mapping, on="ipp_series_code", how="inner")
    annual = (
        merged.groupby(
            ["bridge_id", "ipp_series_code", "ipp_series_name", "mapping_quality", "note", "year"],
            as_index=False,
        )
        .agg(
            annual_mean=("index_value", "mean"),
            december_level=("index_value", lambda s: s.iloc[-1]),
        )
        .sort_values(["bridge_id", "year"])
        .reset_index(drop=True)
    )
    annual["b1_group"] = annual["bridge_id"]
    annual["group_label"] = annual["bridge_id"] + " — IPP"
    annual["median_revenue"] = annual["annual_mean"]
    annual["firms"] = 1
    annual["sector_label"] = annual["ipp_series_name"]
    return annual


def build_revenue_sector_series(
    b2_sector_yearly: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    frame = b2_sector_yearly.copy()
    frame = frame.rename(columns={"sector_2d_code": "revenue_sector_2d", "sector_label": "revenue_sector_label"})
    frame["revenue_sector_2d"] = frame["revenue_sector_2d"].astype(str)
    mapping = mapping.copy()
    mapping["revenue_sector_2d"] = mapping["revenue_sector_2d"].astype(str)
    merged = frame.merge(mapping, on="revenue_sector_2d", how="inner")
    annual = merged[
        [
            "bridge_id",
            "revenue_sector_2d",
            "revenue_sector_label",
            "mapping_quality",
            "note",
            "year",
            "median_revenue",
            "firms",
        ]
    ].copy()
    annual["b1_group"] = annual["bridge_id"]
    annual["group_label"] = annual["bridge_id"] + " — revenue"
    annual["sector_label"] = annual["revenue_sector_label"]
    return annual


def run_bridge_case(
    ipp_long: pd.DataFrame,
    b2_sector_yearly: pd.DataFrame,
    *,
    bridge_map: tuple[dict[str, str], ...] | list[dict[str, str]] | None = None,
    ipp_variant: str = "adj_unsmoothed",
) -> dict[str, pd.DataFrame]:
    mapping = build_bridge_mapping(bridge_map)
    ipp_annual = build_ipp_annual_series(ipp_long, mapping, variant=ipp_variant)
    revenue_annual = build_revenue_sector_series(b2_sector_yearly, mapping)

    ipp_interval_models, ipp_interval_summary = run_b1_interval_models(ipp_annual)
    ipp_window_level, ipp_window_summary = run_b1_short_windows(ipp_annual)
    ipp_shocks = summarize_b1_shocks(ipp_annual)

    rev_interval_models, rev_interval_summary = run_b1_interval_models(revenue_annual)
    rev_window_level, rev_window_summary = run_b1_short_windows(revenue_annual)
    rev_shocks = summarize_b1_shocks(revenue_annual)

    ipp_interval_summary = ipp_interval_summary.rename(
        columns={"b1_group": "bridge_id", "group_label": "source_label"}
    ).assign(source="ipp")
    rev_interval_summary = rev_interval_summary.rename(
        columns={"b1_group": "bridge_id", "group_label": "source_label"}
    ).assign(source="revenue")

    ipp_window_summary = ipp_window_summary.rename(
        columns={"b1_group": "bridge_id", "group_label": "source_label"}
    ).assign(source="ipp")
    rev_window_summary = rev_window_summary.rename(
        columns={"b1_group": "bridge_id", "group_label": "source_label"}
    ).assign(source="revenue")

    ipp_shocks = ipp_shocks.rename(columns={"b1_group": "bridge_id", "group_label": "source_label"}).assign(source="ipp")
    rev_shocks = rev_shocks.rename(columns={"b1_group": "bridge_id", "group_label": "source_label"}).assign(source="revenue")

    comparison_rows: list[dict[str, object]] = []
    for bridge_id in mapping["bridge_id"]:
        for interval in ("2014-2019", "2014-2022"):
            left = ipp_interval_summary[(ipp_interval_summary["bridge_id"] == bridge_id) & (ipp_interval_summary["interval"] == interval)]
            right = rev_interval_summary[(rev_interval_summary["bridge_id"] == bridge_id) & (rev_interval_summary["interval"] == interval)]
            if left.empty or right.empty:
                continue
            left_row = left.iloc[0]
            right_row = right.iloc[0]
            comparison_rows.append(
                {
                    "bridge_id": bridge_id,
                    "interval": interval,
                    "mapping_quality": mapping.loc[mapping["bridge_id"] == bridge_id, "mapping_quality"].iloc[0],
                    "ipp_regime": left_row["dominant_regime"],
                    "revenue_regime": right_row["dominant_regime"],
                    "regime_match": int(left_row["dominant_regime"] == right_row["dominant_regime"]),
                    "ipp_interpretability": left_row["dominant_interpretability"],
                    "revenue_interpretability": right_row["dominant_interpretability"],
                    "interpretability_match": int(left_row["dominant_interpretability"] == right_row["dominant_interpretability"]),
                    "ipp_tool": left_row["dominant_tool"],
                    "revenue_tool": right_row["dominant_tool"],
                    "tool_match": int(left_row["dominant_tool"] == right_row["dominant_tool"]),
                    "ipp_best_adj_r2": float(left_row["best_adj_r2"]),
                    "revenue_best_adj_r2": float(right_row["best_adj_r2"]),
                }
            )
    bridge_comparison = pd.DataFrame(comparison_rows)

    shock_rows: list[dict[str, object]] = []
    for bridge_id in mapping["bridge_id"]:
        for shock_year in (2020, 2022):
            left = ipp_shocks[(ipp_shocks["bridge_id"] == bridge_id) & (ipp_shocks["shock_year"] == shock_year)]
            right = rev_shocks[(rev_shocks["bridge_id"] == bridge_id) & (rev_shocks["shock_year"] == shock_year)]
            if left.empty or right.empty:
                continue
            left_row = left.iloc[0]
            right_row = right.iloc[0]
            shock_rows.append(
                {
                    "bridge_id": bridge_id,
                    "shock_year": shock_year,
                    "ipp_change_before": float(left_row["change_before"]),
                    "ipp_change_after": float(left_row["change_after"]),
                    "revenue_change_before": float(right_row["change_before"]),
                    "revenue_change_after": float(right_row["change_after"]),
                    "same_direction_before": int(np.sign(left_row["change_before"]) == np.sign(right_row["change_before"])),
                    "same_direction_after": int(np.sign(left_row["change_after"]) == np.sign(right_row["change_after"])),
                }
            )
    shock_comparison = pd.DataFrame(shock_rows)

    return {
        "bridge_mapping": mapping,
        "ipp_annual": ipp_annual,
        "revenue_annual": revenue_annual,
        "ipp_interval_summary": ipp_interval_summary,
        "revenue_interval_summary": rev_interval_summary,
        "ipp_window_summary": ipp_window_summary,
        "revenue_window_summary": rev_window_summary,
        "bridge_comparison": bridge_comparison,
        "shock_comparison": shock_comparison,
    }


def _plot_bridge_trajectories(ipp_annual: pd.DataFrame, revenue_annual: pd.DataFrame):
    bridge_ids = list(dict.fromkeys(ipp_annual["bridge_id"]))
    fig, axes = plt.subplots(len(bridge_ids), 1, figsize=(10, max(3 * len(bridge_ids), 4)), sharex=True)
    if len(bridge_ids) == 1:
        axes = [axes]
    for ax, bridge_id in zip(axes, bridge_ids):
        ipp = ipp_annual[ipp_annual["bridge_id"] == bridge_id].sort_values("year")
        rev = revenue_annual[revenue_annual["bridge_id"] == bridge_id].sort_values("year")
        ipp_scaled = ipp["median_revenue"] / ipp["median_revenue"].iloc[0]
        rev_scaled = rev["median_revenue"] / rev["median_revenue"].iloc[0]
        ax.plot(ipp["year"], ipp_scaled, marker="o", label="IPP annual mean")
        ax.plot(rev["year"], rev_scaled, marker="s", label="Revenue median")
        for year in (2020, 2022):
            ax.axvline(year, linestyle="--", linewidth=1.0, color="#999999")
        ax.set_title(bridge_id)
        ax.set_ylabel("Scaled level")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    return fig


def _plot_bridge_match(bridge_comparison: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = bridge_comparison["bridge_id"] + "\n" + bridge_comparison["interval"]
    scores = (
        bridge_comparison["regime_match"]
        + bridge_comparison["interpretability_match"]
        + bridge_comparison["tool_match"]
    ) / 3.0
    ax.bar(labels, scores, color="#6baed6")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Bridge Match Score")
    ax.set_title("Bridge-Case Match by Interval")
    fig.tight_layout()
    return fig


def save_bridge_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)

    if not results["ipp_annual"].empty and not results["revenue_annual"].empty:
        save_figure(_plot_bridge_trajectories(results["ipp_annual"], results["revenue_annual"]), out / "bridge_trajectories.png")
    if not results["bridge_comparison"].empty:
        save_figure(_plot_bridge_match(results["bridge_comparison"]), out / "bridge_match_score.png")

    summary = {
        "bridge_count": int(results["bridge_mapping"]["bridge_id"].nunique()),
        "comparison_rows": int(len(results["bridge_comparison"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
