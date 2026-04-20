from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nonlinear_lab.plotting import save_figure
from nonlinear_lab.revenue_b1 import (
    _fit_spec,
    _interpretability_label,
    _interval_regime,
    _tool_label,
    run_b1_interval_models,
    run_b1_short_windows,
    summarize_b1_shocks,
    summarize_b1_sign_stability,
)


def normalize_old_okved(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().replace(" ", "").replace(",", ".")
    return text


def extract_okved_levels(code: str) -> dict[str, str]:
    normalized = normalize_old_okved(code)
    digits = "".join(ch for ch in normalized if ch.isdigit() or ch == ".")
    sector_2d = digits[:2] if len(digits) >= 2 else ""
    sector_3d = ""
    if "." in digits:
        left, right = digits.split(".", 1)
        if len(left) >= 2 and len(right) >= 1:
            sector_3d = f"{left[:2]}.{right[:1]}"
    elif len(digits) >= 3:
        sector_3d = digits[:3]
    sector_4d = ""
    if "." in digits:
        left, right = digits.split(".", 1)
        if len(left) >= 2 and len(right) >= 2:
            sector_4d = f"{left[:2]}.{right[:2]}"
    elif len(digits) >= 4:
        sector_4d = digits[:4]
    return {
        "okved_norm": digits,
        "sector_2d": sector_2d,
        "sector_3d": sector_3d,
        "sector_4d": sector_4d,
    }


def build_b2_sector_corpus(
    revenue_wide: pd.DataFrame,
    revenue_long: pd.DataFrame,
    *,
    min_firms: int = 100,
    focus_sectors: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    wide = revenue_wide.copy()
    wide = wide[~wide.get("is_summary_row", False)].copy()
    levels = wide["okved_old"].apply(extract_okved_levels).apply(pd.Series)
    wide = pd.concat([wide.reset_index(drop=True), levels.reset_index(drop=True)], axis=1)
    wide["mapping_layer"] = np.where(wide["sector_2d"] != "", "exact_or_normalized", "unmapped")
    wide["included_candidate"] = wide["mapping_layer"] == "exact_or_normalized"

    sector_summary = (
        wide[wide["sector_2d"] != ""]
        .groupby("sector_2d", as_index=False)
        .agg(
            firms=("inn", "nunique"),
            median_2010=("revenue_2010", "median"),
            median_2014=("revenue_2014", "median"),
            median_2022=("revenue_2022", "median"),
            sector_name=("industry_text", lambda s: s.dropna().astype(str).value_counts().idxmax() if not s.dropna().empty else ""),
        )
        .sort_values("firms", ascending=False)
        .reset_index(drop=True)
    )
    sector_summary["included"] = sector_summary["firms"] >= min_firms
    if focus_sectors is not None:
        focus_set = set(focus_sectors)
        sector_summary["focus_sector"] = sector_summary["sector_2d"].isin(focus_set)
    else:
        sector_summary["focus_sector"] = False

    included_sectors = sector_summary[sector_summary["included"]]["sector_2d"].tolist()
    long = revenue_long.copy()
    long = long[~long.get("is_summary_row", False)].copy()
    long_levels = long["okved_old"].apply(extract_okved_levels).apply(pd.Series)
    long = pd.concat([long.reset_index(drop=True), long_levels.reset_index(drop=True)], axis=1)
    long = long[(long["sector_2d"].isin(included_sectors))].copy()
    long["revenue"] = pd.to_numeric(long["revenue"], errors="coerce")
    long = long[long["revenue"] > 0].copy()
    long["mapping_layer"] = "exact_or_normalized"

    coverage = {
        "total_firms": int(revenue_wide.loc[~revenue_wide.get("is_summary_row", False), "inn"].nunique()),
        "firms_with_okved": int(wide[wide["sector_2d"] != ""]["inn"].nunique()),
        "included_firms": int(wide[wide["sector_2d"].isin(included_sectors)]["inn"].nunique()),
        "included_sectors": int(len(included_sectors)),
    }
    coverage_df = pd.DataFrame([coverage])
    return {
        "sector_firms": wide,
        "sector_summary": sector_summary,
        "sector_long": long,
        "coverage_summary": coverage_df,
    }


def compute_b2_sector_medians(sector_long: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sector_long.groupby(["sector_2d", "year"], as_index=False)
        .agg(
            firms=("inn", "nunique"),
            median_revenue=("revenue", "median"),
            mean_revenue=("revenue", "mean"),
            sector_name=("industry_text", lambda s: s.dropna().astype(str).value_counts().idxmax() if not s.dropna().empty else ""),
        )
        .sort_values(["sector_2d", "year"])
        .reset_index(drop=True)
    )
    grouped["b1_group"] = grouped["sector_2d"]
    grouped["group_label"] = grouped["sector_2d"] + " — " + grouped["sector_name"].astype(str).str.slice(0, 48)
    return grouped


def run_b2_experiment(
    revenue_wide: pd.DataFrame,
    revenue_long: pd.DataFrame,
    *,
    min_firms: int = 100,
    focus_sectors: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    corpus = build_b2_sector_corpus(revenue_wide, revenue_long, min_firms=min_firms, focus_sectors=focus_sectors)
    sector_yearly = compute_b2_sector_medians(corpus["sector_long"])
    interval_models, interval_summary = run_b1_interval_models(sector_yearly)
    window_level, window_summary = run_b1_short_windows(sector_yearly)
    sign_stability = summarize_b1_sign_stability(interval_models)
    shock_summary = summarize_b1_shocks(sector_yearly)

    interval_summary = interval_summary.rename(columns={"b1_group": "sector_2d", "group_label": "sector_label"})
    interval_models = interval_models.rename(columns={"b1_group": "sector_2d", "group_label": "sector_label"})
    window_level = window_level.rename(columns={"b1_group": "sector_2d", "group_label": "sector_label"})
    window_summary = window_summary.rename(columns={"b1_group": "sector_2d", "group_label": "sector_label"})
    sign_stability = sign_stability.rename(columns={"b1_group": "sector_2d", "group_label": "sector_label"})
    shock_summary = shock_summary.rename(columns={"b1_group": "sector_2d", "group_label": "sector_label"})
    sector_yearly = sector_yearly.rename(columns={"b1_group": "sector_2d_code", "group_label": "sector_label"})

    return {
        "sector_firms": corpus["sector_firms"],
        "sector_summary": corpus["sector_summary"],
        "coverage_summary": corpus["coverage_summary"],
        "sector_yearly": sector_yearly,
        "interval_models": interval_models,
        "interval_summary": interval_summary,
        "window_level": window_level,
        "window_summary": window_summary,
        "sign_stability": sign_stability,
        "shock_summary": shock_summary,
    }


def _plot_sector_trajectories(sector_yearly: pd.DataFrame, top_n: int = 8):
    sector_order = (
        sector_yearly.groupby(["sector_2d_code", "sector_label"], as_index=False)["firms"]
        .max()
        .sort_values("firms", ascending=False)
        .head(top_n)["sector_2d_code"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for code in sector_order:
        subset = sector_yearly[sector_yearly["sector_2d_code"] == code].sort_values("year")
        ax.plot(subset["year"], subset["median_revenue"], marker="o", label=code)
    for year in (2014, 2020, 2022):
        ax.axvline(year, linestyle="--", linewidth=1.0, color="#999999")
    ax.set_title("Медианные траектории отраслевой выручки")
    ax.set_ylabel("Медианная выручка")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def _plot_interpretable_share(window_summary: pd.DataFrame):
    pivot = window_summary.pivot_table(index="sector_2d", columns="window", values="interpretable_share")
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(pivot))
    width = 0.35
    windows = list(pivot.columns)
    for idx, window in enumerate(windows):
        ax.bar(positions + idx * width, pivot[window].to_numpy(dtype=float), width=width, label=f"окно {window} лет")
    ax.set_xticks(positions + width * max(len(windows) - 1, 0) / 2)
    ax.set_xticklabels(pivot.index.astype(str), rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Доля интерпретируемых окон")
    ax.set_title("Доля интерпретируемых окон по секторам")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_interval_best_fit(interval_summary: pd.DataFrame):
    table = interval_summary.pivot_table(index="sector_2d", columns="interval", values="best_adj_r2")
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(table))
    width = 0.35
    intervals = list(table.columns)
    for idx, interval in enumerate(intervals):
        ax.bar(positions + idx * width, table[interval].to_numpy(dtype=float), width=width, label=interval)
    ax.set_xticks(positions + width * max(len(intervals) - 1, 0) / 2)
    ax.set_xticklabels(table.index.astype(str), rotation=45, ha="right")
    ax.set_ylabel("Лучший скорректированный R²")
    ax.set_title("Качество подгонки на полном интервале по секторам")
    ax.legend()
    fig.tight_layout()
    return fig


def save_b2_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)

    if not results["sector_yearly"].empty:
        save_figure(_plot_sector_trajectories(results["sector_yearly"]), out / "sector_trajectories.png")
    if not results["window_summary"].empty:
        save_figure(_plot_interpretable_share(results["window_summary"]), out / "sector_interpretable_share.png")
    if not results["interval_summary"].empty:
        save_figure(_plot_interval_best_fit(results["interval_summary"]), out / "sector_interval_best_fit.png")

    summary = {
        "included_sector_count": int(results["sector_summary"]["included"].sum()) if not results["sector_summary"].empty else 0,
        "interval_rows": int(len(results["interval_models"])),
        "window_rows": int(len(results["window_level"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
