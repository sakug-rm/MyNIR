from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nonlinear_lab.plotting import save_figure


def _agreement_share(values: pd.Series) -> float:
    usable = values.dropna().astype(str)
    if usable.empty:
        return float("nan")
    return float(usable.value_counts(normalize=True).max())


def run_ipp_consistency_analysis(
    routing_windows: pd.DataFrame,
    structural_windows: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    routing_cols = ["series_code", "variant", "window", "regime_label", "interpretability_label", "preferred_tool"]
    structural_cols = ["series_code", "variant", "window", "reading_mode", "top_beta_predictor", "top_b_predictor"]

    routing = routing_windows[routing_cols].copy()
    structural = structural_windows[structural_cols].copy()
    merged = routing.merge(structural, on=["series_code", "variant", "window"], how="left")

    series_consistency = (
        merged.groupby(["series_code", "window"], as_index=False)
        .agg(
            variants=("variant", "nunique"),
            regime_consistency=("regime_label", _agreement_share),
            interpretability_consistency=("interpretability_label", _agreement_share),
            tool_consistency=("preferred_tool", _agreement_share),
            top_beta_consistency=("top_beta_predictor", _agreement_share),
            top_b_consistency=("top_b_predictor", _agreement_share),
            reading_mode_consistency=("reading_mode", _agreement_share),
        )
        .sort_values(["window", "series_code"])
        .reset_index(drop=True)
    )

    variant_consistency = (
        merged.groupby(["variant", "window"], as_index=False)
        .agg(
            regime_consistency=("regime_label", _agreement_share),
            interpretability_consistency=("interpretability_label", _agreement_share),
            tool_consistency=("preferred_tool", _agreement_share),
            top_beta_consistency=("top_beta_predictor", _agreement_share),
            top_b_consistency=("top_b_predictor", _agreement_share),
        )
        .sort_values(["window", "variant"])
        .reset_index(drop=True)
    )

    overall_summary = pd.DataFrame(
        [
            {
                "scope": "all_series",
                "mean_regime_consistency": float(series_consistency["regime_consistency"].mean()),
                "mean_interpretability_consistency": float(series_consistency["interpretability_consistency"].mean()),
                "mean_tool_consistency": float(series_consistency["tool_consistency"].mean()),
                "mean_top_beta_consistency": float(series_consistency["top_beta_consistency"].mean()),
                "mean_top_b_consistency": float(series_consistency["top_b_consistency"].mean()),
                "mean_reading_mode_consistency": float(series_consistency["reading_mode_consistency"].mean()),
            }
        ]
    )

    return {
        "series_consistency": series_consistency,
        "variant_consistency": variant_consistency,
        "overall_summary": overall_summary,
    }


def _plot_variant_consistency(variant_consistency: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    for column in [
        "regime_consistency",
        "interpretability_consistency",
        "tool_consistency",
        "top_beta_consistency",
    ]:
        ax.plot(variant_consistency["variant"], variant_consistency[column], marker="o", label=column)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Agreement Share")
    ax.set_title("Variant-Level Consistency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_ipp_consistency_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)
    if not results["variant_consistency"].empty:
        save_figure(_plot_variant_consistency(results["variant_consistency"]), out / "variant_consistency.png")
    summary = {
        "series_count": int(results["series_consistency"]["series_code"].nunique()) if not results["series_consistency"].empty else 0,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
