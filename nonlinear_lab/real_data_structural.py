from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.plotting import save_figure
from nonlinear_lab.regression import fit_enter_with_beta, stepwise_selection


def _structural_mode(regime_label: str, interpretability_label: str) -> str:
    if interpretability_label in {"plateau_degenerate", "collapse", "low_dispersion"}:
        return "do_not_read"
    if regime_label == "growth_no_memory":
        return "current_state"
    if regime_label == "memory_like_growth":
        return "memory_structure"
    if interpretability_label == "collinearity_heavy":
        return "beta_bsum_caution"
    if regime_label == "oscillatory_informative":
        return "beta_bsum_then_stepwise"
    if regime_label == "turbulent_informative":
        return "phase_caution"
    return "beta_bsum_then_stepwise"


def _dominant_name(values: pd.Series) -> str:
    usable = values.dropna()
    if usable.empty:
        return ""
    return str(usable.value_counts().idxmax())


def _fit_window_structure(block: np.ndarray, lags: int, threshold_in: float, threshold_out: float) -> dict[str, Any]:
    reg_df = make_regression_df(block, lags=lags)
    if len(reg_df) < 8:
        return {
            "R2_enter": float("nan"),
            "top_beta_predictor": "",
            "top_b_predictor": "",
            "b_sum": float("nan"),
            "selected": [],
            "selected_count": 0.0,
            "beta_x": float("nan"),
            "beta_lag1": float("nan"),
            "b_x": float("nan"),
            "b_lag1": float("nan"),
        }

    y = reg_df["omega"]
    X = reg_df.drop(columns=["omega"])
    model, beta = fit_enter_with_beta(X, y)
    selected = stepwise_selection(X, y, threshold_in=threshold_in, threshold_out=threshold_out)

    params = model.params.drop(labels="const", errors="ignore")
    abs_b = params.abs().sort_values(ascending=False)
    abs_beta = beta.abs().sort_values(ascending=False)
    lag_cols = [column for column in X.columns if column.startswith("Lag_")]
    b_sum = float(sum(float(params.get(column, 0.0)) for column in lag_cols))

    return {
        "R2_enter": float(model.rsquared),
        "top_beta_predictor": str(abs_beta.index[0]) if not abs_beta.empty else "",
        "top_b_predictor": str(abs_b.index[0]) if not abs_b.empty else "",
        "b_sum": b_sum,
        "selected": selected,
        "selected_count": float(len(selected)),
        "beta_x": float(beta.get("X_n", np.nan)),
        "beta_lag1": float(beta.get("Lag_1", np.nan)),
        "b_x": float(params.get("X_n", np.nan)),
        "b_lag1": float(params.get("Lag_1", np.nan)),
    }


def run_ipp_structural_reading(
    ipp_long: pd.DataFrame,
    window_features: pd.DataFrame,
    *,
    lags: int = 3,
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
) -> dict[str, pd.DataFrame]:
    data = ipp_long.sort_values("date")
    rows: list[dict[str, Any]] = []

    for _, window_row in window_features.iterrows():
        series = data[
            (data["series_code"] == window_row["series_code"])
            & (data["variant"] == window_row["variant"])
        ]["index_value"].to_numpy(dtype=float)
        start = int(window_row["start"])
        end = int(window_row["end"])
        block = np.asarray(series[start:end], dtype=float)
        reading_mode = _structural_mode(window_row["regime_label"], window_row["interpretability_label"])
        metrics = _fit_window_structure(block, lags=lags, threshold_in=threshold_in, threshold_out=threshold_out)
        rows.append(
            {
                "series_code": window_row["series_code"],
                "series_name": window_row["series_name"],
                "variant": window_row["variant"],
                "window": window_row["window"],
                "start": start,
                "end": end,
                "regime_label": window_row["regime_label"],
                "interpretability_label": window_row["interpretability_label"],
                "reading_mode": reading_mode,
                **metrics,
            }
        )

    window_structures = pd.DataFrame(rows)
    if window_structures.empty:
        return {
            "window_structures": window_structures,
            "series_summary": pd.DataFrame(),
            "mode_summary": pd.DataFrame(),
        }

    series_summary = (
        window_structures.assign(
            is_interpretable=(window_structures["interpretability_label"] == "interpretable").astype(int),
            is_memory=(window_structures["reading_mode"] == "memory_structure").astype(int),
        )
        .groupby(["series_code", "series_name", "variant"], as_index=False)
        .agg(
            windows=("series_code", "size"),
            interpretable_share=("is_interpretable", "mean"),
            mean_r2=("R2_enter", "mean"),
            mean_b_sum=("b_sum", "mean"),
            mean_selected_count=("selected_count", "mean"),
            dominant_mode=("reading_mode", _dominant_name),
            dominant_top_beta=("top_beta_predictor", _dominant_name),
            dominant_top_b=("top_b_predictor", _dominant_name),
        )
        .sort_values(["interpretable_share", "mean_r2"], ascending=[False, False])
        .reset_index(drop=True)
    )

    mode_summary = (
        window_structures.groupby(["variant", "reading_mode"], as_index=False)
        .agg(window_share=("reading_mode", "size"))
        .sort_values(["variant", "reading_mode"])
        .reset_index(drop=True)
    )
    mode_summary["window_share"] = mode_summary.groupby("variant")["window_share"].transform(lambda s: s / s.sum())

    return {
        "window_structures": window_structures,
        "series_summary": series_summary,
        "mode_summary": mode_summary,
    }


def _plot_mode_shares(mode_summary: pd.DataFrame):
    table = mode_summary.pivot_table(index="variant", columns="reading_mode", values="window_share", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(table))
    for column in table.columns:
        values = table[column].to_numpy(dtype=float)
        ax.bar(table.index, values, bottom=bottom, label=column)
        bottom += values
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Window Share")
    ax.set_title("Structural Reading Modes by Variant")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_top_interpretable(series_summary: pd.DataFrame, top_n: int = 10):
    top = series_summary.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top["series_code"].astype(str), top["interpretable_share"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Interpretable Share")
    ax.set_title("Top Series by Interpretable Share")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def save_ipp_structural_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)

    if not results["mode_summary"].empty:
        save_figure(_plot_mode_shares(results["mode_summary"]), out / "mode_shares.png")
    if not results["series_summary"].empty:
        save_figure(_plot_top_interpretable(results["series_summary"]), out / "top_interpretable_series.png")

    summary = {
        "window_count": int(len(results["window_structures"])),
        "series_count": int(results["series_summary"]["series_code"].nunique()) if not results["series_summary"].empty else 0,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
