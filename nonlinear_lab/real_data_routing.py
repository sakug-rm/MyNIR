from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nonlinear_lab.features import growth_rate, make_regression_df
from nonlinear_lab.plan_g_experiment import extract_window_diagnostics, predict_regime_label
from nonlinear_lab.plan_h_experiment import compute_condition_number
from nonlinear_lab.plotting import save_figure


REGIME_RENAMES = {
    "growth_no_memory": "growth_no_memory",
    "growth_with_memory": "memory_like_growth",
    "stable_cycle": "oscillatory_informative",
    "chaotic_informative": "turbulent_informative",
    "plateau_degenerate": "plateau_degenerate",
    "collapse": "collapse",
}

TOOL_BY_REGIME = {
    "growth_no_memory": "direct_probe + enter",
    "memory_like_growth": "structural_regression",
    "oscillatory_informative": "beta_bsum_then_stepwise",
    "turbulent_informative": "phase_short_forecast",
    "plateau_degenerate": "do_not_read_regression",
    "collapse": "degradation_only",
}


def _mean_abs_corr(X: pd.DataFrame) -> float:
    if X.shape[1] < 2:
        return 0.0
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    values = upper.stack().to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    return float(np.nanmean(values))


def _window_feature_rows(
    series_values: np.ndarray,
    *,
    window: int,
    lags: int,
    zero_guard: float = 1e-12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(series_values) - window + 1):
        end = start + window
        block = np.asarray(series_values[start:end], dtype=float)
        diagnostics = extract_window_diagnostics(block, zero_guard=zero_guard)
        reg_df = make_regression_df(block, lags=lags, zero_guard=zero_guard)
        omega = growth_rate(block, zero_guard=zero_guard)
        if len(reg_df) >= 8:
            X = reg_df.drop(columns=["omega"])
            cond_scaled = compute_condition_number(X, standardize=True)
            mean_abs_pair_corr = _mean_abs_corr(X)
        else:
            cond_scaled = float("nan")
            mean_abs_pair_corr = float("nan")

        predicted = REGIME_RENAMES[predict_regime_label(diagnostics)]
        rows.append(
            {
                "start": start,
                "end": end,
                "var_omega": float(np.var(omega, ddof=0)) if len(omega) else float("nan"),
                "cond_scaled": cond_scaled,
                "mean_abs_pair_corr": mean_abs_pair_corr,
                "regime_label": predicted,
                "preferred_tool": TOOL_BY_REGIME[predicted],
                **diagnostics,
            }
        )
    return rows


def _assign_interpretability(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    var_threshold = float(out["var_omega"].quantile(0.10))
    cond_threshold = float(out["cond_scaled"].replace([np.inf], np.nan).dropna().quantile(0.90))
    corr_threshold = float(out["mean_abs_pair_corr"].dropna().quantile(0.90))

    labels = []
    for _, row in out.iterrows():
        if row["regime_label"] in {"plateau_degenerate", "collapse"}:
            labels.append(row["regime_label"])
        elif np.isfinite(row["var_omega"]) and row["var_omega"] <= var_threshold:
            labels.append("low_dispersion")
        elif (
            np.isfinite(row["cond_scaled"])
            and row["cond_scaled"] >= cond_threshold
        ) or (
            np.isfinite(row["mean_abs_pair_corr"])
            and row["mean_abs_pair_corr"] >= corr_threshold
        ):
            labels.append("collinearity_heavy")
        else:
            labels.append("interpretable")

    out["interpretability_label"] = labels
    out["var_q10_threshold"] = var_threshold
    out["cond_q90_threshold"] = cond_threshold
    out["corr_q90_threshold"] = corr_threshold
    out["preferred_tool"] = np.where(
        out["interpretability_label"].isin(["low_dispersion", "plateau_degenerate", "collapse"]),
        "do_not_read_regression",
        out["preferred_tool"],
    )
    return out


def run_ipp_routing_experiment(
    ipp_long: pd.DataFrame,
    *,
    windows: list[int] | tuple[int, ...] = (24,),
    lags: int = 3,
    variant_filter: list[str] | tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    required = {"date", "series_code", "series_name", "variant", "index_value"}
    missing = required - set(ipp_long.columns)
    if missing:
        raise ValueError(f"IPP long data is missing columns: {sorted(missing)}")

    frame = ipp_long.copy()
    if variant_filter is not None:
        frame = frame[frame["variant"].isin(list(variant_filter))].copy()

    window_rows: list[dict[str, Any]] = []
    grouped = frame.sort_values("date").groupby(["series_code", "series_name", "variant"], sort=True)
    for (series_code, series_name, variant), group in grouped:
        values = group["index_value"].to_numpy(dtype=float)
        for window in windows:
            if window >= len(values):
                continue
            for row in _window_feature_rows(values, window=window, lags=lags):
                row.update(
                    {
                        "series_code": series_code,
                        "series_name": series_name,
                        "variant": variant,
                        "window": window,
                        "lags": lags,
                    }
                )
                window_rows.append(row)

    window_features = pd.DataFrame(window_rows)
    if window_features.empty:
        return {
            "window_features": window_features,
            "regime_summary": pd.DataFrame(),
            "interpretability_summary": pd.DataFrame(),
            "routing_summary": pd.DataFrame(),
        }

    grouped = window_features.groupby(["series_code", "series_name", "variant", "window"], sort=False)
    window_features = pd.concat([_assign_interpretability(group) for _, group in grouped], ignore_index=True)

    regime_summary = (
        window_features.groupby(["variant", "window", "regime_label"], as_index=False)
        .agg(window_share=("regime_label", "size"))
    )
    regime_summary["window_share"] = regime_summary.groupby(["variant", "window"])["window_share"].transform(
        lambda s: s / s.sum()
    )

    interpretability_summary = (
        window_features.groupby(["variant", "window", "interpretability_label"], as_index=False)
        .agg(window_share=("interpretability_label", "size"))
    )
    interpretability_summary["window_share"] = interpretability_summary.groupby(["variant", "window"])[
        "window_share"
    ].transform(lambda s: s / s.sum())

    routing_summary = (
        window_features.groupby(["variant", "window", "preferred_tool"], as_index=False)
        .agg(window_share=("preferred_tool", "size"))
    )
    routing_summary["window_share"] = routing_summary.groupby(["variant", "window"])["window_share"].transform(
        lambda s: s / s.sum()
    )

    return {
        "window_features": window_features,
        "regime_summary": regime_summary,
        "interpretability_summary": interpretability_summary,
        "routing_summary": routing_summary,
    }


def _plot_stacked_share(summary: pd.DataFrame, label_col: str, title: str):
    variants = list(summary["variant"].drop_duplicates())
    labels = list(summary[label_col].drop_duplicates())
    table = (
        summary.pivot_table(index="variant", columns=label_col, values="window_share", fill_value=0.0)
        .reindex(index=variants, columns=labels, fill_value=0.0)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(table))
    for column in table.columns:
        values = table[column].to_numpy(dtype=float)
        ax.bar(table.index, values, bottom=bottom, label=column)
        bottom += values
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Window Share")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_var_cond_scatter(window_features: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = list(window_features["interpretability_label"].drop_duplicates())
    for category in categories:
        subset = window_features[window_features["interpretability_label"] == category]
        ax.scatter(
            subset["var_omega"],
            subset["cond_scaled"].clip(upper=1e6),
            s=18,
            alpha=0.65,
            label=category,
        )
    ax.set_xlabel("Var(omega)")
    ax.set_ylabel("cond(X_scaled) clipped")
    ax.set_title("Window Interpretability Map")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_ipp_routing_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)

    if not results["regime_summary"].empty:
        save_figure(
            _plot_stacked_share(results["regime_summary"], "regime_label", "Regime Shares by Variant"),
            out / "regime_shares.png",
        )
    if not results["interpretability_summary"].empty:
        save_figure(
            _plot_stacked_share(
                results["interpretability_summary"],
                "interpretability_label",
                "Interpretability Shares by Variant",
            ),
            out / "interpretability_shares.png",
        )
    if not results["window_features"].empty:
        save_figure(_plot_var_cond_scatter(results["window_features"]), out / "var_cond_scatter.png")

    summary = {
        "window_count": int(len(results["window_features"])),
        "series_count": int(results["window_features"]["series_code"].nunique()) if not results["window_features"].empty else 0,
        "variant_count": int(results["window_features"]["variant"].nunique()) if not results["window_features"].empty else 0,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
