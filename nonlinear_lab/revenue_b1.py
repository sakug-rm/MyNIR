from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nonlinear_lab.features import growth_rate, make_regression_df
from nonlinear_lab.plotting import save_figure
from nonlinear_lab.regression import fit_enter_with_beta


def build_b1_groups(
    revenue_wide: pd.DataFrame,
    *,
    start_year: int = 2010,
    end_year: int = 2014,
    quantile_share: float = 0.10,
) -> pd.DataFrame:
    start_col = f"revenue_{start_year}"
    end_col = f"revenue_{end_year}"
    required = {"inn", "company_name", start_col, end_col}
    missing = required - set(revenue_wide.columns)
    if missing:
        raise ValueError(f"Revenue wide data is missing columns: {sorted(missing)}")

    frame = revenue_wide.copy()
    frame = frame[~frame.get("is_summary_row", False)].copy()
    frame[start_col] = pd.to_numeric(frame[start_col], errors="coerce")
    frame[end_col] = pd.to_numeric(frame[end_col], errors="coerce")
    frame = frame[(frame[start_col] > 0) & (frame[end_col] > 0)].copy()
    if frame.empty:
        raise ValueError("No valid firms available after start/end-year filtering.")

    frame["growth_2010_2014"] = frame[end_col] / frame[start_col] - 1.0
    frame["log_growth_2010_2014"] = np.log(frame[end_col]) - np.log(frame[start_col])
    frame = frame.sort_values(["growth_2010_2014", "inn", "company_name"]).reset_index(drop=True)
    n = len(frame)
    group_size = int(np.floor(n * quantile_share))
    if group_size < 1:
        raise ValueError("Group size is zero; dataset is too small for B1 grouping.")

    frame["growth_rank"] = np.arange(1, n + 1)
    frame["growth_pct"] = (frame["growth_rank"] - 0.5) / n
    frame["b1_group"] = ""

    low_idx = frame.index[:group_size]
    high_idx = frame.index[-group_size:]
    frame.loc[low_idx, "b1_group"] = "low_10_50"
    frame.loc[high_idx, "b1_group"] = "high_90_50"

    remaining = frame[frame["b1_group"] == ""].copy()
    remaining["median_distance"] = (remaining["growth_pct"] - 0.50).abs()
    middle_idx = remaining.sort_values(["median_distance", "growth_rank"]).index[:group_size]
    frame.loc[middle_idx, "b1_group"] = "middle_45_55_50"

    frame["group_label"] = frame["b1_group"].replace(
        {
            "high_90_50": "High growth",
            "low_10_50": "Low growth",
            "middle_45_55_50": "Middle growth",
            "": "unused",
        }
    )
    return frame.reset_index(drop=True)


def compute_b1_group_medians(
    revenue_long: pd.DataFrame,
    group_membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    member_cols = ["inn", "b1_group", "group_label", "growth_2010_2014", "log_growth_2010_2014", "growth_pct"]
    members = group_membership[group_membership["b1_group"] != ""][member_cols].copy()
    frame = revenue_long.copy()
    frame = frame[~frame.get("is_summary_row", False)].copy()
    frame = frame.merge(members, on="inn", how="inner")
    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")
    frame = frame[frame["revenue"] > 0].copy()
    frame["log_revenue"] = np.log(frame["revenue"])

    yearly = (
        frame.groupby(["b1_group", "group_label", "year"], as_index=False)
        .agg(
            median_revenue=("revenue", "median"),
            mean_revenue=("revenue", "mean"),
            median_log_revenue=("log_revenue", "median"),
            firms=("inn", "nunique"),
        )
        .sort_values(["b1_group", "year"])
        .reset_index(drop=True)
    )
    yearly["median_growth"] = yearly.groupby("b1_group")["median_revenue"].transform(
        lambda s: pd.Series(growth_rate(s.to_numpy(dtype=float)), index=s.index[1:])
    )
    yearly["median_growth"] = yearly.groupby("b1_group")["median_growth"].shift(-1)
    yearly["log_diff_median"] = yearly.groupby("b1_group")["median_log_revenue"].diff()

    group_summary = (
        yearly[yearly["year"].isin([2010, 2014, 2022])]
        .pivot_table(index=["b1_group", "group_label"], columns="year", values="median_revenue")
        .reset_index()
    )
    group_summary.columns = [
        "b1_group" if column == "b1_group" else "group_label" if column == "group_label" else f"median_{column}"
        for column in group_summary.columns
    ]
    group_summary = group_summary.merge(
        members.groupby(["b1_group", "group_label"], as_index=False)
        .agg(
            firms=("inn", "nunique"),
            mean_growth_2010_2014=("growth_2010_2014", "mean"),
            median_growth_2010_2014=("growth_2010_2014", "median"),
            min_growth_pct=("growth_pct", "min"),
            max_growth_pct=("growth_pct", "max"),
        ),
        on=["b1_group", "group_label"],
        how="left",
    )
    group_summary["cumulative_growth_2010_2014"] = (
        group_summary["median_2014"] / group_summary["median_2010"] - 1.0
    )
    group_summary["cumulative_growth_2014_2022"] = (
        group_summary["median_2022"] / group_summary["median_2014"] - 1.0
    )
    return yearly, group_summary.sort_values("b1_group").reset_index(drop=True)


def _interval_regime(block: np.ndarray, omega: np.ndarray) -> str:
    diffs = np.diff(block)
    if len(diffs) == 0:
        return "plateau_like"
    turning_rate = float(np.mean(np.sign(diffs[1:]) != np.sign(diffs[:-1]))) if len(diffs) >= 2 else 0.0
    rel_range = float((np.max(block) - np.min(block)) / max(np.mean(block), 1e-12))
    var_omega = float(np.var(omega, ddof=0)) if len(omega) else 0.0
    median_abs_omega = float(np.median(np.abs(omega))) if len(omega) else 0.0
    max_abs_omega = float(np.max(np.abs(omega))) if len(omega) else 0.0
    net_change = float(block[-1] / block[0] - 1.0) if abs(block[0]) > 1e-12 else 0.0

    if rel_range < 0.08 or var_omega < 0.0025:
        return "plateau_like"
    if block[-1] / max(np.max(block), 1e-12) < 0.72 and net_change < -0.1:
        return "collapse_like"
    if max_abs_omega > max(0.25, 2.5 * max(median_abs_omega, 1e-12)):
        return "shock_transition"
    if turning_rate >= 0.4:
        return "oscillatory"
    if net_change >= 0.08:
        return "monotone_growth"
    if net_change <= -0.08:
        return "monotone_decline"
    return "shock_transition"


def _cond_and_corr(reg_df: pd.DataFrame) -> tuple[float, float]:
    if reg_df.empty:
        return float("nan"), float("nan")
    X = reg_df.drop(columns=["omega"]).to_numpy(dtype=float)
    if X.size == 0:
        return float("nan"), float("nan")
    scaled = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    scaled = scaled / std
    cond = float(np.linalg.cond(scaled))
    if scaled.shape[1] < 2:
        return cond, 0.0
    corr = np.corrcoef(scaled, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return cond, float(np.nanmean(np.abs(upper))) if upper.size else 0.0


def _interpretability_label(regime_label: str, var_omega: float, cond_scaled: float, mean_abs_corr: float) -> str:
    if regime_label in {"plateau_like", "collapse_like"}:
        return "low_dispersion"
    if not np.isfinite(var_omega) or var_omega < 0.0025:
        return "low_dispersion"
    if np.isfinite(cond_scaled) and cond_scaled > 80:
        return "collinearity_heavy"
    if np.isfinite(mean_abs_corr) and mean_abs_corr > 0.92:
        return "collinearity_heavy"
    return "interpretable"


def _tool_label(regime_label: str, interpretability_label: str) -> str:
    if interpretability_label == "low_dispersion":
        return "do_not_read_regression"
    if regime_label == "shock_transition":
        return "phase_trajectory"
    if interpretability_label == "collinearity_heavy":
        return "enter_beta_bsum"
    return "structural_m1_m2"


def _fit_spec(series: np.ndarray, spec: str) -> dict[str, Any]:
    if spec == "M1_current":
        reg_df = make_regression_df(series, lags=0)
        if len(reg_df) < 4:
            return {"usable": False}
        y = reg_df["omega"]
        X = reg_df[["X_n"]]
    elif spec == "M1_lag1":
        reg_df = make_regression_df(series, lags=1)
        if len(reg_df) < 4:
            return {"usable": False}
        y = reg_df["omega"]
        X = reg_df[["Lag_1"]]
    elif spec == "M2_current_lag1":
        reg_df = make_regression_df(series, lags=1)
        if len(reg_df) < 4:
            return {"usable": False}
        y = reg_df["omega"]
        X = reg_df[["X_n", "Lag_1"]]
    elif spec == "M3_lag1_lag2":
        reg_df = make_regression_df(series, lags=2)
        if len(reg_df) < 5:
            return {"usable": False}
        y = reg_df["omega"]
        X = reg_df[["Lag_1", "Lag_2"]]
    else:
        raise ValueError(f"Unknown B1 spec: {spec}")

    model, beta = fit_enter_with_beta(X, y)
    params = model.params.drop(labels="const", errors="ignore")
    top_beta = str(beta.abs().sort_values(ascending=False).index[0]) if not beta.empty else ""
    top_b = str(params.abs().sort_values(ascending=False).index[0]) if not params.empty else ""
    lag_cols = [column for column in X.columns if column.startswith("Lag_")]
    b_sum = float(sum(float(params.get(column, 0.0)) for column in lag_cols))
    cond_scaled, mean_abs_corr = _cond_and_corr(pd.concat([y, X], axis=1))
    return {
        "usable": True,
        "n_obs": int(len(y)),
        "adj_r2": float(model.rsquared_adj),
        "r2": float(model.rsquared),
        "top_beta": top_beta,
        "top_b": top_b,
        "beta_x": float(beta.get("X_n", np.nan)),
        "beta_lag1": float(beta.get("Lag_1", np.nan)),
        "beta_lag2": float(beta.get("Lag_2", np.nan)),
        "b_x": float(params.get("X_n", np.nan)),
        "b_lag1": float(params.get("Lag_1", np.nan)),
        "b_lag2": float(params.get("Lag_2", np.nan)),
        "b_sum": b_sum,
        "cond_scaled": cond_scaled,
        "mean_abs_corr": mean_abs_corr,
        "sign_x": np.sign(params.get("X_n", np.nan)),
        "sign_lag1": np.sign(params.get("Lag_1", np.nan)),
    }


def _leave_one_out_stability(series: np.ndarray, spec: str) -> dict[str, Any]:
    base = _fit_spec(series, spec)
    if not base.get("usable", False):
        return {"loo_runs": 0, "loo_top_beta_consistency": np.nan, "loo_sign_lag1_consistency": np.nan}
    top_hits = 0
    sign_hits = 0
    runs = 0
    for idx in range(len(series)):
        reduced = np.delete(series, idx)
        fitted = _fit_spec(reduced, spec)
        if not fitted.get("usable", False):
            continue
        runs += 1
        if fitted.get("top_beta") == base.get("top_beta"):
            top_hits += 1
        base_sign = base.get("sign_lag1")
        fitted_sign = fitted.get("sign_lag1")
        if np.isfinite(base_sign) and np.isfinite(fitted_sign) and base_sign == fitted_sign:
            sign_hits += 1
    return {
        "loo_runs": runs,
        "loo_top_beta_consistency": float(top_hits / runs) if runs else np.nan,
        "loo_sign_lag1_consistency": float(sign_hits / runs) if runs else np.nan,
    }


def run_b1_interval_models(
    group_yearly: pd.DataFrame,
    *,
    intervals: tuple[tuple[int, int], ...] = ((2014, 2019), (2014, 2022)),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for group_name, group in group_yearly.groupby("b1_group", sort=False):
        label = str(group["group_label"].iloc[0])
        for start_year, end_year in intervals:
            subset = group[(group["year"] >= start_year) & (group["year"] <= end_year)].sort_values("year")
            series = subset["median_revenue"].to_numpy(dtype=float)
            years = subset["year"].tolist()
            if len(series) < 5:
                continue
            omega = growth_rate(series)
            regime_label = _interval_regime(series, omega)
            probe = _fit_spec(series, "M2_current_lag1")
            interpretability_label = _interpretability_label(
                regime_label,
                float(np.var(omega, ddof=0)) if len(omega) else float("nan"),
                float(probe.get("cond_scaled", np.nan)),
                float(probe.get("mean_abs_corr", np.nan)),
            )
            tool_label = _tool_label(regime_label, interpretability_label)
            for spec in ("M1_current", "M1_lag1", "M2_current_lag1", "M3_lag1_lag2"):
                fitted = _fit_spec(series, spec)
                if not fitted.get("usable", False):
                    continue
                loo = _leave_one_out_stability(series, spec)
                rows.append(
                    {
                        "b1_group": group_name,
                        "group_label": label,
                        "interval": f"{start_year}-{end_year}",
                        "start_year": start_year,
                        "end_year": end_year,
                        "n_years": len(series),
                        "years": ",".join(str(year) for year in years),
                        "regime_label": regime_label,
                        "interpretability_label": interpretability_label,
                        "tool_label": tool_label,
                        "spec": spec,
                        **{key: value for key, value in fitted.items() if key != "usable"},
                        **loo,
                    }
                )

    interval_models = pd.DataFrame(rows)
    summary = (
        interval_models.groupby(["b1_group", "group_label", "interval"], as_index=False)
        .agg(
            dominant_regime=("regime_label", lambda s: s.value_counts().idxmax()),
            dominant_interpretability=("interpretability_label", lambda s: s.value_counts().idxmax()),
            dominant_tool=("tool_label", lambda s: s.value_counts().idxmax()),
            best_spec=("adj_r2", lambda s: interval_models.loc[s.index, "spec"].iloc[int(np.nanargmax(s.to_numpy(dtype=float)))]),
            best_adj_r2=("adj_r2", "max"),
            mean_b_sum=("b_sum", "mean"),
            top_beta_mode=("top_beta", lambda s: s.dropna().value_counts().idxmax() if not s.dropna().empty else ""),
            mean_loo_top_beta_consistency=("loo_top_beta_consistency", "mean"),
            mean_loo_sign_lag1_consistency=("loo_sign_lag1_consistency", "mean"),
        )
        .sort_values(["interval", "b1_group"])
        .reset_index(drop=True)
    )
    return interval_models, summary


def run_b1_short_windows(
    group_yearly: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 7),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for group_name, group in group_yearly.groupby("b1_group", sort=False):
        label = str(group["group_label"].iloc[0])
        group = group.sort_values("year").reset_index(drop=True)
        series = group["median_revenue"].to_numpy(dtype=float)
        years = group["year"].to_numpy(dtype=int)
        for window in windows:
            if window > len(series):
                continue
            for start in range(0, len(series) - window + 1):
                end = start + window
                block = series[start:end]
                block_years = years[start:end]
                omega = growth_rate(block)
                regime_label = _interval_regime(block, omega)
                specs = {spec: _fit_spec(block, spec) for spec in ("M1_current", "M1_lag1", "M2_current_lag1")}
                usable_specs = {key: value for key, value in specs.items() if value.get("usable", False)}
                fitted = usable_specs.get("M2_current_lag1")
                if fitted is None:
                    fitted = max(usable_specs.items(), key=lambda item: item[1].get("adj_r2", -np.inf))[1] if usable_specs else {}
                preferred_spec = (
                    max(usable_specs.items(), key=lambda item: item[1].get("adj_r2", -np.inf))[0] if usable_specs else ""
                )
                interpretability_label = _interpretability_label(
                    regime_label,
                    float(np.var(omega, ddof=0)) if len(omega) else float("nan"),
                    float(fitted.get("cond_scaled", np.nan)),
                    float(fitted.get("mean_abs_corr", np.nan)),
                )
                tool_label = _tool_label(regime_label, interpretability_label)
                rows.append(
                    {
                        "b1_group": group_name,
                        "group_label": label,
                        "window": window,
                        "start_year": int(block_years[0]),
                        "end_year": int(block_years[-1]),
                        "regime_label": regime_label,
                        "interpretability_label": interpretability_label,
                        "tool_label": tool_label,
                        "var_omega": float(np.var(omega, ddof=0)) if len(omega) else float("nan"),
                        "rel_range": float((np.max(block) - np.min(block)) / max(np.mean(block), 1e-12)),
                        "preferred_spec": preferred_spec,
                        "adj_r2_best": float(fitted.get("adj_r2", np.nan)),
                        "top_beta_best": str(fitted.get("top_beta", "")),
                        "b_sum_best": float(fitted.get("b_sum", np.nan)),
                        "cond_scaled_best": float(fitted.get("cond_scaled", np.nan)),
                    }
                )

    window_level = pd.DataFrame(rows)
    window_summary = (
        window_level.groupby(["b1_group", "group_label", "window"], as_index=False)
        .agg(
            dominant_regime=("regime_label", lambda s: s.value_counts().idxmax()),
            dominant_interpretability=("interpretability_label", lambda s: s.value_counts().idxmax()),
            dominant_tool=("tool_label", lambda s: s.value_counts().idxmax()),
            interpretable_share=("interpretability_label", lambda s: float((s == "interpretable").mean())),
            phase_share=("tool_label", lambda s: float((s == "phase_trajectory").mean())),
            do_not_read_share=("tool_label", lambda s: float((s == "do_not_read_regression").mean())),
            best_spec_mode=("preferred_spec", lambda s: s.dropna().value_counts().idxmax() if not s.dropna().empty else ""),
            mean_adj_r2_best=("adj_r2_best", "mean"),
            top_beta_mode=("top_beta_best", lambda s: s.dropna().value_counts().idxmax() if not s.dropna().empty else ""),
            mean_b_sum_best=("b_sum_best", "mean"),
        )
        .sort_values(["window", "b1_group"])
        .reset_index(drop=True)
    )
    return window_level, window_summary


def summarize_b1_sign_stability(interval_models: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (group_name, spec), subset in interval_models.groupby(["b1_group", "spec"], sort=False):
        if subset["interval"].nunique() < 2:
            continue
        ordered = subset.sort_values("interval")
        rows.append(
            {
                "b1_group": group_name,
                "group_label": str(ordered["group_label"].iloc[0]),
                "spec": spec,
                "top_beta_consistent": int(ordered["top_beta"].nunique(dropna=True) == 1),
                "lag1_sign_consistent": int(ordered["sign_lag1"].dropna().nunique() <= 1),
                "x_sign_consistent": int(ordered["sign_x"].dropna().nunique() <= 1),
                "adj_r2_range": float(ordered["adj_r2"].max() - ordered["adj_r2"].min()),
            }
        )
    return pd.DataFrame(rows)


def summarize_b1_shocks(group_yearly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, subset in group_yearly.groupby("b1_group", sort=False):
        subset = subset.sort_values("year").reset_index(drop=True)
        label = str(subset["group_label"].iloc[0])
        for shock_year in (2020, 2022):
            match = subset[subset["year"] == shock_year]
            if match.empty:
                continue
            idx = int(match.index[0])
            level = float(match["median_revenue"].iloc[0])
            before = float(subset["median_revenue"].iloc[max(idx - 1, 0)])
            after = float(subset["median_revenue"].iloc[min(idx + 1, len(subset) - 1)])
            rows.append(
                {
                    "b1_group": group_name,
                    "group_label": label,
                    "shock_year": shock_year,
                    "median_revenue": level,
                    "change_before": float(level / before - 1.0) if abs(before) > 1e-12 else np.nan,
                    "change_after": float(after / level - 1.0) if abs(level) > 1e-12 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_b1_experiment(
    revenue_wide: pd.DataFrame,
    revenue_long: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    group_membership = build_b1_groups(revenue_wide)
    group_yearly, group_summary = compute_b1_group_medians(revenue_long, group_membership)
    interval_models, interval_summary = run_b1_interval_models(group_yearly)
    window_level, window_summary = run_b1_short_windows(group_yearly)
    sign_stability = summarize_b1_sign_stability(interval_models)
    shock_summary = summarize_b1_shocks(group_yearly)
    return {
        "group_membership": group_membership,
        "group_yearly": group_yearly,
        "group_summary": group_summary,
        "interval_models": interval_models,
        "interval_summary": interval_summary,
        "window_level": window_level,
        "window_summary": window_summary,
        "sign_stability": sign_stability,
        "shock_summary": shock_summary,
    }


def _plot_group_trajectories(group_yearly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    label_map = {
        "High growth": "Высокие темпы роста",
        "Low growth": "Низкие темпы роста",
        "Middle growth": "Средние темпы роста",
    }
    for label, subset in group_yearly.groupby("group_label", sort=False):
        ax.plot(subset["year"], subset["median_revenue"], marker="o", label=label_map.get(label, label))
    for year in (2014, 2020, 2022):
        ax.axvline(year, linestyle="--", linewidth=1.0, color="#999999")
    ax.set_title("Медианные траектории выручки по группам B1")
    ax.set_ylabel("Медианная выручка")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_group_growth(group_yearly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    label_map = {
        "High growth": "Высокие темпы роста",
        "Low growth": "Низкие темпы роста",
        "Middle growth": "Средние темпы роста",
    }
    for label, subset in group_yearly.groupby("group_label", sort=False):
        valid = subset.dropna(subset=["median_growth"])
        ax.plot(valid["year"], valid["median_growth"], marker="o", label=label_map.get(label, label))
    for year in (2014, 2020, 2022):
        ax.axvline(year, linestyle="--", linewidth=1.0, color="#999999")
    ax.set_title("Темпы роста медианной выручки по группам B1")
    ax.set_ylabel("Темп роста")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_interval_adj_r2(interval_models: pd.DataFrame):
    table = interval_models.pivot_table(index=["group_label", "interval"], columns="spec", values="adj_r2", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(table))
    width = 0.2
    for idx, column in enumerate(table.columns):
        ax.bar(x + idx * width, table[column].to_numpy(dtype=float), width=width, label=column)
    ax.set_xticks(x + width * max(len(table.columns) - 1, 0) / 2)
    ax.set_xticklabels([f"{group}\n{interval}" for group, interval in table.index], rotation=0)
    ax.set_ylabel("Скорректированный R²")
    ax.set_title("Модели B1 на полных интервалах: скорректированный R²")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_window_modes(window_summary: pd.DataFrame):
    table = window_summary.pivot_table(index=["group_label", "window"], columns="dominant_tool", values="interpretable_share", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"{group}\nW={window}" for group, window in table.index]
    ax.bar(labels, table.max(axis=1).to_numpy(dtype=float), color="#6baed6")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Доля интерпретируемых окон")
    ax.set_title("Короткие окна B1: доля интерпретируемых окон")
    fig.tight_layout()
    return fig


def save_b1_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)

    if not results["group_yearly"].empty:
        save_figure(_plot_group_trajectories(results["group_yearly"]), out / "group_trajectories.png")
        save_figure(_plot_group_growth(results["group_yearly"]), out / "group_growth.png")
    if not results["interval_models"].empty:
        save_figure(_plot_interval_adj_r2(results["interval_models"]), out / "interval_adj_r2.png")
    if not results["window_summary"].empty:
        save_figure(_plot_window_modes(results["window_summary"]), out / "window_interpretable_share.png")

    summary = {
        "group_count": int(results["group_summary"]["b1_group"].nunique()) if not results["group_summary"].empty else 0,
        "interval_rows": int(len(results["interval_models"])),
        "window_rows": int(len(results["window_level"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
