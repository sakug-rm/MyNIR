from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nonlinear_lab.features import growth_rate, make_regression_df
from nonlinear_lab.plan_h_experiment import compute_condition_number
from nonlinear_lab.regression import fit_enter_with_beta


def _acf(series: np.ndarray, lag: int) -> float:
    x = np.asarray(series, dtype=float)
    if len(x) <= lag:
        return float("nan")
    x0 = x[:-lag]
    x1 = x[lag:]
    if np.std(x0) <= 1e-12 or np.std(x1) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x0, x1)[0, 1])


def _sign_stability(values: list[float]) -> float:
    usable = [value for value in values if np.isfinite(value) and abs(value) > 1e-12]
    if not usable:
        return 0.0
    signs = np.sign(usable)
    return float(max(np.mean(signs > 0), np.mean(signs < 0)))


def _window_scan(
    series: np.ndarray,
    *,
    window: int,
    lags: int,
    zero_guard: float = 1e-12,
    y_std_floor: float = 1e-8,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(series) - window + 1):
        end = start + window
        reg_df = make_regression_df(series[start:end], lags=lags, zero_guard=zero_guard)
        if len(reg_df) < 8:
            rows.append({"start": start, "end": end, "no_model": True, "cond": float("nan")})
            continue

        y = reg_df["omega"]
        X = reg_df.drop(columns=["omega"])
        cond = compute_condition_number(X, standardize=True)
        if y.std(ddof=0) <= y_std_floor:
            rows.append({"start": start, "end": end, "no_model": True, "cond": cond})
            continue

        try:
            model, beta = fit_enter_with_beta(X, y)
        except Exception:
            rows.append({"start": start, "end": end, "no_model": True, "cond": cond})
            continue

        rows.append(
            {
                "start": start,
                "end": end,
                "no_model": False,
                "cond": cond,
                "B_X_n": float(model.params.get("X_n", np.nan)),
                "B_Lag_1": float(model.params.get("Lag_1", np.nan)),
                "Beta_X_n": float(beta.get("X_n", np.nan)),
                "Beta_Lag_1": float(beta.get("Lag_1", np.nan)),
                "R2": float(model.rsquared),
            }
        )
    return rows


def run_ipp_variant_qc(
    ipp_long: pd.DataFrame,
    *,
    windows: list[int] | tuple[int, ...] = (24, 36, 48),
    lag_options: list[int] | tuple[int, ...] = (3, 6),
) -> dict[str, pd.DataFrame]:
    required = {"date", "series_code", "series_name", "variant", "index_value"}
    missing = required - set(ipp_long.columns)
    if missing:
        raise ValueError(f"IPP long data is missing columns: {sorted(missing)}")

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    grouped = ipp_long.sort_values("date").groupby(["series_code", "series_name", "variant"], sort=True)
    for (series_code, series_name, variant), frame in grouped:
        values = frame["index_value"].to_numpy(dtype=float)
        if len(values) < 6:
            continue
        omega = growth_rate(values)
        base_metrics = {
            "series_code": series_code,
            "series_name": series_name,
            "variant": variant,
            "n_obs": float(len(values)),
            "var_omega": float(np.var(omega, ddof=0)) if len(omega) else float("nan"),
            "acf1": _acf(values, 1),
            "acf2": _acf(values, 2),
        }

        for window in windows:
            if window >= len(values):
                continue
            for lags in lag_options:
                per_window = _window_scan(values, window=window, lags=lags)
                if not per_window:
                    continue
                for row in per_window:
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

                scan = pd.DataFrame(per_window)
                summary_rows.append(
                    {
                        **base_metrics,
                        "window": window,
                        "lags": lags,
                        "no_model_share": float(scan["no_model"].mean()),
                        "mean_cond": float(scan["cond"].replace([np.inf], np.nan).mean()),
                        "median_cond": float(scan["cond"].replace([np.inf], np.nan).median()),
                        "sign_stability_x": _sign_stability(scan.get("B_X_n", pd.Series(dtype=float)).tolist()),
                        "sign_stability_lag1": _sign_stability(scan.get("B_Lag_1", pd.Series(dtype=float)).tolist()),
                        "mean_r2": float(scan.get("R2", pd.Series(dtype=float)).mean()),
                    }
                )

    series_summary = pd.DataFrame(summary_rows).sort_values(["series_code", "variant", "window", "lags"]).reset_index(drop=True)
    window_level = pd.DataFrame(window_rows).sort_values(
        ["series_code", "variant", "window", "lags", "start"]
    ).reset_index(drop=True)
    variant_summary = (
        series_summary.groupby(["variant", "window", "lags"], as_index=False)
        .agg(
            series_count=("series_code", "nunique"),
            mean_var_omega=("var_omega", "mean"),
            mean_acf1=("acf1", "mean"),
            mean_acf2=("acf2", "mean"),
            mean_no_model_share=("no_model_share", "mean"),
            mean_cond=("mean_cond", "mean"),
            mean_r2=("mean_r2", "mean"),
        )
        .sort_values(["window", "lags", "variant"])
        .reset_index(drop=True)
    )
    return {
        "series_summary": series_summary,
        "window_summary": window_level.copy(),
        "window_level": window_level,
        "variant_summary": variant_summary,
    }


def save_ipp_variant_qc_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)

    summary = {
        "series_count": int(results["series_summary"]["series_code"].nunique()) if not results["series_summary"].empty else 0,
        "variant_count": int(results["series_summary"]["variant"].nunique()) if not results["series_summary"].empty else 0,
        "window_count": int(len(results["window_level"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
