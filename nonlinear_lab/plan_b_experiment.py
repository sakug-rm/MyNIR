from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.models import (
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
)
from nonlinear_lab.regression import fit_enter_with_beta, stepwise_selection


PLAN_B_CASES = [
    {
        "model": "base",
        "case": "base_a_2_2",
        "regime": "затухающие колебания",
        "params": {"a": 2.2, "k": 1.0},
        "true_predictors": {"X_n"},
    },
    {
        "model": "base",
        "case": "base_a_2_52",
        "regime": "предтурбулентный цикл",
        "params": {"a": 2.52, "k": 1.0},
        "true_predictors": {"X_n"},
    },
    {
        "model": "delay",
        "case": "delay_g_0_8",
        "regime": "затухающие колебания",
        "params": {"g": 0.8},
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "delay",
        "case": "delay_g_1_05",
        "regime": "цикл",
        "params": {"g": 1.05},
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "delay",
        "case": "delay_g_1_25",
        "regime": "сложный цикл",
        "params": {"g": 1.25},
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_2_8_gamma_0_5",
        "regime": "цикл со смешанной памятью",
        "params": {"q": 2.8, "gamma": 0.5},
        "true_predictors": {"X_n", "Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_2_8_gamma_neg_0_2",
        "regime": "цикл с отрицательной памятью",
        "params": {"q": 2.8, "gamma": -0.2},
        "true_predictors": {"X_n", "Lag_1"},
    },
]


def estimate_characteristic_period(
    series: np.ndarray,
    min_lag: int = 2,
    max_lag: int | None = None,
) -> int:
    """Estimate the dominant cycle length from the autocorrelation profile."""
    x = np.asarray(series, dtype=float)
    if len(x) < min_lag + 2:
        raise ValueError("series is too short to estimate a characteristic period")

    centered = x - np.mean(x)
    if np.allclose(centered, 0.0):
        return min_lag

    n = len(centered)
    upper = min(max_lag or (n // 2), n - 2)
    lags = np.arange(min_lag, upper + 1)
    if len(lags) == 0:
        return min_lag

    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return min_lag

    acf = np.array([np.dot(centered[:-lag], centered[lag:]) / denom for lag in lags], dtype=float)
    peak_idx: list[int] = []
    for idx in range(1, len(acf) - 1):
        if acf[idx] >= acf[idx - 1] and acf[idx] >= acf[idx + 1] and acf[idx] > 0:
            peak_idx.append(idx)

    if peak_idx:
        best = max(peak_idx, key=lambda idx: (acf[idx], -lags[idx]))
        return int(lags[best])

    return int(lags[int(np.argmax(acf))])


def score_selected_features(selected: list[str], true_predictors: set[str]) -> dict[str, float]:
    """Compute hit-rate and false-lag statistics for a selected feature set."""
    selected_set = set(selected)
    hits = len(selected_set & true_predictors)
    false_lags = len([feature for feature in selected if feature not in true_predictors])
    return {
        "selected_count": float(len(selected)),
        "hit_rate": hits / max(len(true_predictors), 1),
        "false_lag_count": float(false_lags),
        "false_lag_rate": false_lags / max(len(selected), 1),
    }


def _generate_case_series(case: dict[str, Any], steps: int, x0: float = 1e-4) -> np.ndarray:
    model = case["model"]
    if model == "base":
        return generate_base_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "delay":
        return generate_delay_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "mixed":
        return generate_mixed_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    raise ValueError("Unsupported model")


def _rho_band(rho: float) -> str:
    if rho < 1.0:
        return "rho<1"
    if rho < 2.0:
        return "1<=rho<2"
    return "rho>=2"


def _rolling_stepwise_case(
    series: np.ndarray,
    *,
    model: str,
    case: str,
    regime: str,
    true_predictors: set[str],
    window: int,
    lags: int,
    period: int,
    threshold_in: float,
    threshold_out: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(series) - window):
        end = start + window
        block = np.asarray(series[start:end], dtype=float)
        df = make_regression_df(block, lags=lags)
        if len(df) < 8:
            continue

        y = df["omega"]
        X_mat = df.drop(columns=["omega"])
        selected = stepwise_selection(
            X_mat,
            y,
            threshold_in=threshold_in,
            threshold_out=threshold_out,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            enter_model, _ = fit_enter_with_beta(X_mat, y)
        lag_columns = [f"Lag_{idx}" for idx in range(1, lags + 1) if f"Lag_{idx}" in enter_model.params.index]
        b_sum = float(sum(float(enter_model.params.get(col, 0.0)) for col in lag_columns))

        row = {
            "model": model,
            "case": case,
            "regime": regime,
            "start": start,
            "end": end,
            "window": window,
            "lags": lags,
            "period": period,
            "rho": window / max(period, 1),
            "rho_band": _rho_band(window / max(period, 1)),
            "no_model": 1 if not selected else 0,
            "b_sum": b_sum,
            "R2_enter": float(enter_model.rsquared),
            "selected": selected,
        }
        row.update(score_selected_features(selected, true_predictors))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_plan_b_results(window_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Plan B diagnostics by model, lags and rho band."""
    data = window_level.copy()
    if "rho_band" not in data.columns:
        data["rho_band"] = data["rho"].map(_rho_band)
    count_column = "case" if "case" in window_level.columns else "model"
    grouped = (
        data.groupby(["model", "lags", "rho_band"], as_index=False)
        .agg(
            windows=(count_column, "size"),
            false_lag_rate=("false_lag_rate", "mean"),
            hit_rate=("hit_rate", "mean"),
            no_model_share=("no_model", "mean"),
            mean_selected_count=("selected_count", "mean"),
            mean_false_lag_count=("false_lag_count", "mean"),
            mean_b_sum=("b_sum", "mean"),
            std_b_sum=("b_sum", "std"),
            mean_period=("period", "mean"),
        )
        .sort_values(["model", "lags", "rho_band"])
        .reset_index(drop=True)
    )
    grouped["std_b_sum"] = grouped["std_b_sum"].fillna(0.0)
    return grouped


def run_plan_b_experiment(
    *,
    steps: int = 180,
    window_sizes: list[int] | tuple[int, ...] = (10, 15, 20, 25, 30, 40, 50),
    lag_options: list[int] | tuple[int, ...] = (3, 5, 10),
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
) -> dict[str, pd.DataFrame]:
    window_rows: list[pd.DataFrame] = []
    period_rows: list[dict[str, Any]] = []

    for case in PLAN_B_CASES:
        series = _generate_case_series(case, steps=steps)
        period = estimate_characteristic_period(series, min_lag=2, max_lag=min(60, steps // 2))
        period_rows.append(
            {
                "model": case["model"],
                "case": case["case"],
                "regime": case["regime"],
                "period": period,
            }
        )
        for window in window_sizes:
            for lags in lag_options:
                min_window = lags + 8
                if window < min_window:
                    continue
                frame = _rolling_stepwise_case(
                    series,
                    model=case["model"],
                    case=case["case"],
                    regime=case["regime"],
                    true_predictors=case["true_predictors"],
                    window=window,
                    lags=lags,
                    period=period,
                    threshold_in=threshold_in,
                    threshold_out=threshold_out,
                )
                if not frame.empty:
                    window_rows.append(frame)

    window_level = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    period_summary = pd.DataFrame(period_rows)

    case_summary = (
        window_level.groupby(["model", "case", "regime", "window", "lags"], as_index=False)
        .agg(
            period=("period", "first"),
            rho=("rho", "first"),
            windows=("selected", "size"),
            false_lag_rate=("false_lag_rate", "mean"),
            hit_rate=("hit_rate", "mean"),
            no_model_share=("no_model", "mean"),
            mean_false_lag_count=("false_lag_count", "mean"),
            mean_selected_count=("selected_count", "mean"),
            mean_b_sum=("b_sum", "mean"),
            std_b_sum=("b_sum", "std"),
            mean_R2_enter=("R2_enter", "mean"),
        )
        .sort_values(["model", "case", "lags", "window"])
        .reset_index(drop=True)
    )
    case_summary["std_b_sum"] = case_summary["std_b_sum"].fillna(0.0)
    case_summary["rho_band"] = case_summary["rho"].map(_rho_band)

    overall_summary = summarize_plan_b_results(window_level)

    rho_curve = (
        case_summary.groupby(["model", "lags", "window"], as_index=False)
        .agg(
            mean_rho=("rho", "mean"),
            false_lag_rate=("false_lag_rate", "mean"),
            hit_rate=("hit_rate", "mean"),
            no_model_share=("no_model_share", "mean"),
            std_b_sum=("std_b_sum", "mean"),
        )
        .sort_values(["model", "lags", "window"])
        .reset_index(drop=True)
    )

    return {
        "window_level": window_level,
        "case_summary": case_summary,
        "overall_summary": overall_summary,
        "period_summary": period_summary,
        "rho_curve": rho_curve,
    }


def save_plan_b_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(path / f"{name}.csv", index=False)

    summary = {
        name: {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }
        for name, frame in results.items()
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
