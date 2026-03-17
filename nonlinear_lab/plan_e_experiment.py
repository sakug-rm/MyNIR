from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.models import (
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
)
from nonlinear_lab.plan_a_experiment import detect_degenerate_window
from nonlinear_lab.plan_b_experiment import score_selected_features
from nonlinear_lab.plan_h_experiment import compute_condition_number
from nonlinear_lab.regression import fit_enter_with_beta, stepwise_selection


PLAN_E_CASES = [
    {
        "model": "base",
        "case": "base_a_0_8",
        "regime": "устойчивое насыщение",
        "params": {"a": 0.8, "k": 1.0},
        "true_predictors": {"X_n"},
    },
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
        "case": "delay_g_0_2",
        "regime": "устойчивое насыщение с памятью",
        "params": {"g": 0.2},
        "true_predictors": {"Lag_1"},
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
        "case": "delay_g_1_25",
        "regime": "усиленный цикл",
        "params": {"g": 1.25},
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_1_5_gamma_0_5",
        "regime": "смешанная память и стационаризация",
        "params": {"q": 1.5, "gamma": 0.5},
        "true_predictors": {"X_n", "Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_2_8_gamma_0_5",
        "regime": "смешанная память, цикл",
        "params": {"q": 2.8, "gamma": 0.5},
        "true_predictors": {"X_n", "Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_1_5_gamma_neg_0_2",
        "regime": "отрицательная память и стационаризация",
        "params": {"q": 1.5, "gamma": -0.2},
        "true_predictors": {"X_n", "Lag_1"},
    },
]

DEFAULT_ALPHAS = (0.01, 0.05, 0.10)
VAR_OMEGA_THRESHOLD = 2.4e-8
COND_SCALED_THRESHOLD = 7.0e16


def _generate_case_series(case: dict[str, Any], steps: int, x0: float = 1e-4) -> np.ndarray:
    model = case["model"]
    if model == "base":
        return generate_base_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "delay":
        return generate_delay_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "mixed":
        return generate_mixed_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    raise ValueError("Unsupported model")


def alpha_configs(alphas: Iterable[float]) -> list[dict[str, float]]:
    configs: list[dict[str, float]] = []
    for alpha in sorted(float(value) for value in alphas):
        configs.append(
            {
                "alpha": alpha,
                "threshold_in": alpha,
                "threshold_out": min(alpha * 1.5, 0.20),
            }
        )
    return configs


def classify_interpretability_window(var_omega: float, cond_scaled: float) -> str:
    if var_omega <= VAR_OMEGA_THRESHOLD:
        return "low_dispersion"
    if cond_scaled >= COND_SCALED_THRESHOLD:
        return "collinearity_heavy"
    return "interpretable"


def _safe_rsquared(model: Any) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        value = float(model.rsquared)
    return value if np.isfinite(value) else float("nan")


def run_plan_e_experiment(
    *,
    steps: int = 180,
    window: int = 25,
    lags: int = 10,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    selected_cases = list(cases or PLAN_E_CASES)
    alpha_grid = alpha_configs(alphas)
    rows: list[dict[str, Any]] = []

    for case in selected_cases:
        series = _generate_case_series(case, steps=steps)
        for start in range(0, len(series) - window):
            end = start + window
            block = np.asarray(series[start:end], dtype=float)
            df = make_regression_df(block, lags=lags)
            if len(df) < 8:
                continue

            y = df["omega"]
            X_mat = df.drop(columns=["omega"])
            var_omega = float(np.var(y, ddof=0))
            cond_scaled = compute_condition_number(X_mat, standardize=True)
            interpretability = classify_interpretability_window(var_omega, cond_scaled)
            degenerate = int(detect_degenerate_window(block))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                enter_model, _ = fit_enter_with_beta(X_mat, y)

            midpoint = 0.5 * (start + end)
            stage = "early" if midpoint < steps / 3 else "late" if midpoint >= 2 * steps / 3 else "middle"

            for cfg in alpha_grid:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    selected = stepwise_selection(
                        X_mat,
                        y,
                        threshold_in=cfg["threshold_in"],
                        threshold_out=cfg["threshold_out"],
                    )

                score = score_selected_features(selected, case["true_predictors"])
                selected_set = set(selected)
                exact_recovery = float(selected_set == case["true_predictors"])
                rows.append(
                    {
                        "alpha": cfg["alpha"],
                        "threshold_in": cfg["threshold_in"],
                        "threshold_out": cfg["threshold_out"],
                        "model": case["model"],
                        "case": case["case"],
                        "regime": case["regime"],
                        "start": start,
                        "end": end,
                        "stage": stage,
                        "var_omega": var_omega,
                        "cond_scaled": cond_scaled,
                        "interpretability": interpretability,
                        "degenerate_window": degenerate,
                        "R2_enter": _safe_rsquared(enter_model),
                        "selected_count": score["selected_count"],
                        "hit_rate": score["hit_rate"],
                        "false_lag_count": score["false_lag_count"],
                        "false_lag_rate": score["false_lag_rate"],
                        "miss_rate": 1.0 - score["hit_rate"],
                        "no_model": float(not selected),
                        "exact_recovery": exact_recovery,
                    }
                )

    window_level = pd.DataFrame(rows).sort_values(["alpha", "model", "case", "start"]).reset_index(drop=True)

    alpha_summary = (
        window_level.groupby("alpha", as_index=False)
        .agg(
            windows=("case", "size"),
            false_lag_rate=("false_lag_rate", "mean"),
            miss_rate=("miss_rate", "mean"),
            no_model_share=("no_model", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            mean_selected_count=("selected_count", "mean"),
        )
        .sort_values("alpha")
        .reset_index(drop=True)
    )

    case_summary = (
        window_level.groupby(["alpha", "model", "case", "regime"], as_index=False)
        .agg(
            windows=("case", "size"),
            false_lag_rate=("false_lag_rate", "mean"),
            miss_rate=("miss_rate", "mean"),
            no_model_share=("no_model", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            mean_selected_count=("selected_count", "mean"),
            interpretable_share=("interpretability", lambda s: float(np.mean(s == "interpretable"))),
        )
        .sort_values(["alpha", "model", "case"])
        .reset_index(drop=True)
    )

    interpretability_summary = (
        window_level.groupby(["alpha", "interpretability"], as_index=False)
        .agg(
            windows=("case", "size"),
            false_lag_rate=("false_lag_rate", "mean"),
            miss_rate=("miss_rate", "mean"),
            no_model_share=("no_model", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            mean_selected_count=("selected_count", "mean"),
        )
        .sort_values(["alpha", "interpretability"])
        .reset_index(drop=True)
    )

    model_summary = (
        window_level.groupby(["alpha", "model"], as_index=False)
        .agg(
            windows=("case", "size"),
            false_lag_rate=("false_lag_rate", "mean"),
            miss_rate=("miss_rate", "mean"),
            no_model_share=("no_model", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            mean_selected_count=("selected_count", "mean"),
        )
        .sort_values(["alpha", "model"])
        .reset_index(drop=True)
    )

    summary = {
        "alphas": list(alpha_summary["alpha"]),
        "cases": [case["case"] for case in selected_cases],
        "window_count": int(len(window_level)),
        "var_omega_threshold": VAR_OMEGA_THRESHOLD,
        "cond_scaled_threshold": COND_SCALED_THRESHOLD,
    }

    return {
        "window_level": window_level,
        "alpha_summary": alpha_summary,
        "case_summary": case_summary,
        "interpretability_summary": interpretability_summary,
        "model_summary": model_summary,
        "summary": pd.DataFrame([summary]),
    }


def save_plan_e_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for name, frame in results.items():
        if name == "summary":
            continue
        frame.to_csv(destination / f"{name}.csv", index=False)

    summary_payload = results["summary"].iloc[0].to_dict()
    with (destination / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    return destination
