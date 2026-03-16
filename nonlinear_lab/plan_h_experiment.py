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
from nonlinear_lab.plan_a_experiment import detect_degenerate_window
from nonlinear_lab.plan_b_experiment import score_selected_features
from nonlinear_lab.plan_c_experiment import compute_ranking_metrics
from nonlinear_lab.regression import fit_enter_with_beta, stepwise_selection


PLAN_H_CASES = [
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


def _generate_case_series(case: dict[str, Any], steps: int, x0: float = 1e-4) -> np.ndarray:
    model = case["model"]
    if model == "base":
        return generate_base_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "delay":
        return generate_delay_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "mixed":
        return generate_mixed_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    raise ValueError("Unsupported model")


def compute_condition_number(X: pd.DataFrame, standardize: bool = False) -> float:
    values = np.asarray(X, dtype=float)
    if values.size == 0:
        return 0.0
    if standardize:
        std = values.std(axis=0, ddof=0)
        good = std > 0
        values = values[:, good]
        if values.size == 0:
            return 0.0
        values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cond = float(np.linalg.cond(values))
    return cond if np.isfinite(cond) else float("inf")


def compute_vif(X: pd.DataFrame) -> pd.Series:
    frame = X.copy()
    if frame.empty:
        return pd.Series(dtype=float)
    out: dict[str, float] = {}
    for column in frame.columns:
        y = frame[column].to_numpy(dtype=float)
        others = frame.drop(columns=[column]).to_numpy(dtype=float)
        if others.size == 0:
            out[column] = 1.0
            continue
        design = np.column_stack([np.ones(len(others)), others])
        try:
            coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coeffs
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        except np.linalg.LinAlgError:
            r2 = 1.0
        if r2 >= 1.0 - 1e-12:
            out[column] = float("inf")
        else:
            out[column] = float(1.0 / (1.0 - r2))
    return pd.Series(out, dtype=float)


def classify_window_failure(
    *,
    degenerate_window: bool,
    no_model: bool,
    false_lag_count: float,
    beta_pairwise: float,
) -> str:
    if degenerate_window or no_model:
        return "low_dispersion"
    if false_lag_count > 0 or beta_pairwise < 0.5:
        return "collinearity"
    return "stable"


def _safe_log10(values: pd.Series | np.ndarray, floor: float = 1e-12) -> pd.Series:
    series = pd.Series(values, dtype=float)
    return np.log10(series.clip(lower=floor))


def _risk_quartile(series: pd.Series, ascending: bool) -> pd.Series:
    ranks = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranks, q=4, labels=["Q1", "Q2", "Q3", "Q4"])


def _scan_threshold_rule(window_level: pd.DataFrame) -> pd.DataFrame:
    var_candidates = np.quantile(window_level["var_omega"], [0.1, 0.2, 0.25, 0.33, 0.4, 0.5])
    cond_candidates = np.quantile(window_level["cond_scaled"].replace([np.inf], np.nan).dropna(), [0.5, 0.6, 0.67, 0.75, 0.8, 0.9])
    rows: list[dict[str, float]] = []
    target = window_level["degraded_window"].to_numpy(dtype=int)
    for var_threshold in np.unique(np.round(var_candidates, 12)):
        for cond_threshold in np.unique(np.round(cond_candidates, 12)):
            predicted = (
                (window_level["var_omega"] <= var_threshold)
                | (window_level["cond_scaled"] >= cond_threshold)
            ).astype(int)
            tp = float(np.sum((predicted == 1) & (target == 1)))
            tn = float(np.sum((predicted == 0) & (target == 0)))
            fp = float(np.sum((predicted == 1) & (target == 0)))
            fn = float(np.sum((predicted == 0) & (target == 1)))
            recall = tp / max(tp + fn, 1.0)
            specificity = tn / max(tn + fp, 1.0)
            precision = tp / max(tp + fp, 1.0)
            rows.append(
                {
                    "var_omega_threshold": float(var_threshold),
                    "cond_scaled_threshold": float(cond_threshold),
                    "balanced_accuracy": 0.5 * (recall + specificity),
                    "recall": recall,
                    "specificity": specificity,
                    "precision": precision,
                    "predicted_share": float(np.mean(predicted)),
                }
            )
    threshold_summary = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "precision", "recall"],
        ascending=False,
    )
    return threshold_summary.reset_index(drop=True)


def run_plan_h_experiment(
    *,
    steps: int = 180,
    window: int = 25,
    lags: int = 10,
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    for case in PLAN_H_CASES:
        series = _generate_case_series(case, steps=steps)
        for start in range(0, len(series) - window):
            end = start + window
            block = np.asarray(series[start:end], dtype=float)
            df = make_regression_df(block, lags=lags)
            if len(df) < 8:
                continue

            y = df["omega"]
            X_mat = df.drop(columns=["omega"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                enter_model, beta = fit_enter_with_beta(X_mat, y)
                selected = stepwise_selection(
                    X_mat,
                    y,
                    threshold_in=threshold_in,
                    threshold_out=threshold_out,
                )

            params = enter_model.params.drop(labels="const", errors="ignore")
            abs_b = params.abs()
            abs_beta = beta.abs()
            b_metrics = compute_ranking_metrics(abs_b, case["true_predictors"])
            beta_metrics = compute_ranking_metrics(abs_beta, case["true_predictors"])
            vif = compute_vif(X_mat)
            score_metrics = score_selected_features(selected, case["true_predictors"])
            degenerate = detect_degenerate_window(block)
            no_model = int(not selected)
            failure_type = classify_window_failure(
                degenerate_window=degenerate,
                no_model=no_model == 1,
                false_lag_count=score_metrics["false_lag_count"],
                beta_pairwise=float(beta_metrics["pairwise_score"]),
            )
            degraded_window = int(failure_type != "stable")

            total_beta_mass = float(abs_beta.sum())
            true_beta_mass = float(sum(float(abs_beta.get(name, 0.0)) for name in case["true_predictors"]))
            total_b_mass = float(abs_b.sum())
            true_b_mass = float(sum(float(abs_b.get(name, 0.0)) for name in case["true_predictors"]))
            midpoint = 0.5 * (start + end)
            stage = "early" if midpoint < steps / 3 else "late" if midpoint >= 2 * steps / 3 else "middle"

            rows.append(
                {
                    "model": case["model"],
                    "case": case["case"],
                    "regime": case["regime"],
                    "start": start,
                    "end": end,
                    "stage": stage,
                    "var_x": float(np.var(df["X_n"], ddof=0)),
                    "var_omega": float(np.var(y, ddof=0)),
                    "mean_x": float(np.mean(df["X_n"])),
                    "mean_omega": float(np.mean(y)),
                    "cond_raw": compute_condition_number(X_mat, standardize=False),
                    "cond_scaled": compute_condition_number(X_mat, standardize=True),
                    "max_vif": float(vif.replace([np.inf], np.nan).max()) if not vif.empty else np.nan,
                    "mean_vif": float(vif.replace([np.inf], np.nan).mean()) if not vif.empty else np.nan,
                    "infinite_vif_share": float(np.mean(np.isinf(vif))) if not vif.empty else 0.0,
                    "selected_count": score_metrics["selected_count"],
                    "false_lag_count": score_metrics["false_lag_count"],
                    "false_lag_rate": score_metrics["false_lag_rate"],
                    "hit_rate": score_metrics["hit_rate"],
                    "no_model": no_model,
                    "degenerate_window": int(degenerate),
                    "failure_type": failure_type,
                    "degraded_window": degraded_window,
                    "R2_enter": float(enter_model.rsquared),
                    "pairwise_b": float(b_metrics["pairwise_score"]),
                    "pairwise_beta": float(beta_metrics["pairwise_score"]),
                    "correct_top_b": float(b_metrics["correct_top"]),
                    "correct_top_beta": float(beta_metrics["correct_top"]),
                    "true_beta_mass_share": true_beta_mass / max(total_beta_mass, 1e-12),
                    "true_b_mass_share": true_b_mass / max(total_b_mass, 1e-12),
                    "beta_scale_gap": float(abs(abs_b - abs_beta.reindex(abs_b.index).fillna(0.0)).mean()),
                }
            )

    window_level = pd.DataFrame(rows).sort_values(["model", "case", "start"]).reset_index(drop=True)
    window_level["log10_var_x"] = _safe_log10(window_level["var_x"])
    window_level["log10_var_omega"] = _safe_log10(window_level["var_omega"])
    window_level["log10_cond_raw"] = _safe_log10(window_level["cond_raw"].replace([np.inf], np.nan).fillna(window_level["cond_raw"].replace([np.inf], np.nan).max()))
    finite_scaled = window_level["cond_scaled"].replace([np.inf], np.nan)
    window_level["log10_cond_scaled"] = _safe_log10(finite_scaled.fillna(finite_scaled.max()))
    window_level["var_omega_band"] = _risk_quartile(window_level["var_omega"], ascending=True)
    window_level["cond_scaled_band"] = _risk_quartile(finite_scaled.fillna(finite_scaled.max()), ascending=False)

    case_summary = (
        window_level.groupby(["model", "case", "regime"], as_index=False)
        .agg(
            windows=("case", "size"),
            degraded_share=("degraded_window", "mean"),
            low_dispersion_share=("failure_type", lambda s: float(np.mean(s == "low_dispersion"))),
            collinearity_share=("failure_type", lambda s: float(np.mean(s == "collinearity"))),
            mean_var_x=("var_x", "mean"),
            mean_var_omega=("var_omega", "mean"),
            median_cond_raw=("cond_raw", "median"),
            median_cond_scaled=("cond_scaled", "median"),
            mean_max_vif=("max_vif", "mean"),
            false_lag_rate=("false_lag_rate", "mean"),
            no_model_share=("no_model", "mean"),
            mean_pairwise_beta=("pairwise_beta", "mean"),
            mean_true_beta_mass_share=("true_beta_mass_share", "mean"),
        )
        .sort_values(["model", "case"])
        .reset_index(drop=True)
    )

    overall_rows: list[dict[str, Any]] = []
    for scope, part in [("overall", window_level), *[(name, frame) for name, frame in window_level.groupby("model")]]:
        degraded = part[part["degraded_window"] == 1]
        stable = part[part["degraded_window"] == 0]
        overall_rows.append(
            {
                "scope": scope,
                "windows": float(len(part)),
                "degraded_share": float(part["degraded_window"].mean()),
                "low_dispersion_share": float(np.mean(part["failure_type"] == "low_dispersion")),
                "collinearity_share": float(np.mean(part["failure_type"] == "collinearity")),
                "stable_share": float(np.mean(part["failure_type"] == "stable")),
                "median_var_omega_degraded": float(degraded["var_omega"].median()) if len(degraded) else np.nan,
                "median_var_omega_stable": float(stable["var_omega"].median()) if len(stable) else np.nan,
                "median_cond_scaled_degraded": float(degraded["cond_scaled"].median()) if len(degraded) else np.nan,
                "median_cond_scaled_stable": float(stable["cond_scaled"].median()) if len(stable) else np.nan,
                "median_max_vif_degraded": float(degraded["max_vif"].median()) if len(degraded) else np.nan,
                "median_max_vif_stable": float(stable["max_vif"].median()) if len(stable) else np.nan,
                "mean_false_lag_rate_degraded": float(degraded["false_lag_rate"].mean()) if len(degraded) else np.nan,
                "mean_false_lag_rate_stable": float(stable["false_lag_rate"].mean()) if len(stable) else np.nan,
            }
        )
    overall_summary = pd.DataFrame(overall_rows)

    risk_grid = (
        window_level.groupby(["var_omega_band", "cond_scaled_band"], observed=False, as_index=False)
        .agg(
            windows=("case", "size"),
            degraded_share=("degraded_window", "mean"),
            false_lag_rate=("false_lag_rate", "mean"),
            no_model_share=("no_model", "mean"),
            mean_pairwise_beta=("pairwise_beta", "mean"),
        )
        .sort_values(["var_omega_band", "cond_scaled_band"])
        .reset_index(drop=True)
    )

    threshold_summary = _scan_threshold_rule(window_level)
    best_threshold = threshold_summary.iloc[[0]].copy()

    summary = {
        "window_level": window_level,
        "case_summary": case_summary,
        "overall_summary": overall_summary,
        "risk_grid": risk_grid,
        "threshold_summary": threshold_summary,
        "best_threshold": best_threshold,
    }
    return summary


def save_plan_h_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(output / f"{name}.csv", index=False)

    best = results["best_threshold"].iloc[0].to_dict()
    summary_json = {
        "windows": int(len(results["window_level"])),
        "best_threshold": {key: float(value) for key, value in best.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False))
    return output
