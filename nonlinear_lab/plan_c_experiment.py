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


PLAN_C_CASES = [
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


def _mean_abs_offdiag_corr(X: pd.DataFrame) -> tuple[float, float]:
    corr = X.corr().abs()
    if corr.empty or len(corr.columns) < 2:
        return 0.0, 0.0
    mask = ~np.eye(len(corr), dtype=bool)
    values = corr.where(mask).stack()
    if len(values) == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.max())


def compute_ranking_metrics(scores: pd.Series, true_predictors: set[str]) -> dict[str, float | str]:
    """Summarize how well a score vector ranks the true structural predictors."""
    usable = scores.dropna().abs().sort_values(ascending=False)
    if usable.empty:
        return {
            "top_predictor": "",
            "correct_top": 0.0,
            "topk_recall": 0.0,
            "pairwise_score": 0.0,
            "margin": 0.0,
        }

    k = max(len(true_predictors), 1)
    top_predictor = str(usable.index[0])
    topk = set(usable.head(k).index)
    false_predictors = [name for name in usable.index if name not in true_predictors]

    pairwise: list[float] = []
    for true_name in true_predictors:
        for false_name in false_predictors:
            pairwise.append(float(usable.get(true_name, 0.0) > usable.get(false_name, 0.0)))

    best_true = max((float(usable.get(name, 0.0)) for name in true_predictors), default=0.0)
    best_false = max((float(usable.get(name, 0.0)) for name in false_predictors), default=0.0)

    return {
        "top_predictor": top_predictor,
        "correct_top": float(top_predictor in true_predictors),
        "topk_recall": len(topk & true_predictors) / k,
        "pairwise_score": float(np.mean(pairwise)) if pairwise else 1.0,
        "margin": best_true - best_false,
    }


def run_plan_c_experiment(
    *,
    steps: int = 180,
    window: int = 25,
    lags: int = 10,
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
    corr_threshold: float = 0.9,
) -> dict[str, pd.DataFrame]:
    variable_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for case in PLAN_C_CASES:
        series = _generate_case_series(case, steps=steps)
        for start in range(0, len(series) - window):
            end = start + window
            df = make_regression_df(series[start:end], lags=lags)
            if len(df) < 8:
                continue

            y = df["omega"]
            X_mat = df.drop(columns=["omega"])
            mean_abs_corr, max_abs_corr = _mean_abs_offdiag_corr(X_mat)

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
            pvalues = enter_model.pvalues.drop(labels="const", errors="ignore")
            abs_b = params.abs()
            abs_beta = beta.abs()
            true_predictors = case["true_predictors"]
            false_stepwise = int(bool(selected) and any(feature not in true_predictors for feature in selected))
            problem_window = int(false_stepwise or mean_abs_corr >= corr_threshold)

            b_metrics = compute_ranking_metrics(abs_b, true_predictors)
            beta_metrics = compute_ranking_metrics(abs_beta, true_predictors)
            stage = "early" if start < (steps - window) / 3 else "late" if start >= 2 * (steps - window) / 3 else "middle"

            window_rows.append(
                {
                    "model": case["model"],
                    "case": case["case"],
                    "regime": case["regime"],
                    "start": start,
                    "end": end,
                    "stage": stage,
                    "mean_abs_corr": mean_abs_corr,
                    "max_abs_corr": max_abs_corr,
                    "false_stepwise": false_stepwise,
                    "problem_window": problem_window,
                    "selected_count": float(len(selected)),
                    "R2_enter": float(enter_model.rsquared),
                    "top_b": b_metrics["top_predictor"],
                    "top_beta": beta_metrics["top_predictor"],
                    "correct_top_b": b_metrics["correct_top"],
                    "correct_top_beta": beta_metrics["correct_top"],
                    "topk_b": b_metrics["topk_recall"],
                    "topk_beta": beta_metrics["topk_recall"],
                    "pairwise_b": b_metrics["pairwise_score"],
                    "pairwise_beta": beta_metrics["pairwise_score"],
                    "margin_b": b_metrics["margin"],
                    "margin_beta": beta_metrics["margin"],
                    "delta_pairwise": b_metrics["pairwise_score"] - beta_metrics["pairwise_score"],
                }
            )

            b_rank = abs_b.rank(method="dense", ascending=False)
            beta_rank = abs_beta.rank(method="dense", ascending=False)
            for variable in X_mat.columns:
                variable_rows.append(
                    {
                        "model": case["model"],
                        "case": case["case"],
                        "regime": case["regime"],
                        "start": start,
                        "end": end,
                        "stage": stage,
                        "variable": variable,
                        "is_true_predictor": int(variable in true_predictors),
                        "B": float(params.get(variable, 0.0)),
                        "abs_B": float(abs_b.get(variable, 0.0)),
                        "Beta": float(beta.get(variable, 0.0)),
                        "abs_Beta": float(abs_beta.get(variable, 0.0)),
                        "p_value": float(pvalues.get(variable, np.nan)),
                        "selected_stepwise": int(variable in selected),
                        "false_stepwise": false_stepwise,
                        "problem_window": problem_window,
                        "mean_abs_corr": mean_abs_corr,
                        "rank_B": float(b_rank.get(variable, np.nan)),
                        "rank_Beta": float(beta_rank.get(variable, np.nan)),
                    }
                )

    window_summary = pd.DataFrame(window_rows)
    window_variable_level = pd.DataFrame(variable_rows)

    case_rows: list[dict[str, Any]] = []
    for (model, case_name, regime), case_frame in window_summary.groupby(["model", "case", "regime"], as_index=False):
        problem_frame = case_frame[case_frame["problem_window"] == 1]
        case_rows.append(
            {
                "model": model,
                "case": case_name,
                "regime": regime,
                "windows": float(len(case_frame)),
                "problem_share": float(case_frame["problem_window"].mean()),
                "false_stepwise_share": float(case_frame["false_stepwise"].mean()),
                "mean_abs_corr": float(case_frame["mean_abs_corr"].mean()),
                "all_correct_top_b": float(case_frame["correct_top_b"].mean()),
                "all_correct_top_beta": float(case_frame["correct_top_beta"].mean()),
                "all_topk_b": float(case_frame["topk_b"].mean()),
                "all_topk_beta": float(case_frame["topk_beta"].mean()),
                "all_pairwise_b": float(case_frame["pairwise_b"].mean()),
                "all_pairwise_beta": float(case_frame["pairwise_beta"].mean()),
                "problem_correct_top_b": float(problem_frame["correct_top_b"].mean()) if len(problem_frame) else np.nan,
                "problem_correct_top_beta": float(problem_frame["correct_top_beta"].mean()) if len(problem_frame) else np.nan,
                "problem_topk_b": float(problem_frame["topk_b"].mean()) if len(problem_frame) else np.nan,
                "problem_topk_beta": float(problem_frame["topk_beta"].mean()) if len(problem_frame) else np.nan,
                "problem_pairwise_b": float(problem_frame["pairwise_b"].mean()) if len(problem_frame) else np.nan,
                "problem_pairwise_beta": float(problem_frame["pairwise_beta"].mean()) if len(problem_frame) else np.nan,
            }
        )
    case_summary = pd.DataFrame(case_rows).sort_values(["model", "case"]).reset_index(drop=True)

    overall_rows: list[dict[str, Any]] = []
    for model, model_frame in window_summary.groupby("model", as_index=False):
        problem_frame = model_frame[model_frame["problem_window"] == 1]
        overall_rows.append(
            {
                "scope": model,
                "windows": float(len(model_frame)),
                "problem_share": float(model_frame["problem_window"].mean()),
                "false_stepwise_share": float(model_frame["false_stepwise"].mean()),
                "all_correct_top_b": float(model_frame["correct_top_b"].mean()),
                "all_correct_top_beta": float(model_frame["correct_top_beta"].mean()),
                "all_topk_b": float(model_frame["topk_b"].mean()),
                "all_topk_beta": float(model_frame["topk_beta"].mean()),
                "all_pairwise_b": float(model_frame["pairwise_b"].mean()),
                "all_pairwise_beta": float(model_frame["pairwise_beta"].mean()),
                "problem_correct_top_b": float(problem_frame["correct_top_b"].mean()) if len(problem_frame) else np.nan,
                "problem_correct_top_beta": float(problem_frame["correct_top_beta"].mean()) if len(problem_frame) else np.nan,
                "problem_topk_b": float(problem_frame["topk_b"].mean()) if len(problem_frame) else np.nan,
                "problem_topk_beta": float(problem_frame["topk_beta"].mean()) if len(problem_frame) else np.nan,
                "problem_pairwise_b": float(problem_frame["pairwise_b"].mean()) if len(problem_frame) else np.nan,
                "problem_pairwise_beta": float(problem_frame["pairwise_beta"].mean()) if len(problem_frame) else np.nan,
            }
        )

    overall_rows.append(
        {
            "scope": "overall",
            "windows": float(len(window_summary)),
            "problem_share": float(window_summary["problem_window"].mean()),
            "false_stepwise_share": float(window_summary["false_stepwise"].mean()),
            "all_correct_top_b": float(window_summary["correct_top_b"].mean()),
            "all_correct_top_beta": float(window_summary["correct_top_beta"].mean()),
            "all_topk_b": float(window_summary["topk_b"].mean()),
            "all_topk_beta": float(window_summary["topk_beta"].mean()),
            "all_pairwise_b": float(window_summary["pairwise_b"].mean()),
            "all_pairwise_beta": float(window_summary["pairwise_beta"].mean()),
            "problem_correct_top_b": float(window_summary.loc[window_summary["problem_window"] == 1, "correct_top_b"].mean()),
            "problem_correct_top_beta": float(window_summary.loc[window_summary["problem_window"] == 1, "correct_top_beta"].mean()),
            "problem_topk_b": float(window_summary.loc[window_summary["problem_window"] == 1, "topk_b"].mean()),
            "problem_topk_beta": float(window_summary.loc[window_summary["problem_window"] == 1, "topk_beta"].mean()),
            "problem_pairwise_b": float(window_summary.loc[window_summary["problem_window"] == 1, "pairwise_b"].mean()),
            "problem_pairwise_beta": float(window_summary.loc[window_summary["problem_window"] == 1, "pairwise_beta"].mean()),
        }
    )
    overall_summary = pd.DataFrame(overall_rows)

    frequency_summary = (
        window_variable_level.groupby(["model", "case", "variable", "is_true_predictor"], as_index=False)
        .agg(
            mean_abs_B=("abs_B", "mean"),
            mean_abs_Beta=("abs_Beta", "mean"),
            selection_rate=("selected_stepwise", "mean"),
            problem_mean_abs_B=("abs_B", lambda s: float(s[window_variable_level.loc[s.index, "problem_window"] == 1].mean()) if (window_variable_level.loc[s.index, "problem_window"] == 1).any() else np.nan),
            problem_mean_abs_Beta=("abs_Beta", lambda s: float(s[window_variable_level.loc[s.index, "problem_window"] == 1].mean()) if (window_variable_level.loc[s.index, "problem_window"] == 1).any() else np.nan),
            problem_selection_rate=("selected_stepwise", lambda s: float(s[window_variable_level.loc[s.index, "problem_window"] == 1].mean()) if (window_variable_level.loc[s.index, "problem_window"] == 1).any() else np.nan),
        )
        .sort_values(["model", "case", "selection_rate"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    example_windows = (
        window_summary.loc[window_summary["correct_top_beta"] > window_summary["correct_top_b"]]
        .sort_values(["pairwise_beta", "pairwise_b", "mean_abs_corr"], ascending=[False, True, False])
        .head(12)
        .reset_index(drop=True)
    )

    return {
        "window_variable_level": window_variable_level,
        "window_summary": window_summary,
        "case_summary": case_summary,
        "overall_summary": overall_summary,
        "frequency_summary": frequency_summary,
        "example_windows": example_windows,
    }


def save_plan_c_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    """Persist Plan C results to CSV and JSON artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, frame in results.items():
        frame.to_csv(output_path / f"{name}.csv", index=False)

    overall = results["overall_summary"].copy()
    summary = {
        row["scope"]: {
            key: float(value)
            for key, value in row.items()
            if key != "scope" and isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value)
        }
        for row in overall.to_dict(orient="records")
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return output_path
