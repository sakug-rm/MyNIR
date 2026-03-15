from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nonlinear_lab.features import growth_rate, make_regression_df
from nonlinear_lab.models import (
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
)
from nonlinear_lab.plan_a_experiment import detect_degenerate_window
from nonlinear_lab.plan_b_experiment import estimate_characteristic_period, score_selected_features
from nonlinear_lab.regression import fit_enter_with_beta, stepwise_selection


PLAN_G_CASES = [
    {
        "model": "base",
        "case": "base_a_0_8",
        "params": {"a": 0.8, "k": 1.0},
        "regime_family": "growth_no_memory",
        "true_predictors": {"X_n"},
    },
    {
        "model": "base",
        "case": "base_a_1_5",
        "params": {"a": 1.5, "k": 1.0},
        "regime_family": "growth_no_memory",
        "true_predictors": {"X_n"},
    },
    {
        "model": "base",
        "case": "base_a_2_2",
        "params": {"a": 2.2, "k": 1.0},
        "regime_family": "stable_cycle",
        "true_predictors": {"X_n"},
    },
    {
        "model": "base",
        "case": "base_a_2_8",
        "params": {"a": 2.8, "k": 1.0},
        "regime_family": "chaotic_informative",
        "true_predictors": {"X_n"},
    },
    {
        "model": "delay",
        "case": "delay_g_0_2",
        "params": {"g": 0.2},
        "regime_family": "growth_with_memory",
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "delay",
        "case": "delay_g_0_8",
        "params": {"g": 0.8},
        "regime_family": "stable_cycle",
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "delay",
        "case": "delay_g_1_25",
        "params": {"g": 1.25},
        "regime_family": "stable_cycle",
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "delay",
        "case": "delay_g_1_6",
        "params": {"g": 1.6},
        "regime_family": "chaotic_informative",
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_1_5_gamma_0_5",
        "params": {"q": 1.5, "gamma": 0.5},
        "regime_family": "growth_with_memory",
        "true_predictors": {"X_n", "Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_2_8_gamma_0_5",
        "params": {"q": 2.8, "gamma": 0.5},
        "regime_family": "stable_cycle",
        "true_predictors": {"X_n", "Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_3_5_gamma_0_5",
        "params": {"q": 3.5, "gamma": 0.5},
        "regime_family": "chaotic_informative",
        "true_predictors": {"X_n", "Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_1_5_gamma_neg_0_2",
        "params": {"q": 1.5, "gamma": -0.2},
        "regime_family": "growth_with_memory",
        "true_predictors": {"X_n", "Lag_1"},
    },
]


REGRESSION_ADMISSIBLE = {"growth_no_memory", "growth_with_memory", "stable_cycle"}
TOOL_BY_LABEL = {
    "growth_no_memory": "direct_identification",
    "growth_with_memory": "regression",
    "stable_cycle": "regression",
    "chaotic_informative": "phase_analysis",
    "plateau_degenerate": "b_sum",
    "collapse": "degradation",
}


def extract_window_diagnostics(window: np.ndarray, zero_guard: float = 1e-12) -> dict[str, float]:
    """Compute simple regime-diagnostic features for a window."""
    x = np.asarray(window, dtype=float)
    mean_x = float(np.mean(x))
    std_x = float(np.std(x, ddof=0))
    rel_range = float((np.max(x) - np.min(x)) / max(abs(mean_x), zero_guard))
    omega = growth_rate(x, zero_guard=zero_guard)
    omega_std = float(np.std(omega, ddof=0)) if len(omega) else 0.0

    diffs = np.diff(x)
    turning_rate = 0.0
    if len(diffs) >= 2:
        turning_rate = float(np.mean(np.sign(diffs[1:]) != np.sign(diffs[:-1])))

    rounded = np.round(x, 4)
    unique_ratio = float(len(np.unique(rounded)) / max(len(x), 1))

    lag1_autocorr = _safe_corr(x[:-1], x[1:])
    lag2_autocorr = _safe_corr(x[:-2], x[2:]) if len(x) > 2 else 1.0

    dominant_period = float(estimate_characteristic_period(x, min_lag=2, max_lag=min(20, max(len(x) - 2, 2))))
    dominant_acf = _acf_at_lag(x, int(dominant_period))

    df = make_regression_df(x, lags=1, zero_guard=zero_guard)
    corr_omega_x = _safe_corr(df["omega"], df["X_n"]) if not df.empty else 0.0
    corr_omega_lag1 = _safe_corr(df["omega"], df["Lag_1"]) if not df.empty and "Lag_1" in df else 0.0

    third = max(len(x) // 3, 1)
    start_mean = float(np.mean(x[:third]))
    end_mean = float(np.mean(x[-third:]))
    max_x = float(np.max(x))
    collapse_ratio = end_mean / max(max_x, zero_guard)
    growth_ratio = end_mean / max(start_mean, zero_guard)

    return {
        "mean_x": mean_x,
        "std_x": std_x,
        "rel_range": rel_range,
        "omega_std": omega_std,
        "turning_rate": turning_rate,
        "unique_ratio": unique_ratio,
        "lag1_autocorr": lag1_autocorr,
        "lag2_autocorr": lag2_autocorr,
        "dominant_period": dominant_period,
        "dominant_acf": dominant_acf,
        "corr_omega_x": corr_omega_x,
        "corr_omega_lag1": corr_omega_lag1,
        "start_mean": start_mean,
        "end_mean": end_mean,
        "max_x": max_x,
        "collapse_ratio": collapse_ratio,
        "growth_ratio": growth_ratio,
    }


def oracle_regime_label(window: np.ndarray, regime_family: str, degenerate: bool) -> str:
    """Assign the synthetic ground-truth regime label for a window."""
    x = np.asarray(window, dtype=float)
    third = max(len(x) // 3, 1)
    start_mean = float(np.mean(x[:third]))
    end_mean = float(np.mean(x[-third:]))
    max_x = float(np.max(x))
    collapse = max_x > 0.4 and end_mean / max(max_x, 1e-12) < 0.2 and end_mean < 0.4 * max(start_mean, 1e-12)
    if collapse:
        return "collapse"
    if degenerate:
        return "plateau_degenerate"
    return regime_family


def predict_regime_label(features: dict[str, float]) -> str:
    """Predict a regime label using only simple diagnostics, before regression."""
    if (
        features["collapse_ratio"] < 0.05
        and features["growth_ratio"] < 0.1
        and features["rel_range"] > 5.0
        and features["std_x"] > 0.1
    ):
        return "collapse"

    if (
        features["rel_range"] < 0.08
        or features["std_x"] < 0.02 * max(features["mean_x"], 1e-12)
        or features["omega_std"] < 0.015
    ):
        return "plateau_degenerate"

    if features["turning_rate"] > 0.35:
        if features["dominant_acf"] > 0.65 and features["unique_ratio"] < 0.95:
            return "stable_cycle"
        return "chaotic_informative"

    if abs(features["corr_omega_lag1"]) > max(0.25, 0.85 * abs(features["corr_omega_x"])):
        return "growth_with_memory"

    return "growth_no_memory"


def _safe_corr(x: Any, y: Any) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) < 2 or len(b) < 2 or np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _acf_at_lag(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return 0.0
    centered = x - np.mean(x)
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(centered[:-lag], centered[lag:]) / denom)


def _generate_case_series(case: dict[str, Any], steps: int, x0: float = 1e-4) -> np.ndarray:
    model = case["model"]
    if model == "base":
        return generate_base_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "delay":
        return generate_delay_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "mixed":
        return generate_mixed_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    raise ValueError("Unsupported model")


def _stepwise_window_metrics(window: np.ndarray, lags: int, true_predictors: set[str]) -> dict[str, Any]:
    df = make_regression_df(np.asarray(window, dtype=float), lags=lags)
    if len(df) < 8:
        return {
            "selected": [],
            "false_lag_rate": np.nan,
            "hit_rate": np.nan,
            "false_lag_count": np.nan,
            "selected_count": 0.0,
            "no_model": 1,
            "b_sum": np.nan,
            "R2_enter": np.nan,
        }

    y = df["omega"]
    X_mat = df.drop(columns=["omega"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        selected = stepwise_selection(X_mat, y)
        enter_model, _ = fit_enter_with_beta(X_mat, y)
    lag_columns = [f"Lag_{idx}" for idx in range(1, lags + 1) if f"Lag_{idx}" in enter_model.params.index]
    b_sum = float(sum(float(enter_model.params.get(col, 0.0)) for col in lag_columns))

    row = {
        "selected": selected,
        "no_model": 1 if not selected else 0,
        "b_sum": b_sum,
        "R2_enter": float(enter_model.rsquared),
    }
    row.update(score_selected_features(selected, true_predictors))
    return row


def run_plan_g_experiment(
    *,
    steps: int = 180,
    window: int = 25,
    lags: int = 10,
) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    for case in PLAN_G_CASES:
        series = _generate_case_series(case, steps=steps)
        for start in range(0, len(series) - window):
            end = start + window
            block = np.asarray(series[start:end], dtype=float)
            degenerate = detect_degenerate_window(block)
            features = extract_window_diagnostics(block)
            true_label = oracle_regime_label(block, regime_family=case["regime_family"], degenerate=degenerate)
            predicted_label = predict_regime_label(features)
            stepwise_metrics = _stepwise_window_metrics(block, lags=lags, true_predictors=case["true_predictors"])
            row = {
                "model": case["model"],
                "case": case["case"],
                "regime_family": case["regime_family"],
                "start": start,
                "end": end,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "true_tool": TOOL_BY_LABEL[true_label],
                "recommended_tool": TOOL_BY_LABEL[predicted_label],
                "true_regression_admissible": int(true_label in REGRESSION_ADMISSIBLE),
                "predicted_regression_admissible": int(predicted_label in REGRESSION_ADMISSIBLE),
            }
            row.update(features)
            row.update(stepwise_metrics)
            rows.append(row)

    window_diagnostics = pd.DataFrame(rows)

    classification_summary = (
        window_diagnostics.groupby(["model", "case"], as_index=False)
        .agg(
            windows=("true_label", "size"),
            regime_accuracy=("true_label", lambda s: float(np.mean(s == window_diagnostics.loc[s.index, "predicted_label"]))),
            tool_accuracy=("true_tool", lambda s: float(np.mean(s == window_diagnostics.loc[s.index, "recommended_tool"]))),
            bad_window_recall=("true_regression_admissible", lambda s: _binary_recall(1 - s.to_numpy(), 1 - window_diagnostics.loc[s.index, "predicted_regression_admissible"].to_numpy())),
            informative_retention=("true_regression_admissible", lambda s: _informative_retention(s.to_numpy(), window_diagnostics.loc[s.index, "predicted_regression_admissible"].to_numpy())),
        )
        .sort_values(["model", "case"])
        .reset_index(drop=True)
    )

    confusion_matrix = (
        pd.crosstab(window_diagnostics["true_label"], window_diagnostics["predicted_label"], dropna=False)
        .reset_index()
        .rename(columns={"true_label": "true_label"})
    )

    downstream_rows = []
    for model, part in window_diagnostics.groupby("model"):
        for scope, mask in (
            ("all_windows", np.ones(len(part), dtype=bool)),
            ("oracle_filtered", part["true_regression_admissible"].to_numpy(dtype=bool)),
            ("predicted_filtered", part["predicted_regression_admissible"].to_numpy(dtype=bool)),
        ):
            subset = part.loc[mask].copy()
            downstream_rows.append(
                {
                    "model": model,
                    "scope": scope,
                    "windows": int(len(subset)),
                    "false_lag_rate": float(subset["false_lag_rate"].dropna().mean()) if len(subset) else np.nan,
                    "hit_rate": float(subset["hit_rate"].dropna().mean()) if len(subset) else np.nan,
                    "no_model_share": float(subset["no_model"].mean()) if len(subset) else np.nan,
                    "std_b_sum": float(subset["b_sum"].dropna().std(ddof=0)) if len(subset["b_sum"].dropna()) else np.nan,
                }
            )
    downstream_summary = pd.DataFrame(downstream_rows)

    tool_summary = (
        window_diagnostics.groupby(["true_tool", "recommended_tool"], as_index=False)
        .size()
        .rename(columns={"size": "windows"})
        .sort_values(["true_tool", "recommended_tool"])
        .reset_index(drop=True)
    )

    overall_summary = pd.DataFrame(
        [
            {
                "regime_accuracy": float(np.mean(window_diagnostics["true_label"] == window_diagnostics["predicted_label"])),
                "tool_accuracy": float(np.mean(window_diagnostics["true_tool"] == window_diagnostics["recommended_tool"])),
                "bad_window_recall": _binary_recall(
                    1 - window_diagnostics["true_regression_admissible"].to_numpy(),
                    1 - window_diagnostics["predicted_regression_admissible"].to_numpy(),
                ),
                "informative_retention": _informative_retention(
                    window_diagnostics["true_regression_admissible"].to_numpy(),
                    window_diagnostics["predicted_regression_admissible"].to_numpy(),
                ),
            }
        ]
    )

    return {
        "window_diagnostics": window_diagnostics,
        "classification_summary": classification_summary,
        "downstream_summary": downstream_summary,
        "confusion_matrix": confusion_matrix,
        "tool_summary": tool_summary,
        "overall_summary": overall_summary,
    }


def _binary_recall(truth: np.ndarray, predicted: np.ndarray) -> float:
    truth_arr = np.asarray(truth, dtype=bool)
    pred_arr = np.asarray(predicted, dtype=bool)
    positives = int(np.sum(truth_arr))
    if positives == 0:
        return np.nan
    return float(np.sum(truth_arr & pred_arr) / positives)


def _informative_retention(truth: np.ndarray, predicted: np.ndarray) -> float:
    truth_arr = np.asarray(truth, dtype=bool)
    pred_arr = np.asarray(predicted, dtype=bool)
    informative = int(np.sum(truth_arr))
    if informative == 0:
        return np.nan
    return float(np.sum(truth_arr & pred_arr) / informative)


def save_plan_g_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name, frame in results.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    metadata = {
        "files": [f"{name}.csv" for name in results],
        "n_windows": int(len(results["window_diagnostics"])),
    }
    (output / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
