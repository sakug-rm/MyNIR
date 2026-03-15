from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nonlinear_lab.direct_identification import (
    fit_direct_identification,
    rolling_direct_identification,
    summarize_parameter_errors,
)
from nonlinear_lab.features import growth_rate
from nonlinear_lab.models import (
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
)


BASE_CASES = [
    {"case": "base_a_0_8", "a": 0.8, "k": 1.0, "regime": "устойчивое насыщение"},
    {"case": "base_a_1_5", "a": 1.5, "k": 1.0, "regime": "быстрое насыщение"},
    {"case": "base_a_2_2", "a": 2.2, "k": 1.0, "regime": "затухающие колебания"},
    {"case": "base_a_2_52", "a": 2.52, "k": 1.0, "regime": "предтурбулентный режим"},
    {"case": "base_a_2_8", "a": 2.8, "k": 1.0, "regime": "хаотический режим"},
]

DELAY_CASES = [
    {"case": "delay_g_0_2", "g": 0.2, "k": 1.0, "regime": "устойчивое насыщение"},
    {"case": "delay_g_0_8", "g": 0.8, "k": 1.0, "regime": "колебательный рост"},
    {"case": "delay_g_1_05", "g": 1.05, "k": 1.0, "regime": "циклический режим"},
    {"case": "delay_g_1_25", "g": 1.25, "k": 1.0, "regime": "усиленный цикл"},
    {"case": "delay_g_1_6", "g": 1.6, "k": 1.0, "regime": "турбулентный режим"},
]

MIXED_CASES = [
    {"case": "mixed_q_1_5_gamma_0_5", "q": 1.5, "gamma": 0.5, "k": 1.0, "regime": "смешанная память, устойчивый"},
    {"case": "mixed_q_2_8_gamma_0_5", "q": 2.8, "gamma": 0.5, "k": 1.0, "regime": "смешанная память, цикл"},
    {"case": "mixed_q_3_5_gamma_0_5", "q": 3.5, "gamma": 0.5, "k": 1.0, "regime": "смешанная память, турбулентный"},
    {"case": "mixed_q_2_8_gamma_0_1", "q": 2.8, "gamma": 0.1, "k": 1.0, "regime": "слабая память"},
    {"case": "mixed_q_1_5_gamma_neg_0_2", "q": 1.5, "gamma": -0.2, "k": 1.0, "regime": "отрицательная память, умеренная"},
    {"case": "mixed_q_1_5_gamma_neg_0_5", "q": 1.5, "gamma": -0.5, "k": 1.0, "regime": "отрицательная память, сильная"},
    {"case": "mixed_q_1_5_gamma_neg_0_8", "q": 1.5, "gamma": -0.8, "k": 1.0, "regime": "отрицательная память, очень сильная"},
]


def _safe_div(num: float, den: float, zero_guard: float = 1e-12) -> float:
    return num / (den if abs(den) > zero_guard else zero_guard)


def _structural_dataset(series: np.ndarray, model_name: str, zero_guard: float = 1e-12) -> tuple[pd.DataFrame, pd.Series]:
    x = np.asarray(series, dtype=float)
    model = model_name.lower()

    if model == "base":
        y = pd.Series(growth_rate(x, zero_guard=zero_guard))
        X = pd.DataFrame({"X_n": x[:-1]})
        return X, y

    if model == "delay":
        rows = []
        target = []
        for i in range(1, len(x) - 1):
            rows.append({"Lag_1": x[i - 1]})
            target.append(_safe_div(x[i + 1] - x[i], x[i], zero_guard=zero_guard))
        return pd.DataFrame(rows), pd.Series(target)

    if model == "mixed":
        rows = []
        target = []
        for i in range(1, len(x) - 1):
            rows.append({"X_n": x[i], "Lag_1": x[i - 1]})
            target.append(_safe_div(x[i + 1] - x[i], x[i], zero_guard=zero_guard))
        return pd.DataFrame(rows), pd.Series(target)

    raise ValueError("model_name must be one of: base, delay, mixed")


def fit_structural_regression(window: np.ndarray, model_name: str, zero_guard: float = 1e-12) -> dict[str, Any]:
    """Estimate structural parameters via OLS on the model-consistent regressors."""
    X, y = _structural_dataset(window, model_name=model_name, zero_guard=zero_guard)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    const = float(model.params.get("const", np.nan))

    if model_name == "base":
        b_x = float(model.params.get("X_n", np.nan))
        a = -b_x
        k = const / a if np.isfinite(a) and abs(a) > zero_guard else np.nan
        valid = np.isfinite(a) and np.isfinite(k) and a > 0.0 and k > 0.0
        return {"model_name": "base", "valid": valid, "a": a, "k": k, "alpha": const, "R2": float(model.rsquared)}

    if model_name == "delay":
        b_lag = float(model.params.get("Lag_1", np.nan))
        g = -b_lag
        k = const / g if np.isfinite(g) and abs(g) > zero_guard else np.nan
        valid = np.isfinite(g) and np.isfinite(k) and g > 0.0 and k > 0.0
        return {"model_name": "delay", "valid": valid, "g": g, "k": k, "alpha": const, "R2": float(model.rsquared)}

    if model_name == "mixed":
        b_x = float(model.params.get("X_n", np.nan))
        b_lag = float(model.params.get("Lag_1", np.nan))
        q = -b_x
        gamma = b_lag / b_x if np.isfinite(b_x) and abs(b_x) > zero_guard else np.nan
        k = const / q if np.isfinite(q) and abs(q) > zero_guard else np.nan
        valid = np.isfinite(q) and np.isfinite(gamma) and np.isfinite(k) and q > 0.0 and k > 0.0
        return {
            "model_name": "mixed",
            "valid": valid,
            "q": q,
            "gamma": gamma,
            "k": k,
            "alpha": const,
            "R2": float(model.rsquared),
        }

    raise ValueError("model_name must be one of: base, delay, mixed")


def rolling_structural_regression(
    series: np.ndarray,
    model_name: str,
    window_size: int = 25,
    zero_guard: float = 1e-12,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    x = np.asarray(series, dtype=float)
    for start in range(0, len(x) - window_size + 1):
        end = start + window_size
        result = fit_structural_regression(x[start:end], model_name=model_name, zero_guard=zero_guard)
        row = {"start": start, "end": end}
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)


def add_multiplicative_noise(
    series: np.ndarray,
    sigma: float = 0.02,
    seed: int = 0,
    clip_min: float = 1e-12,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observed = np.asarray(series, dtype=float) * np.exp(rng.normal(loc=0.0, scale=sigma, size=len(series)))
    return np.clip(observed, clip_min, None)


def detect_degenerate_window(
    window: np.ndarray | list[float],
    relative_range_threshold: float = 0.01,
    std_threshold: float = 1e-4,
) -> bool:
    """Flag windows that are nearly constant and therefore weakly informative."""
    x = np.asarray(window, dtype=float)
    if len(x) < 2:
        return True
    mean_level = max(float(np.mean(np.abs(x))), 1e-12)
    rel_range = float((np.max(x) - np.min(x)) / mean_level)
    std = float(np.std(x, ddof=0))
    return rel_range <= relative_range_threshold or std <= std_threshold


def label_segments(df: pd.DataFrame, total_length: int) -> pd.DataFrame:
    out = df.copy()
    midpoint = 0.5 * (out["start"] + out["end"])
    rel = midpoint / max(total_length - 1, 1)
    out["segment"] = np.where(rel < 1.0 / 3.0, "early", np.where(rel < 2.0 / 3.0, "middle", "late"))
    return out


def _truth_for_case(model_name: str, case: dict[str, Any]) -> dict[str, float]:
    if model_name == "base":
        return {"a": float(case["a"]), "k": float(case["k"])}
    if model_name == "delay":
        return {"g": float(case["g"]), "k": float(case["k"])}
    if model_name == "mixed":
        return {"q": float(case["q"]), "gamma": float(case["gamma"]), "k": float(case["k"])}
    raise ValueError("model_name must be one of: base, delay, mixed")


def _model_cases(model_name: str) -> list[dict[str, Any]]:
    if model_name == "base":
        return BASE_CASES
    if model_name == "delay":
        return DELAY_CASES
    if model_name == "mixed":
        return MIXED_CASES
    raise ValueError("model_name must be one of: base, delay, mixed")


def _generate_series(model_name: str, case: dict[str, Any], steps: int) -> np.ndarray:
    if model_name == "base":
        return generate_base_process(a=case["a"], k=case["k"], steps=steps, clip_max=None)
    if model_name == "delay":
        return generate_delay_process(g=case["g"], steps=steps, clip_max=None)
    if model_name == "mixed":
        return generate_mixed_process(q=case["q"], gamma=case["gamma"], steps=steps, clip_max=None)
    raise ValueError("model_name must be one of: base, delay, mixed")


def _predict_omega(series: np.ndarray, model_name: str, params: dict[str, Any]) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    model = model_name.lower()

    if model == "base":
        a = params["a"]
        k = params["k"]
        return a * (k - x[:-1])

    if model == "delay":
        g = params["g"]
        k = params["k"]
        return g * (k - x[:-2])

    if model == "mixed":
        q = params["q"]
        gamma = params["gamma"]
        k = params["k"]
        return q * (k - x[1:-1] - gamma * x[:-2])

    raise ValueError("model_name must be one of: base, delay, mixed")


def _window_fit_mse(window: np.ndarray, model_name: str, estimator: str) -> dict[str, Any]:
    if estimator == "regression":
        fit = fit_structural_regression(window, model_name=model_name)
    elif estimator == "direct":
        fit = fit_direct_identification(window, model_name=model_name)
    else:
        raise ValueError("estimator must be 'direct' or 'regression'")

    if not fit.get("valid", False):
        return {"valid": False, "mse": float("inf"), "fit": fit}

    omega_true = np.asarray(growth_rate(np.asarray(window, dtype=float)))
    omega_pred = _predict_omega(np.asarray(window, dtype=float), model_name=model_name, params=fit)

    if model_name == "base":
        target = omega_true
    else:
        target = omega_true[1:]

    mse = float(np.mean((target - omega_pred) ** 2))
    return {"valid": True, "mse": mse, "fit": fit}


def identify_best_model(window: np.ndarray, estimator: str = "regression") -> dict[str, Any]:
    """Choose the structural family with the lowest local omega-prediction error."""
    candidates = {}
    for model_name in ("base", "delay", "mixed"):
        min_points = {"base": 3, "delay": 4, "mixed": 5}[model_name]
        if len(window) < min_points:
            continue
        candidates[model_name] = _window_fit_mse(np.asarray(window, dtype=float), model_name=model_name, estimator=estimator)

    valid = {name: payload for name, payload in candidates.items() if payload["valid"]}
    if not valid:
        return {"best_model": None, "best_mse": float("inf"), "candidate_mse": {k: v["mse"] for k, v in candidates.items()}}

    best_model = min(valid, key=lambda name: valid[name]["mse"])
    return {
        "best_model": best_model,
        "best_mse": valid[best_model]["mse"],
        "candidate_mse": {k: v["mse"] for k, v in candidates.items()},
    }


def _summarize_case(
    estimates: pd.DataFrame,
    truth: dict[str, float],
    model_name: str,
    case: dict[str, Any],
    method: str,
    observation: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment, segment_df in estimates.groupby("segment"):
        summary = summarize_parameter_errors(segment_df, truth=truth)
        row = {
            "model": model_name,
            "case": case["case"],
            "regime": case["regime"],
            "segment": segment,
            "method": method,
            "observation": observation,
            "valid_share": summary["valid_share"],
        }
        for key, value in summary.items():
            if key != "valid_share":
                row[key] = value
        if "R2" in segment_df.columns:
            row["mean_R2"] = float(segment_df.loc[segment_df["valid"], "R2"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _diagnostic_rows(
    estimates: pd.DataFrame,
    original_series: np.ndarray,
    model_name: str,
    case: dict[str, Any],
    method: str,
    observation: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment, segment_df in estimates.groupby("segment"):
        degenerate_truth = []
        predicted_degenerate = []
        recognized = []
        for _, row in segment_df.iterrows():
            window = np.asarray(original_series[int(row["start"]): int(row["end"])], dtype=float)
            is_degenerate = detect_degenerate_window(window)
            degenerate_truth.append(is_degenerate)
            predicted_degenerate.append(not bool(row.get("valid", False)))

            if observation == "clean":
                identified = identify_best_model(window, estimator=method)
                recognized.append(identified["best_model"] == model_name)

        degenerate_truth_arr = np.asarray(degenerate_truth, dtype=bool)
        predicted_arr = np.asarray(predicted_degenerate, dtype=bool)

        tp = int(np.sum(predicted_arr & degenerate_truth_arr))
        fp = int(np.sum(predicted_arr & ~degenerate_truth_arr))
        tn = int(np.sum(~predicted_arr & ~degenerate_truth_arr))
        fn = int(np.sum(~predicted_arr & degenerate_truth_arr))

        recall = tp / (tp + fn) if (tp + fn) else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        informative_retention = tn / max(int(np.sum(~degenerate_truth_arr)), 1)

        row = {
            "model": model_name,
            "case": case["case"],
            "regime": case["regime"],
            "segment": segment,
            "method": method,
            "observation": observation,
            "degenerate_truth_share": float(np.mean(degenerate_truth_arr)) if len(degenerate_truth_arr) else np.nan,
            "degenerate_recall": recall,
            "degenerate_precision": precision,
            "informative_specificity": specificity,
            "informative_retention": informative_retention,
        }
        if observation == "clean":
            row["family_accuracy"] = float(np.mean(recognized)) if recognized else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_plan_a_experiment(
    steps: int = 180,
    regression_window: int = 25,
    noise_sigma: float = 0.02,
    noise_seed: int = 7,
) -> dict[str, pd.DataFrame]:
    raw_rows: list[pd.DataFrame] = []
    summary_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[pd.DataFrame] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for model_name in ("base", "delay", "mixed"):
            for case_index, case in enumerate(_model_cases(model_name)):
                clean = _generate_series(model_name, case=case, steps=steps)
                noisy = add_multiplicative_noise(clean, sigma=noise_sigma, seed=noise_seed + case_index)
                truth = _truth_for_case(model_name, case)

                for observation, series in (("clean", clean), ("noisy", noisy)):
                    direct = rolling_direct_identification(series, model_name=model_name)
                    direct = label_segments(direct, total_length=len(series))
                    direct["model"] = model_name
                    direct["case"] = case["case"]
                    direct["regime"] = case["regime"]
                    direct["method"] = "direct"
                    direct["observation"] = observation
                    raw_rows.append(direct)
                    summary_rows.append(
                        _summarize_case(
                            direct, truth=truth, model_name=model_name, case=case, method="direct", observation=observation
                        )
                    )
                    diagnostic_rows.append(
                        _diagnostic_rows(
                            direct,
                            original_series=series,
                            model_name=model_name,
                            case=case,
                            method="direct",
                            observation=observation,
                        )
                    )

                    regression = rolling_structural_regression(series, model_name=model_name, window_size=regression_window)
                    regression = label_segments(regression, total_length=len(series))
                    regression["model"] = model_name
                    regression["case"] = case["case"]
                    regression["regime"] = case["regime"]
                    regression["method"] = "regression"
                    regression["observation"] = observation
                    raw_rows.append(regression)
                    summary_rows.append(
                        _summarize_case(
                            regression,
                            truth=truth,
                            model_name=model_name,
                            case=case,
                            method="regression",
                            observation=observation,
                        )
                    )
                    diagnostic_rows.append(
                        _diagnostic_rows(
                            regression,
                            original_series=series,
                            model_name=model_name,
                            case=case,
                            method="regression",
                            observation=observation,
                        )
                    )

    raw = pd.concat(raw_rows, ignore_index=True, sort=False)
    summary = pd.concat(summary_rows, ignore_index=True, sort=False)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True, sort=False)

    overall = (
        summary.groupby(["model", "segment", "method", "observation"], dropna=False)
        .agg(
            cases=("case", "nunique"),
            valid_share=("valid_share", "mean"),
            mean_R2=("mean_R2", "mean"),
            a_mae=("a_mae", "mean"),
            g_mae=("g_mae", "mean"),
            q_mae=("q_mae", "mean"),
            gamma_mae=("gamma_mae", "mean"),
            k_mae=("k_mae", "mean"),
        )
        .reset_index()
    )

    diagnostic_overall = (
        diagnostics.groupby(["model", "segment", "method", "observation"], dropna=False)
        .agg(
            cases=("case", "nunique"),
            degenerate_truth_share=("degenerate_truth_share", "mean"),
            degenerate_recall=("degenerate_recall", "mean"),
            degenerate_precision=("degenerate_precision", "mean"),
            informative_specificity=("informative_specificity", "mean"),
            informative_retention=("informative_retention", "mean"),
            family_accuracy=("family_accuracy", "mean"),
        )
        .reset_index()
    )

    return {
        "raw_estimates": raw,
        "case_summary": summary,
        "overall_summary": overall,
        "diagnostic_summary": diagnostics,
        "diagnostic_overall": diagnostic_overall,
    }


def run_plan_a_noise_sweep(
    noise_levels: list[float] | tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05),
    steps: int = 180,
    regression_window: int = 25,
    noise_seed: int = 7,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for noise_level in noise_levels:
        observation = "clean" if np.isclose(noise_level, 0.0) else "noisy"
        result = run_plan_a_experiment(
            steps=steps,
            regression_window=regression_window,
            noise_sigma=float(noise_level),
            noise_seed=noise_seed,
        )
        summary = result["overall_summary"].copy()
        summary = summary[summary["observation"] == observation].copy()
        summary["noise_level"] = float(noise_level)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False)


def save_plan_a_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        df.to_csv(output / f"{name}.csv", index=False)

    metadata = {
        "files": [f"{name}.csv" for name in results],
        "n_raw_rows": int(len(results["raw_estimates"])),
        "n_summary_rows": int(len(results["case_summary"])),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    return output
