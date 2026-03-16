from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.models import (
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
)
from nonlinear_lab.plan_a_experiment import detect_degenerate_window
from nonlinear_lab.regression import fit_enter_with_beta, stepwise_selection


PLAN_D_CASES = [
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
        "case": "delay_g_1_25",
        "regime": "усиленный цикл",
        "params": {"g": 1.25},
        "true_predictors": {"Lag_1"},
    },
    {
        "model": "mixed",
        "case": "mixed_q_1_5_gamma_0_5",
        "regime": "смешанная память, умеренный режим",
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
]

METHOD_ORDER = ["enter", "stepwise", "ridge", "lasso", "elastic_net"]


def _generate_case_series(case: dict[str, Any], steps: int, x0: float = 1e-4) -> np.ndarray:
    model = case["model"]
    if model == "base":
        return generate_base_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "delay":
        return generate_delay_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "mixed":
        return generate_mixed_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    raise ValueError("Unsupported model")


def true_coefficients_for_case(model: str, params: dict[str, float]) -> dict[str, float]:
    if model == "base":
        return {"X_n": -float(params["a"])}
    if model == "delay":
        return {"Lag_1": -float(params["g"])}
    if model == "mixed":
        q = float(params["q"])
        gamma = float(params["gamma"])
        return {"X_n": -q, "Lag_1": -q * gamma}
    raise ValueError("Unsupported model")


def active_features_from_coefficients(coefficients: pd.Series, threshold: float = 0.05) -> list[str]:
    usable = coefficients.fillna(0.0)
    active = usable[usable.abs() >= threshold]
    return active.abs().sort_values(ascending=False).index.tolist()


def score_support(selected: list[str], true_predictors: set[str]) -> dict[str, float]:
    selected_set = set(selected)
    if not selected:
        return {
            "support_size": 0.0,
            "true_lag_count": 0.0,
            "false_lag_count": 0.0,
            "hit_rate": 0.0,
            "false_lag_rate": 0.0,
            "no_model": 1.0,
        }
    true_count = float(len(selected_set & true_predictors))
    false_count = float(len(selected_set - true_predictors))
    return {
        "support_size": float(len(selected)),
        "true_lag_count": true_count,
        "false_lag_count": false_count,
        "hit_rate": true_count / max(len(true_predictors), 1),
        "false_lag_rate": false_count / max(len(selected), 1),
        "no_model": 0.0,
    }


def _split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    validation_share: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = max(int(np.floor(len(X) * (1.0 - validation_share))), 8)
    split_idx = min(split_idx, len(X) - 3)
    if split_idx <= 0 or split_idx >= len(X):
        split_idx = len(X)
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_valid = X.iloc[split_idx:].copy()
    y_valid = y.iloc[split_idx:].copy()
    return X_train, X_valid, y_train, y_valid


def _validation_r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    y_true_values = y_true.to_numpy(dtype=float)
    ss_res = float(np.sum((y_true_values - y_pred) ** 2))
    ss_tot = float(np.sum((y_true_values - y_true_values.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _full_zero_series(columns: list[str]) -> pd.Series:
    return pd.Series(0.0, index=columns, dtype=float)


def _fit_intercept_only(y_train: pd.Series, X_valid: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    mean_value = float(y_train.mean()) if len(y_train) else 0.0
    return {
        "selected": [],
        "coef_raw": _full_zero_series(columns),
        "coef_std": _full_zero_series(columns),
        "intercept": mean_value,
        "validation_pred": np.repeat(mean_value, len(X_valid)),
        "alpha": np.nan,
        "l1_ratio": np.nan,
    }


def _standardize_train_valid(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, float]:
    x_mean = X_train.mean()
    x_std = X_train.std(ddof=0).replace(0.0, np.nan)
    good_cols = x_std.dropna().index.tolist()
    if not good_cols:
        raise ValueError("No varying columns available for standardization")

    X_train_scaled = (X_train[good_cols] - x_mean[good_cols]) / x_std[good_cols]
    X_valid_scaled = (X_valid[good_cols] - x_mean[good_cols]) / x_std[good_cols]

    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    if not np.isfinite(y_std) or y_std <= 1e-12:
        raise ValueError("Target variance is too small for standardized regression")
    y_train_scaled = (y_train - y_mean) / y_std
    return X_train_scaled, X_valid_scaled, y_train_scaled, x_std[good_cols], x_mean[good_cols], y_mean, y_std


def _restore_coefficients(
    scaled_coef: pd.Series,
    x_std: pd.Series,
    x_mean: pd.Series,
    y_mean: float,
    y_std: float,
    all_columns: list[str],
) -> tuple[pd.Series, float]:
    coef_raw = pd.Series(0.0, index=all_columns, dtype=float)
    restored = y_std * scaled_coef / x_std.loc[scaled_coef.index]
    coef_raw.loc[scaled_coef.index] = restored.astype(float)
    intercept = float(y_mean - np.dot(coef_raw.loc[scaled_coef.index], x_mean.loc[scaled_coef.index]))
    return coef_raw, intercept


def _fit_regularized_method(
    method: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
    active_threshold: float,
) -> dict[str, Any]:
    X_train_scaled, X_valid_scaled, y_train_scaled, x_std, x_mean, y_mean, y_std = _standardize_train_valid(
        X_train,
        X_valid,
        y_train,
    )
    cv = 3
    alphas = np.logspace(-3, 1, 8)

    if method == "ridge":
        estimator = RidgeCV(alphas=alphas, fit_intercept=True, cv=None)
    elif method == "lasso":
        estimator = LassoCV(alphas=alphas, fit_intercept=True, cv=cv, max_iter=20000, random_state=0)
    elif method == "elastic_net":
        estimator = ElasticNetCV(
            alphas=alphas,
            l1_ratio=[0.2, 0.5, 0.8, 0.95],
            fit_intercept=True,
            cv=cv,
            max_iter=20000,
            random_state=0,
        )
    else:
        raise ValueError("Unsupported regularized method")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        estimator.fit(X_train_scaled, y_train_scaled)

    coef_std = _full_zero_series(columns)
    scaled_coef = pd.Series(estimator.coef_, index=X_train_scaled.columns, dtype=float)
    coef_std.loc[scaled_coef.index] = scaled_coef
    coef_raw, intercept = _restore_coefficients(scaled_coef, x_std, x_mean, y_mean, y_std, columns)
    validation_pred = intercept + X_valid.to_numpy(dtype=float) @ coef_raw.to_numpy(dtype=float)
    selected = active_features_from_coefficients(coef_std, threshold=active_threshold)
    return {
        "selected": selected,
        "coef_raw": coef_raw,
        "coef_std": coef_std,
        "intercept": intercept,
        "validation_pred": validation_pred,
        "alpha": float(getattr(estimator, "alpha_", getattr(estimator, "alpha", np.nan))),
        "l1_ratio": float(getattr(estimator, "l1_ratio_", np.nan)),
    }


def _fit_enter_method(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
    active_threshold: float,
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model, beta = fit_enter_with_beta(X_train, y_train)
    params = model.params.drop(labels="const", errors="ignore")
    coef_raw = _full_zero_series(columns)
    coef_raw.loc[params.index] = params.astype(float)
    coef_std = _full_zero_series(columns)
    coef_std.loc[beta.index] = beta.astype(float)
    validation_pred = model.predict(sm.add_constant(X_valid, has_constant="add")).to_numpy(dtype=float)
    selected = active_features_from_coefficients(coef_std, threshold=active_threshold)
    return {
        "selected": selected,
        "coef_raw": coef_raw,
        "coef_std": coef_std,
        "intercept": float(model.params.get("const", 0.0)),
        "validation_pred": validation_pred,
        "alpha": np.nan,
        "l1_ratio": np.nan,
    }


def _fit_stepwise_method(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        selected = stepwise_selection(X_train, y_train, threshold_in=0.01, threshold_out=0.05)

    if not selected:
        return _fit_intercept_only(y_train, X_valid, columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = sm.OLS(y_train, sm.add_constant(X_train[selected], has_constant="add")).fit()
        _, beta = fit_enter_with_beta(X_train[selected], y_train)

    coef_raw = _full_zero_series(columns)
    coef_std = _full_zero_series(columns)
    params = model.params.drop(labels="const", errors="ignore")
    coef_raw.loc[params.index] = params.astype(float)
    coef_std.loc[beta.index] = beta.astype(float)
    validation_pred = model.predict(sm.add_constant(X_valid[selected], has_constant="add")).to_numpy(dtype=float)
    return {
        "selected": list(selected),
        "coef_raw": coef_raw,
        "coef_std": coef_std,
        "intercept": float(model.params.get("const", 0.0)),
        "validation_pred": validation_pred,
        "alpha": np.nan,
        "l1_ratio": np.nan,
    }


def _fit_method(
    method: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
    active_threshold: float,
) -> dict[str, Any]:
    if method == "enter":
        return _fit_enter_method(X_train, X_valid, y_train, columns, active_threshold)
    if method == "stepwise":
        return _fit_stepwise_method(X_train, X_valid, y_train, columns)
    return _fit_regularized_method(method, X_train, X_valid, y_train, columns, active_threshold)


def _coefficient_metrics(
    estimated: pd.Series,
    truth: pd.Series,
    support: list[str],
    true_predictors: set[str],
) -> dict[str, float]:
    aligned_est = estimated.reindex(truth.index, fill_value=0.0)
    errors = (aligned_est - truth).abs()
    true_errors = errors.loc[list(true_predictors)] if true_predictors else pd.Series(dtype=float)
    false_predictors = [name for name in truth.index if name not in true_predictors]
    false_mass = float(aligned_est.loc[false_predictors].abs().sum()) if false_predictors else 0.0

    sign_hits = []
    for predictor in true_predictors:
        est_value = float(aligned_est.get(predictor, 0.0))
        truth_value = float(truth.get(predictor, 0.0))
        sign_hits.append(float(np.sign(est_value) == np.sign(truth_value) and abs(est_value) > 1e-12))

    return {
        "coef_mae": float(errors.mean()),
        "coef_mae_true": float(true_errors.mean()) if len(true_errors) else 0.0,
        "false_coef_l1": false_mass,
        "sign_correct_share": float(np.mean(sign_hits)) if sign_hits else 1.0,
        "support_intersection": float(len(set(support) & true_predictors)),
    }


def run_plan_d_experiment(
    *,
    steps: int = 180,
    window: int = 25,
    lags: int = 10,
    active_threshold: float = 0.05,
    validation_share: float = 0.2,
    cases: list[dict[str, Any]] | None = None,
    window_stride: int = 5,
) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    all_columns = ["X_n"] + [f"Lag_{idx}" for idx in range(1, lags + 1)]

    for case in (cases or PLAN_D_CASES):
        truth = pd.Series(0.0, index=all_columns, dtype=float)
        truth_dict = true_coefficients_for_case(case["model"], case["params"])
        for name, value in truth_dict.items():
            truth.loc[name] = value

        series = _generate_case_series(case, steps=steps)
        for start in range(0, len(series) - window, window_stride):
            end = start + window
            block = np.asarray(series[start:end], dtype=float)
            if detect_degenerate_window(block):
                continue

            df = make_regression_df(block, lags=lags)
            if len(df) < 12:
                continue

            y = df["omega"]
            X_mat = df.drop(columns=["omega"]).reindex(columns=all_columns, fill_value=0.0)
            if float(y.std(ddof=0)) <= 1e-10:
                continue
            if (X_mat.std(ddof=0) > 0).sum() == 0:
                continue

            X_train, X_valid, y_train, y_valid = _split_train_validation(X_mat, y, validation_share=validation_share)
            if len(X_train) < 8:
                continue

            stage = "early" if start < (steps - window) / 3 else "late" if start >= 2 * (steps - window) / 3 else "middle"

            for method in METHOD_ORDER:
                try:
                    fitted = _fit_method(method, X_train, X_valid, y_train, all_columns, active_threshold)
                except Exception:
                    if method == "stepwise":
                        fitted = _fit_intercept_only(y_train, X_valid, all_columns)
                    else:
                        continue

                support_metrics = score_support(fitted["selected"], case["true_predictors"])
                coef_metrics = _coefficient_metrics(
                    fitted["coef_raw"],
                    truth,
                    fitted["selected"],
                    case["true_predictors"],
                )
                validation_r2 = _validation_r2(y_valid, fitted["validation_pred"])
                validation_mae = float(np.mean(np.abs(y_valid.to_numpy(dtype=float) - fitted["validation_pred"]))) if len(y_valid) else np.nan

                rows.append(
                    {
                        "model": case["model"],
                        "case": case["case"],
                        "regime": case["regime"],
                        "stage": stage,
                        "start": start,
                        "end": end,
                        "method": method,
                        "active_threshold": active_threshold,
                        "validation_r2": validation_r2,
                        "validation_mae": validation_mae,
                        "alpha": fitted["alpha"],
                        "l1_ratio": fitted["l1_ratio"],
                        **support_metrics,
                        **coef_metrics,
                    }
                )

    window_level = pd.DataFrame(rows)
    if window_level.empty:
        raise RuntimeError("Plan D produced no informative windows")

    case_summary = (
        window_level.groupby(["model", "case", "regime", "method"], as_index=False)
        .agg(
            windows=("method", "size"),
            support_size=("support_size", "mean"),
            true_lag_count=("true_lag_count", "mean"),
            false_lag_count=("false_lag_count", "mean"),
            hit_rate=("hit_rate", "mean"),
            false_lag_rate=("false_lag_rate", "mean"),
            no_model_share=("no_model", "mean"),
            coef_mae=("coef_mae", "mean"),
            coef_mae_true=("coef_mae_true", "mean"),
            false_coef_l1=("false_coef_l1", "mean"),
            sign_correct_share=("sign_correct_share", "mean"),
            validation_r2=("validation_r2", "mean"),
            validation_mae=("validation_mae", "mean"),
        )
        .sort_values(["model", "case", "method"])
        .reset_index(drop=True)
    )

    sign_summary = (
        window_level.groupby(["model", "method"], as_index=False)
        .agg(
            sign_correct_share=("sign_correct_share", "mean"),
            coef_mae_true=("coef_mae_true", "mean"),
            false_coef_l1=("false_coef_l1", "mean"),
        )
        .sort_values(["model", "method"])
        .reset_index(drop=True)
    )

    overall_summary = (
        window_level.groupby("method", as_index=False)
        .agg(
            windows=("method", "size"),
            support_size=("support_size", "mean"),
            true_lag_count=("true_lag_count", "mean"),
            false_lag_count=("false_lag_count", "mean"),
            hit_rate=("hit_rate", "mean"),
            false_lag_rate=("false_lag_rate", "mean"),
            no_model_share=("no_model", "mean"),
            coef_mae=("coef_mae", "mean"),
            coef_mae_true=("coef_mae_true", "mean"),
            false_coef_l1=("false_coef_l1", "mean"),
            sign_correct_share=("sign_correct_share", "mean"),
            validation_r2=("validation_r2", "mean"),
            validation_mae=("validation_mae", "mean"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )

    stage_summary = (
        window_level.groupby(["stage", "method"], as_index=False)
        .agg(
            false_lag_rate=("false_lag_rate", "mean"),
            hit_rate=("hit_rate", "mean"),
            coef_mae=("coef_mae", "mean"),
            sign_correct_share=("sign_correct_share", "mean"),
        )
        .sort_values(["stage", "method"])
        .reset_index(drop=True)
    )

    return {
        "window_level": window_level,
        "case_summary": case_summary,
        "overall_summary": overall_summary,
        "sign_summary": sign_summary,
        "stage_summary": stage_summary,
    }


def save_plan_d_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, frame in results.items():
        frame.to_csv(output_path / f"{name}.csv", index=False)

    summary = {
        row["method"]: {
            key: float(value)
            for key, value in row.items()
            if key != "method" and isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value)
        }
        for row in results["overall_summary"].to_dict(orient="records")
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return output_path
