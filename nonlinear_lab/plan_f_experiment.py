from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.models import (
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
)
from nonlinear_lab.plan_a_experiment import detect_degenerate_window
from nonlinear_lab.plan_c_experiment import compute_ranking_metrics
from nonlinear_lab.plan_d_experiment import active_features_from_coefficients, score_support
from nonlinear_lab.plan_h_experiment import compute_condition_number
from nonlinear_lab.regression import fit_enter_with_beta


PLAN_F_CASES = [
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

RHO_GRID = [0.8, 0.9, 0.95, 0.99]


def _generate_case_series(case: dict[str, Any], steps: int, x0: float = 1e-4) -> np.ndarray:
    model = case["model"]
    if model == "base":
        return generate_base_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "delay":
        return generate_delay_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    if model == "mixed":
        return generate_mixed_process(x0=x0, steps=steps, clip_max=None, **case["params"])
    raise ValueError("Unsupported model")


def _split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    validation_share: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = max(int(np.floor(len(X) * (1.0 - validation_share))), 8)
    split_idx = min(split_idx, len(X) - 3)
    if split_idx <= 0 or split_idx >= len(X):
        split_idx = len(X)
    return (
        X.iloc[:split_idx].copy(),
        X.iloc[split_idx:].copy(),
        y.iloc[:split_idx].copy(),
        y.iloc[split_idx:].copy(),
    )


def _validation_metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0:
        return float("nan"), float("nan")
    y_true_values = y_true.to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y_true_values - y_pred)))
    if len(y_true_values) < 2:
        return float("nan"), mae
    ss_res = float(np.sum((y_true_values - y_pred) ** 2))
    ss_tot = float(np.sum((y_true_values - y_true_values.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return r2, mae


def _mass_on_true(scores: pd.Series, true_predictors: set[str]) -> float:
    usable = scores.abs().fillna(0.0)
    total = float(usable.sum())
    if total <= 0:
        return 0.0
    true_mass = float(sum(float(usable.get(name, 0.0)) for name in true_predictors))
    return true_mass / total


def _effective_support(scores: pd.Series, coverage: float = 0.9) -> float:
    usable = scores.abs().sort_values(ascending=False)
    total = float(usable.sum())
    if total <= 0:
        return 0.0
    cumulative = usable.cumsum() / total
    return float(int(np.searchsorted(cumulative.to_numpy(dtype=float), coverage, side="left")) + 1)


def _component_dominance(loadings: np.ndarray) -> float:
    if loadings.size == 0:
        return 0.0
    row_sum = np.abs(loadings).sum(axis=1)
    row_max = np.abs(loadings).max(axis=1)
    good = row_sum > 0
    if not np.any(good):
        return 0.0
    return float(np.mean(row_max[good] / row_sum[good]))


def choose_component_count(explained_variance_ratio: np.ndarray, rho: float = 0.95) -> int:
    cumulative = np.cumsum(np.asarray(explained_variance_ratio, dtype=float))
    if cumulative.size == 0:
        return 0
    idx = int(np.searchsorted(cumulative, rho, side="left"))
    return min(idx + 1, len(cumulative))


def _fit_enter_window(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    true_predictors: set[str],
    active_threshold: float,
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
        _, beta = fit_enter_with_beta(X_train, y_train)
    coef = beta.reindex(X_train.columns).fillna(0.0)
    y_pred = model.predict(sm.add_constant(X_valid, has_constant="add"))
    selected = active_features_from_coefficients(coef, threshold=active_threshold)
    support = score_support(selected, true_predictors)
    ranking = compute_ranking_metrics(coef.abs(), true_predictors)
    r2, mae = _validation_metrics(y_valid, np.asarray(y_pred, dtype=float))
    return {
        "validation_r2": r2,
        "validation_mae": mae,
        "coef_std": coef,
        "selected": selected,
        "support": support,
        "ranking": ranking,
        "true_mass_share": _mass_on_true(coef, true_predictors),
        "effective_support_90": _effective_support(coef, coverage=0.9),
    }


def _fit_pcr_window(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    true_predictors: set[str],
    rho: float,
    active_threshold: float,
) -> dict[str, Any]:
    x_mean = X_train.mean()
    x_std = X_train.std(ddof=0).replace(0.0, np.nan)
    good_cols = x_std.dropna().index.tolist()
    if not good_cols:
        raise ValueError("No varying columns available for PCA")

    X_train_scaled = (X_train[good_cols] - x_mean[good_cols]) / x_std[good_cols]
    X_valid_scaled = (X_valid[good_cols] - x_mean[good_cols]) / x_std[good_cols]

    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    if not np.isfinite(y_std) or y_std <= 1e-12:
        raise ValueError("Target variance is too small for PCR")
    y_train_scaled = (y_train - y_mean) / y_std

    pca = PCA()
    z_train = pca.fit_transform(X_train_scaled)
    z_valid = pca.transform(X_valid_scaled)
    k = choose_component_count(pca.explained_variance_ratio_, rho=rho)
    z_train_k = z_train[:, :k]
    z_valid_k = z_valid[:, :k]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = sm.OLS(y_train_scaled, sm.add_constant(z_train_k, has_constant="add")).fit()

    gamma_values = pd.Series(model.params).iloc[1:].to_numpy(dtype=float)
    gamma = pd.Series(gamma_values, index=[f"PC_{idx}" for idx in range(1, k + 1)], dtype=float)
    beta_back = pd.Series(0.0, index=X_train.columns, dtype=float)
    beta_back.loc[good_cols] = pca.components_[:k].T @ gamma.to_numpy(dtype=float)
    y_pred_scaled = model.predict(sm.add_constant(z_valid_k, has_constant="add"))
    y_pred = y_mean + y_std * np.asarray(y_pred_scaled, dtype=float)
    selected = active_features_from_coefficients(beta_back, threshold=active_threshold)
    support = score_support(selected, true_predictors)
    ranking = compute_ranking_metrics(beta_back.abs(), true_predictors)
    r2, mae = _validation_metrics(y_valid, y_pred)

    retained_scores = z_train_k
    loadings = pca.components_[:k, :]
    true_mask = np.array([name in true_predictors for name in good_cols], dtype=bool)
    loading_true_share = 0.0
    if loadings.size > 0:
        denom = np.abs(loadings).sum(axis=1)
        numer = np.abs(loadings[:, true_mask]).sum(axis=1) if np.any(true_mask) else np.zeros(k)
        good = denom > 0
        if np.any(good):
            loading_true_share = float(np.mean(numer[good] / denom[good]))

    return {
        "validation_r2": r2,
        "validation_mae": mae,
        "coef_std": beta_back,
        "selected": selected,
        "support": support,
        "ranking": ranking,
        "true_mass_share": _mass_on_true(beta_back, true_predictors),
        "effective_support_90": _effective_support(beta_back, coverage=0.9),
        "retained_components": float(k),
        "explained_variance_share": float(np.sum(pca.explained_variance_ratio_[:k])),
        "cond_component_space": compute_condition_number(pd.DataFrame(retained_scores), standardize=False),
        "component_dominance": _component_dominance(loadings),
        "component_true_loading_share": loading_true_share,
        "pc1_share": float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) else 0.0,
        "all_explained_variance_ratio": pca.explained_variance_ratio_,
        "loadings": loadings,
        "good_cols": good_cols,
    }


def run_plan_f_experiment(
    *,
    steps: int = 180,
    window: int = 25,
    lags: int = 10,
    rho: float = 0.95,
    rho_grid: list[float] | None = None,
    validation_share: float = 0.2,
    active_threshold: float = 0.05,
) -> dict[str, pd.DataFrame]:
    rho_values = rho_grid or RHO_GRID
    rows: list[dict[str, Any]] = []
    rho_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []

    for case in PLAN_F_CASES:
        series = _generate_case_series(case, steps=steps)
        for start in range(0, len(series) - window):
            end = start + window
            block = np.asarray(series[start:end], dtype=float)
            if detect_degenerate_window(block):
                continue

            df = make_regression_df(block, lags=lags)
            if len(df) < 8:
                continue

            y = df["omega"]
            X_mat = df.drop(columns=["omega"])
            X_train, X_valid, y_train, y_valid = _split_train_validation(
                X_mat,
                y,
                validation_share=validation_share,
            )

            enter = _fit_enter_window(
                X_train,
                X_valid,
                y_train,
                y_valid,
                case["true_predictors"],
                active_threshold=active_threshold,
            )
            pcr = _fit_pcr_window(
                X_train,
                X_valid,
                y_train,
                y_valid,
                case["true_predictors"],
                rho=rho,
                active_threshold=active_threshold,
            )

            cond_original = compute_condition_number(X_train, standardize=True)
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
                    "true_predictor_count": float(len(case["true_predictors"])),
                    "cond_original": cond_original,
                    "cond_pcr": float(pcr["cond_component_space"]),
                    "cond_reduction_ratio": cond_original / max(float(pcr["cond_component_space"]), 1e-12),
                    "enter_validation_r2": float(enter["validation_r2"]),
                    "pcr_validation_r2": float(pcr["validation_r2"]),
                    "enter_validation_mae": float(enter["validation_mae"]),
                    "pcr_validation_mae": float(pcr["validation_mae"]),
                    "enter_true_mass_share": float(enter["true_mass_share"]),
                    "pcr_true_mass_share": float(pcr["true_mass_share"]),
                    "enter_support_90": float(enter["effective_support_90"]),
                    "pcr_support_90": float(pcr["effective_support_90"]),
                    "enter_pairwise": float(enter["ranking"]["pairwise_score"]),
                    "pcr_pairwise": float(pcr["ranking"]["pairwise_score"]),
                    "enter_topk": float(enter["ranking"]["topk_recall"]),
                    "pcr_topk": float(pcr["ranking"]["topk_recall"]),
                    "enter_false_lag_rate": float(enter["support"]["false_lag_rate"]),
                    "pcr_false_lag_rate": float(pcr["support"]["false_lag_rate"]),
                    "enter_hit_rate": float(enter["support"]["hit_rate"]),
                    "pcr_hit_rate": float(pcr["support"]["hit_rate"]),
                    "retained_components": float(pcr["retained_components"]),
                    "explained_variance_share": float(pcr["explained_variance_share"]),
                    "pc1_share": float(pcr["pc1_share"]),
                    "component_dominance": float(pcr["component_dominance"]),
                    "component_true_loading_share": float(pcr["component_true_loading_share"]),
                }
            )

            for component_idx, row_loadings in enumerate(pcr["loadings"], start=1):
                total = float(np.sum(np.abs(row_loadings)))
                for variable, loading in zip(pcr["good_cols"], row_loadings, strict=True):
                    loading_rows.append(
                        {
                            "model": case["model"],
                            "case": case["case"],
                            "regime": case["regime"],
                            "start": start,
                            "stage": stage,
                            "component": component_idx,
                            "variable": variable,
                            "loading": float(loading),
                            "abs_loading": float(abs(loading)),
                            "loading_share": 0.0 if total <= 0 else float(abs(loading) / total),
                            "is_true_predictor": int(variable in case["true_predictors"]),
                        }
                    )

            for rho_item in rho_values:
                pcr_rho = _fit_pcr_window(
                    X_train,
                    X_valid,
                    y_train,
                    y_valid,
                    case["true_predictors"],
                    rho=float(rho_item),
                    active_threshold=active_threshold,
                )
                rho_rows.append(
                    {
                        "model": case["model"],
                        "case": case["case"],
                        "regime": case["regime"],
                        "start": start,
                        "stage": stage,
                        "rho": float(rho_item),
                        "retained_components": float(pcr_rho["retained_components"]),
                        "explained_variance_share": float(pcr_rho["explained_variance_share"]),
                        "validation_r2": float(pcr_rho["validation_r2"]),
                        "validation_mae": float(pcr_rho["validation_mae"]),
                        "true_mass_share": float(pcr_rho["true_mass_share"]),
                        "support_90": float(pcr_rho["effective_support_90"]),
                        "pairwise_score": float(pcr_rho["ranking"]["pairwise_score"]),
                        "false_lag_rate": float(pcr_rho["support"]["false_lag_rate"]),
                        "component_dominance": float(pcr_rho["component_dominance"]),
                    }
                )

    window_level = pd.DataFrame(rows)
    rho_curve = pd.DataFrame(rho_rows)
    component_loadings = pd.DataFrame(loading_rows)

    case_rows: list[dict[str, Any]] = []
    for (model, case_name, regime), frame in window_level.groupby(["model", "case", "regime"], as_index=False):
        case_rows.append(
            {
                "model": model,
                "case": case_name,
                "regime": regime,
                "windows": float(len(frame)),
                "cond_original": float(frame["cond_original"].mean()),
                "cond_pcr": float(frame["cond_pcr"].mean()),
                "cond_reduction_ratio": float(frame["cond_reduction_ratio"].mean()),
                "enter_validation_r2": float(frame["enter_validation_r2"].mean()),
                "pcr_validation_r2": float(frame["pcr_validation_r2"].mean()),
                "enter_true_mass_share": float(frame["enter_true_mass_share"].mean()),
                "pcr_true_mass_share": float(frame["pcr_true_mass_share"].mean()),
                "enter_support_90": float(frame["enter_support_90"].mean()),
                "pcr_support_90": float(frame["pcr_support_90"].mean()),
                "enter_pairwise": float(frame["enter_pairwise"].mean()),
                "pcr_pairwise": float(frame["pcr_pairwise"].mean()),
                "enter_false_lag_rate": float(frame["enter_false_lag_rate"].mean()),
                "pcr_false_lag_rate": float(frame["pcr_false_lag_rate"].mean()),
                "retained_components": float(frame["retained_components"].mean()),
                "explained_variance_share": float(frame["explained_variance_share"].mean()),
                "component_dominance": float(frame["component_dominance"].mean()),
                "component_true_loading_share": float(frame["component_true_loading_share"].mean()),
            }
        )
    case_summary = pd.DataFrame(case_rows).sort_values(["model", "case"]).reset_index(drop=True)

    overall_rows: list[dict[str, Any]] = []
    grouped_frames: list[tuple[str, pd.DataFrame]] = [("overall", window_level)]
    grouped_frames.extend((str(model_name), model_frame) for model_name, model_frame in window_level.groupby("model"))
    for scope_name, frame in grouped_frames:
        overall_rows.extend(
            [
                {
                    "scope": scope_name,
                    "method": "enter",
                    "windows": float(len(frame)),
                    "validation_r2": float(frame["enter_validation_r2"].mean()),
                    "validation_mae": float(frame["enter_validation_mae"].mean()),
                    "true_mass_share": float(frame["enter_true_mass_share"].mean()),
                    "support_90": float(frame["enter_support_90"].mean()),
                    "pairwise_score": float(frame["enter_pairwise"].mean()),
                    "false_lag_rate": float(frame["enter_false_lag_rate"].mean()),
                    "hit_rate": float(frame["enter_hit_rate"].mean()),
                    "cond": float(frame["cond_original"].mean()),
                    "retained_components": np.nan,
                    "explained_variance_share": np.nan,
                    "component_dominance": np.nan,
                    "component_true_loading_share": np.nan,
                },
                {
                    "scope": scope_name,
                    "method": "pcr",
                    "windows": float(len(frame)),
                    "validation_r2": float(frame["pcr_validation_r2"].mean()),
                    "validation_mae": float(frame["pcr_validation_mae"].mean()),
                    "true_mass_share": float(frame["pcr_true_mass_share"].mean()),
                    "support_90": float(frame["pcr_support_90"].mean()),
                    "pairwise_score": float(frame["pcr_pairwise"].mean()),
                    "false_lag_rate": float(frame["pcr_false_lag_rate"].mean()),
                    "hit_rate": float(frame["pcr_hit_rate"].mean()),
                    "cond": float(frame["cond_pcr"].mean()),
                    "retained_components": float(frame["retained_components"].mean()),
                    "explained_variance_share": float(frame["explained_variance_share"].mean()),
                    "component_dominance": float(frame["component_dominance"].mean()),
                    "component_true_loading_share": float(frame["component_true_loading_share"].mean()),
                },
            ]
        )
    overall_summary = pd.DataFrame(overall_rows)

    rho_summary = (
        rho_curve.groupby("rho", as_index=False)
        .agg(
            windows=("rho", "size"),
            retained_components=("retained_components", "mean"),
            explained_variance_share=("explained_variance_share", "mean"),
            validation_r2=("validation_r2", "mean"),
            validation_mae=("validation_mae", "mean"),
            true_mass_share=("true_mass_share", "mean"),
            support_90=("support_90", "mean"),
            pairwise_score=("pairwise_score", "mean"),
            false_lag_rate=("false_lag_rate", "mean"),
            component_dominance=("component_dominance", "mean"),
        )
        .sort_values("rho")
        .reset_index(drop=True)
    )

    component_summary = (
        component_loadings.groupby(["case", "component", "variable"], as_index=False)
        .agg(
            abs_loading=("abs_loading", "mean"),
            loading_share=("loading_share", "mean"),
            is_true_predictor=("is_true_predictor", "max"),
        )
        .sort_values(["case", "component", "abs_loading"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    return {
        "window_level": window_level,
        "case_summary": case_summary,
        "overall_summary": overall_summary,
        "rho_curve": rho_curve,
        "rho_summary": rho_summary,
        "component_loadings": component_loadings,
        "component_summary": component_summary,
    }


def save_plan_f_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_payload: dict[str, Any] = {}
    for name, frame in results.items():
        csv_path = out / f"{name}.csv"
        frame.to_csv(csv_path, index=False)
        summary_payload[name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)
    return out
