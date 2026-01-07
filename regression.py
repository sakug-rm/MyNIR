"""AR modeling utilities (stepwise selection and full enter models)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _lagged_matrix(series: pd.Series, lags: Sequence[int]) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag{lag}"] = df["y"].shift(lag)
    return df.dropna()


def _ols_with_betas(y: pd.Series, X: pd.DataFrame, add_const: bool = True):
    X_std = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
    X_model = sm.add_constant(X_std) if add_const else X_std
    model = sm.OLS(y, X_model).fit()

    y_sigma = y.std(ddof=0)
    betas = {}
    for col in X_std.columns:
        betas[col] = model.params.get(col, 0.0) * (X_std[col].std(ddof=0) / y_sigma)
    return model, pd.Series(betas)


def fit_ar_enter(
    series: pd.Series,
    max_lag: int = 10,
    add_const: bool = True,
) -> Dict:
    """Fit AR(1..max_lag) with all lags included."""
    if max_lag < 1:
        raise ValueError("max_lag must be >=1")
    df = _lagged_matrix(series, range(1, max_lag + 1))
    if df.empty:
        raise ValueError("Insufficient data for enter AR model.")
    y = df["y"]
    X = df[[f"lag{i}" for i in range(1, max_lag + 1)]]
    model, betas = _ols_with_betas(y, X, add_const=add_const)
    return {
        "model": model,
        "betas": betas,
        "params": model.params,
        "adj_r2": model.rsquared_adj,
        "stderr": model.bse,
        "pvalues": model.pvalues,
        "lags": list(range(1, max_lag + 1)),
        "add_const": add_const,
    }


def fit_ar_stepwise(
    series: pd.Series,
    max_lag: int = 10,
    criterion: str = "bic",
    add_const: bool = True,
    min_improvement: float = 1e-4,
) -> Dict:
    """Stepwise lag selection using AIC/BIC improvement."""
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'")
    remaining = list(range(1, max_lag + 1))
    selected: List[int] = []
    best_score = np.inf
    best_result = None

    while remaining:
        candidate_results = []
        for lag in remaining:
            lags = sorted(selected + [lag])
            df = _lagged_matrix(series, lags)
            if df.empty:
                continue
            y = df["y"]
            X = df[[f"lag{i}" for i in lags]]
            model, betas = _ols_with_betas(y, X, add_const=add_const)
            score = model.aic if criterion == "aic" else model.bic
            candidate_results.append((score, lag, model, betas, lags))

        if not candidate_results:
            break

        score, lag, model, betas, lags = min(candidate_results, key=lambda t: t[0])
        if best_result is None or score + min_improvement < best_score:
            best_score = score
            best_result = (model, betas, lags)
            selected.append(lag)
            remaining.remove(lag)
        else:
            break

    if best_result is None:
        raise ValueError("Failed to fit stepwise AR model (no valid candidates).")

    model, betas, lags = best_result
    return {
        "model": model,
        "betas": betas,
        "params": model.params,
        "adj_r2": model.rsquared_adj,
        "stderr": model.bse,
        "pvalues": model.pvalues,
        "lags": lags,
        "add_const": add_const,
        "selection_criterion": criterion,
        "selection_score": best_score,
    }


def _forecast_one_step(history: List[float], params: pd.Series, lags: Sequence[int], add_const: bool) -> float:
    value = params.get("const", 0.0) if add_const else 0.0
    for lag in lags:
        coef = params.get(f"lag{lag}", 0.0)
        try:
            value += coef * history[-lag]
        except IndexError:
            raise ValueError("History length is insufficient for forecasting.")
    return float(value)


def forecast_ar(
    params: pd.Series,
    lags: Sequence[int],
    history: Sequence[float],
    horizon: int,
    add_const: bool = True,
) -> np.ndarray:
    """Iterative forecast using fitted AR parameters."""
    hist = list(history)
    preds = []
    for _ in range(horizon):
        next_val = _forecast_one_step(hist, params, lags, add_const)
        preds.append(next_val)
        hist.append(next_val)
    return np.array(preds)


def evaluate_forecast(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute MAE and MSE for forecast comparison."""
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    return {"mae": mae, "mse": mse}


@dataclass
class SegmentModel:
    """Container for a fitted AR model on a single segment."""

    segment_index: int
    method: str
    lags: List[int]
    params: pd.Series
    betas: pd.Series
    adj_r2: float
    stderr: pd.Series
    pvalues: pd.Series
    model: object
    add_const: bool = True


def fit_segments(
    segments: Iterable[pd.Series],
    method: str = "stepwise",
    max_lag: int = 10,
    **kwargs,
) -> List[SegmentModel]:
    """Fit AR models for each segment using the chosen method."""
    models: List[SegmentModel] = []
    for idx, seg in enumerate(segments):
        if method == "stepwise":
            res = fit_ar_stepwise(seg, max_lag=max_lag, **kwargs)
        elif method == "enter":
            res = fit_ar_enter(seg, max_lag=max_lag, **kwargs)
        else:
            raise ValueError("method must be 'stepwise' or 'enter'")
        models.append(
            SegmentModel(
                segment_index=idx,
                method=method,
                lags=res["lags"],
                params=res["params"],
                betas=res["betas"],
                adj_r2=res["adj_r2"],
                stderr=res["stderr"],
                pvalues=res["pvalues"],
                model=res["model"],
                add_const=res["add_const"],
            )
        )
    return models
