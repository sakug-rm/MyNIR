"""Analysis helpers: phase portraits, parabola fit, and error metrics."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from regression import evaluate_forecast, forecast_ar


def phase_points(series: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_n, X_{n+1}) pairs for a trajectory."""
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr[:-1], arr[1:]


def fit_phase_parabola(series: Sequence[float]) -> Dict:
    """Fit X_{n+1} = a*X_n^2 + b*X_n + c to the phase points."""
    x, y = phase_points(series)
    if len(x) < 3:
        raise ValueError("At least 3 points required for quadratic fit.")
    try:
        coeffs = np.polyfit(x, y, deg=2)
    except np.linalg.LinAlgError:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "rmse": np.nan, "r2": np.nan}
    y_pred = np.polyval(coeffs, x)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
    return {"a": coeffs[0], "b": coeffs[1], "c": coeffs[2], "rmse": rmse, "r2": r2}


def phase_space_error(true_series: Sequence[float], pred_series: Sequence[float]) -> np.ndarray:
    """Compute per-step Euclidean error in phase space."""
    x_true_n, x_true_n1 = phase_points(true_series)
    x_pred_n, x_pred_n1 = phase_points(pred_series)
    min_len = min(len(x_true_n), len(x_pred_n))
    if min_len == 0:
        return np.array([], dtype=float)
    err = np.sqrt((x_true_n[:min_len] - x_pred_n[:min_len]) ** 2 + (x_true_n1[:min_len] - x_pred_n1[:min_len]) ** 2)
    return err


def cumulative_error(errors: ArrayLike) -> np.ndarray:
    """Cumulative sum of absolute errors."""
    arr = np.abs(np.asarray(errors, dtype=float))
    return np.cumsum(arr)


def compare_forecast(
    model_params: pd.Series,
    lags: Sequence[int],
    history: Sequence[float],
    actual_future: Sequence[float],
    horizon: int,
    add_const: bool = True,
) -> Dict:
    """Forecast and compare with ground truth future values."""
    preds = forecast_ar(model_params, lags, history, horizon, add_const=add_const)
    matched_len = min(len(actual_future), len(preds))
    if matched_len > 0:
        metrics = evaluate_forecast(actual_future[:matched_len], preds[:matched_len])
        phase_err = phase_space_error(
            list(history[-1:]) + list(actual_future[:matched_len]),
            list(history[-1:]) + list(preds[:matched_len]),
        )
        phase_cum = cumulative_error(phase_err)
    else:
        metrics = {"mae": float("nan"), "mse": float("nan")}
        phase_err = np.array([], dtype=float)
        phase_cum = np.array([], dtype=float)
    return {
        "pred": preds,
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "phase_error": phase_err,
        "phase_error_cum": phase_cum,
    }
