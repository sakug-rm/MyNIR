from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_growth_rate(x_prev: float, x_next: float, zero_guard: float = 1e-12) -> float:
    base = x_prev if abs(x_prev) > zero_guard else zero_guard
    return (x_next - x_prev) / base


def _result_dict(model_name: str, valid: bool, **params: float) -> dict[str, Any]:
    out: dict[str, Any] = {"model_name": model_name, "valid": valid}
    out.update(params)
    return out


def estimate_base_from_triplet(points: np.ndarray, zero_guard: float = 1e-12) -> dict[str, Any]:
    """Estimate base-model parameters from three consecutive observations."""
    x = np.asarray(points, dtype=float)
    if len(x) != 3:
        raise ValueError("base triplet estimation expects exactly 3 points")

    x0, x1, x2 = x
    denom = x1 - x0
    if abs(denom) <= zero_guard:
        return _result_dict("base", False, a=np.nan, k=np.nan, alpha=np.nan)

    d0 = _safe_growth_rate(x0, x1, zero_guard=zero_guard)
    d1 = _safe_growth_rate(x1, x2, zero_guard=zero_guard)
    a = (d0 - d1) / denom
    alpha = d0 + a * x0
    if abs(a) <= zero_guard:
        return _result_dict("base", False, a=np.nan, k=np.nan, alpha=alpha)
    k = alpha / a
    valid = np.isfinite(a) and np.isfinite(k) and a > 0.0 and k > 0.0
    return _result_dict("base", valid, a=float(a), k=float(k), alpha=float(alpha))


def estimate_delay_from_quadruplet(points: np.ndarray, zero_guard: float = 1e-12) -> dict[str, Any]:
    """Estimate delay-model parameters from four consecutive observations."""
    x = np.asarray(points, dtype=float)
    if len(x) != 4:
        raise ValueError("delay quadruplet estimation expects exactly 4 points")

    x0, x1, x2, x3 = x
    denom = x1 - x0
    if abs(denom) <= zero_guard:
        return _result_dict("delay", False, g=np.nan, k=np.nan, alpha=np.nan)

    d1 = _safe_growth_rate(x1, x2, zero_guard=zero_guard)
    d2 = _safe_growth_rate(x2, x3, zero_guard=zero_guard)
    g = (d1 - d2) / denom
    alpha = d1 + g * x0
    if abs(g) <= zero_guard:
        return _result_dict("delay", False, g=np.nan, k=np.nan, alpha=alpha)
    k = alpha / g
    valid = np.isfinite(g) and np.isfinite(k) and g > 0.0 and k > 0.0
    return _result_dict("delay", valid, g=float(g), k=float(k), alpha=float(alpha))


def estimate_mixed_from_quintet(points: np.ndarray, zero_guard: float = 1e-12) -> dict[str, Any]:
    """Estimate mixed-model parameters from five consecutive observations."""
    x = np.asarray(points, dtype=float)
    if len(x) != 5:
        raise ValueError("mixed quintet estimation expects exactly 5 points")

    design = np.array(
        [
            [1.0, x[1], x[0]],
            [1.0, x[2], x[1]],
            [1.0, x[3], x[2]],
        ],
        dtype=float,
    )
    target = np.array(
        [
            _safe_growth_rate(x[1], x[2], zero_guard=zero_guard),
            _safe_growth_rate(x[2], x[3], zero_guard=zero_guard),
            _safe_growth_rate(x[3], x[4], zero_guard=zero_guard),
        ],
        dtype=float,
    )

    if abs(np.linalg.det(design)) <= zero_guard:
        return _result_dict("mixed", False, q=np.nan, gamma=np.nan, k=np.nan, alpha=np.nan)

    alpha, b_x, b_lag = np.linalg.solve(design, target)
    q = -b_x
    if abs(q) <= zero_guard:
        return _result_dict("mixed", False, q=np.nan, gamma=np.nan, k=np.nan, alpha=float(alpha))
    gamma = b_lag / b_x
    k = alpha / q
    valid = np.isfinite(q) and np.isfinite(gamma) and np.isfinite(k) and q > 0.0 and k > 0.0
    return _result_dict("mixed", valid, q=float(q), gamma=float(gamma), k=float(k), alpha=float(alpha))


def fit_direct_identification(window: np.ndarray, model_name: str, zero_guard: float = 1e-12) -> dict[str, Any]:
    """Estimate model parameters on a local window using the structural equation."""
    x = np.asarray(window, dtype=float)
    model = model_name.lower()

    if model == "base":
        if len(x) < 3:
            raise ValueError("base direct identification requires at least 3 points")
        y = np.array([_safe_growth_rate(xi, xj, zero_guard=zero_guard) for xi, xj in zip(x[:-1], x[1:])], dtype=float)
        design = np.column_stack([np.ones(len(x) - 1), x[:-1]])
        if design.shape[0] == 2:
            return estimate_base_from_triplet(x[:3], zero_guard=zero_guard)
        beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            return _result_dict("base", False, a=np.nan, k=np.nan, alpha=np.nan)
        alpha, b_x = beta
        a = -b_x
        if abs(a) <= zero_guard:
            return _result_dict("base", False, a=np.nan, k=np.nan, alpha=float(alpha))
        k = alpha / a
        valid = np.isfinite(a) and np.isfinite(k) and a > 0.0 and k > 0.0
        return _result_dict("base", valid, a=float(a), k=float(k), alpha=float(alpha))

    if model == "delay":
        if len(x) < 4:
            raise ValueError("delay direct identification requires at least 4 points")
        y = np.array([_safe_growth_rate(x[i], x[i + 1], zero_guard=zero_guard) for i in range(1, len(x) - 1)], dtype=float)
        design = np.column_stack([np.ones(len(y)), x[:-2]])
        if design.shape[0] == 2:
            return estimate_delay_from_quadruplet(x[:4], zero_guard=zero_guard)
        beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            return _result_dict("delay", False, g=np.nan, k=np.nan, alpha=np.nan)
        alpha, b_lag = beta
        g = -b_lag
        if abs(g) <= zero_guard:
            return _result_dict("delay", False, g=np.nan, k=np.nan, alpha=float(alpha))
        k = alpha / g
        valid = np.isfinite(g) and np.isfinite(k) and g > 0.0 and k > 0.0
        return _result_dict("delay", valid, g=float(g), k=float(k), alpha=float(alpha))

    if model == "mixed":
        if len(x) < 5:
            raise ValueError("mixed direct identification requires at least 5 points")
        if len(x) == 5:
            return estimate_mixed_from_quintet(x, zero_guard=zero_guard)
        y = np.array([_safe_growth_rate(x[i], x[i + 1], zero_guard=zero_guard) for i in range(1, len(x) - 1)], dtype=float)
        design = np.column_stack([np.ones(len(y)), x[1:-1], x[:-2]])
        beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 3:
            return _result_dict("mixed", False, q=np.nan, gamma=np.nan, k=np.nan, alpha=np.nan)
        alpha, b_x, b_lag = beta
        q = -b_x
        if abs(q) <= zero_guard:
            return _result_dict("mixed", False, q=np.nan, gamma=np.nan, k=np.nan, alpha=float(alpha))
        gamma = b_lag / b_x
        k = alpha / q
        valid = np.isfinite(q) and np.isfinite(gamma) and np.isfinite(k) and q > 0.0 and k > 0.0
        return _result_dict("mixed", valid, q=float(q), gamma=float(gamma), k=float(k), alpha=float(alpha))

    raise ValueError("model_name must be one of: base, delay, mixed")


def rolling_direct_identification(
    series: np.ndarray,
    model_name: str,
    window_size: int | None = None,
    zero_guard: float = 1e-12,
) -> pd.DataFrame:
    """Run structural direct identification on a rolling window."""
    x = np.asarray(series, dtype=float)
    min_window = {"base": 3, "delay": 4, "mixed": 5}[model_name.lower()]
    width = window_size or min_window
    if width < min_window:
        raise ValueError(f"window_size must be >= {min_window} for model {model_name}")

    rows: list[dict[str, Any]] = []
    for start in range(0, len(x) - width + 1):
        end = start + width
        result = fit_direct_identification(x[start:end], model_name=model_name, zero_guard=zero_guard)
        row = {"start": start, "end": end}
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_parameter_errors(estimates: pd.DataFrame | list[dict[str, Any]], truth: dict[str, float]) -> dict[str, float]:
    """Summarize absolute estimation errors for all parameters present in truth."""
    df = pd.DataFrame(estimates)
    if df.empty:
        summary = {"valid_share": float("nan")}
        for param in truth:
            summary[f"{param}_mae"] = float("nan")
            summary[f"{param}_median_ae"] = float("nan")
            summary[f"{param}_iqr_ae"] = float("nan")
        return summary

    valid = df.get("valid", pd.Series([True] * len(df))).fillna(False).astype(bool)
    summary: dict[str, float] = {"valid_share": float(valid.mean())}
    valid_df = df.loc[valid].copy()

    for param, true_value in truth.items():
        if param not in valid_df.columns:
            continue
        errors = (valid_df[param] - true_value).abs().dropna()
        if errors.empty:
            summary[f"{param}_mae"] = float("nan")
            summary[f"{param}_median_ae"] = float("nan")
            summary[f"{param}_iqr_ae"] = float("nan")
            continue
        q1 = float(errors.quantile(0.25))
        q3 = float(errors.quantile(0.75))
        summary[f"{param}_mae"] = float(errors.mean())
        summary[f"{param}_median_ae"] = float(errors.median())
        summary[f"{param}_iqr_ae"] = float(q3 - q1)
    return summary
