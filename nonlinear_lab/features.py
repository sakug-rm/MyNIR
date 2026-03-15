from __future__ import annotations

import numpy as np
import pandas as pd


def growth_rate(series: np.ndarray, zero_guard: float = 1e-12) -> np.ndarray:
    """Compute omega[n+1] = (x[n+1] - x[n]) / x[n] with zero protection."""
    x = np.asarray(series, dtype=float)
    base = x[:-1].copy()
    base[base == 0] = zero_guard
    return (x[1:] - x[:-1]) / base


def make_regression_df(series: np.ndarray, lags: int = 10, zero_guard: float = 1e-12) -> pd.DataFrame:
    """Build a regression table with omega, X_n and lagged X features."""
    x = np.asarray(series, dtype=float)
    omega = growth_rate(x, zero_guard=zero_guard)
    x_n = x[:-1]

    df = pd.DataFrame({"omega": omega, "X_n": x_n})
    source = pd.Series(x_n)
    for lag in range(1, lags + 1):
        df[f"Lag_{lag}"] = source.shift(lag)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
