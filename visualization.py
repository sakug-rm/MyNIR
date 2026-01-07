"""Visualization helpers for trajectories, phase portraits, and forecasts."""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trajectory(df: pd.DataFrame, ax=None, label: str | None = None):
    """Plot X over time."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(df["t"], df["X"], label=label or "X")
    ax.set_xlabel("t")
    ax.set_ylabel("X")
    if label:
        ax.legend()
    return ax


def plot_phase_portrait(
    series: Sequence[float],
    parabola: Dict | None = None,
    ax=None,
    scatter_kwargs: Dict | None = None,
):
    """Plot (X_n, X_{n+1}) scatter and optional parabola fit."""
    if ax is None:
        _, ax = plt.subplots()
    scatter_kwargs = scatter_kwargs or {"s": 12, "alpha": 0.6}
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    ax.scatter(arr[:-1], arr[1:], **scatter_kwargs)
    ax.set_xlabel("X_n")
    ax.set_ylabel("X_{n+1}")
    if parabola and np.isfinite([parabola.get("a", np.nan), parabola.get("b", np.nan), parabola.get("c", np.nan)]).all():
        xs = np.linspace(arr.min(), arr.max(), 200)
        ys = parabola["a"] * xs**2 + parabola["b"] * xs + parabola["c"]
        label = f"fit: a={parabola['a']:.3f}, b={parabola['b']:.3f}, c={parabola['c']:.3f}"
        ax.plot(xs, ys, color="red", label=label)
        ax.legend()
    return ax


def plot_forecast(
    history: Sequence[float],
    forecast: Sequence[float],
    actual_future: Sequence[float] | None = None,
    ax=None,
):
    """Plot forecast continuation against optional actual trajectory."""
    if ax is None:
        _, ax = plt.subplots()
    hist_arr = np.asarray(history, dtype=float)
    fc_arr = np.asarray(forecast, dtype=float)
    time_hist = np.arange(len(hist_arr))
    time_fc = np.arange(len(hist_arr), len(hist_arr) + len(fc_arr))

    ax.plot(time_hist, hist_arr, label="history")
    ax.plot(time_fc, fc_arr, label="forecast", linestyle="--")
    if actual_future is not None:
        actual_arr = np.asarray(actual_future, dtype=float)
        m = min(len(actual_arr), len(time_fc))
        if m > 0:
            ax.plot(time_fc[:m], actual_arr[:m], label="actual future", alpha=0.7)
    ax.set_xlabel("t")
    ax.set_ylabel("X")
    ax.legend()
    return ax


def plot_phase_error(errors: Sequence[float], ax=None):
    """Plot cumulative phase-space error."""
    if ax is None:
        _, ax = plt.subplots()
    arr = np.asarray(errors, dtype=float)
    ax.plot(np.arange(len(arr)), arr, label="cum phase error")
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative error")
    ax.legend()
    return ax
