from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_series(series: np.ndarray, title: str = "Time Series", xlabel: str = "n", ylabel: str = "X"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.asarray(series, dtype=float))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax


def plot_phase_portrait(series: np.ndarray, title: str = "Phase Portrait", xlabel: str = "X_n", ylabel: str = "X_{n+1}"):
    x = np.asarray(series, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x[:-1], x[1:], s=18, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax


def plot_rolling_coefficients(
    windows: pd.DataFrame,
    coefficient_columns: list[str],
    title: str = "Rolling Coefficients",
    xlabel: str = "Window Start",
    ylabel: str = "Coefficient",
):
    fig, ax = plt.subplots(figsize=(10, 4))
    for column in coefficient_columns:
        if column in windows.columns:
            ax.plot(windows["start"], windows[column], marker=".", label=column)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_stepwise_frequency(freq: pd.DataFrame, title: str = "Stepwise Frequency", xlabel: str = "Variable", ylabel: str = "Count"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(freq["var"], freq["count"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax


def save_figure(fig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
