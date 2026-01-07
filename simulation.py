"""Simulation utilities for nonlinear discrete dynamics.

Implements the base logistic-like map
    X_{n+1} = X_n + X_n * A * (K - X_n)
and helpers for parameter sweeps.

All functions return pandas DataFrames to stay compatible with analysis and
visualization modules.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    n_steps: int = 300
    x0: float = 0.0001
    A: float = 2.0
    K: float = 2.0
    noise_std: float = 0.0
    random_state: int | None = None


def logistic_map(cfg: SimulationConfig) -> pd.DataFrame:
    """Simulate X_{n+1} = X_n + X_n * A * (K - X_n).

    Parameters
    ----------
    cfg : SimulationConfig
        Parameters of the simulation.

    Returns
    -------
    pd.DataFrame
        Columns: t, X, dX, A, K
    """
    rng = np.random.default_rng(cfg.random_state)
    X = np.empty(cfg.n_steps, dtype=float)
    X[0] = cfg.x0

    for i in range(1, cfg.n_steps):
        xn = X[i - 1]
        xn1 = xn + xn * cfg.A * (cfg.K - xn)
        if cfg.noise_std > 0:
            xn1 += rng.normal(0.0, cfg.noise_std)
        if not np.isfinite(xn1):
            # Деградация траектории — заполняем NaN до конца и выходим
            X[i:] = np.nan
            break
        X[i] = xn1

    dX = np.diff(X, prepend=X[0])
    return pd.DataFrame(
        {
            "t": np.arange(cfg.n_steps, dtype=int),
            "X": X,
            "dX": dX,
            "A": cfg.A,
            "K": cfg.K,
        }
    )


def parameter_range(start: float, stop: float, step: float) -> List[float]:
    """Helper to build an inclusive float range with a fixed step."""
    if step <= 0:
        raise ValueError("step must be positive")
    values = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 12))
        current += step
    return values


def sweep_parameters(
    A_values: Iterable[float],
    K_values: Iterable[float],
    n_steps: int = 300,
    x0: float = 0.0001,
    noise_std: float = 0.0,
    random_state: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Dict[Tuple[float, float], pd.DataFrame]:
    """Run the model for all combinations of A and K.

    Parameters
    ----------
    A_values : iterable of float
        Values of A to simulate.
    K_values : iterable of float
        Values of K to simulate.
    n_steps : int
        Length of each trajectory.
    x0 : float
        Initial value.
    noise_std : float
        Optional additive Gaussian noise.
    random_state : int, optional
        Seed for reproducibility.
    progress_cb : callable, optional
        Called with (index, total) after each simulation.

    Returns
    -------
    dict
        Mapping (A, K) -> trajectory DataFrame.
    """
    results: Dict[Tuple[float, float], pd.DataFrame] = {}
    pairs = list(itertools.product(A_values, K_values))
    total = len(pairs)
    for idx, (A, K) in enumerate(pairs, start=1):
        cfg = SimulationConfig(
            n_steps=n_steps,
            x0=x0,
            A=A,
            K=K,
            noise_std=noise_std,
            random_state=random_state,
        )
        results[(A, K)] = logistic_map(cfg)
        if progress_cb:
            progress_cb(idx, total)
    return results


def split_series(series: pd.Series, n_segments: int = 5) -> List[pd.Series]:
    """Split a series into equal-length contiguous segments."""
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    length = len(series)
    segment_len = length // n_segments
    if segment_len == 0:
        raise ValueError(
            f"Series too short for {n_segments} segments (len={length})."
        )
    segments = []
    for i in range(n_segments):
        start = i * segment_len
        end = (i + 1) * segment_len if i < n_segments - 1 else length
        segments.append(series.iloc[start:end].reset_index(drop=True))
    return segments
