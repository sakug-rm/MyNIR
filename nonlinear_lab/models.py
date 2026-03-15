from __future__ import annotations

import numpy as np


def _apply_clip(value: float, clip_min: float = 0.0, clip_max: float | None = None) -> float:
    if value < clip_min:
        return clip_min
    if clip_max is not None and value > clip_max:
        return clip_max
    return value


def generate_base_process(
    a: float,
    k: float = 1.0,
    x0: float = 1e-4,
    steps: int = 200,
    clip_min: float = 0.0,
    clip_max: float | None = None,
) -> np.ndarray:
    """Generate the base nonlinear process.

    Model:
        x[n+1] = x[n] + a * x[n] * (k - x[n])

    Args:
        a: Growth intensity coefficient.
        k: Carrying capacity.
        x0: Initial state.
        steps: Number of time steps to generate.
        clip_min: Lower bound applied after each step.
        clip_max: Optional upper bound applied after each step.

    Returns:
        NumPy array of generated states.
    """
    x = np.zeros(steps, dtype=float)
    x[0] = x0
    for i in range(steps - 1):
        x_next = x[i] + x[i] * a * (k - x[i])
        x[i + 1] = _apply_clip(x_next, clip_min=clip_min, clip_max=clip_max)
    return x


def generate_delay_process(
    g: float,
    x0: float = 1e-4,
    steps: int = 200,
    clip_min: float = 0.0,
    clip_max: float | None = 2.0,
) -> np.ndarray:
    """Generate the delayed nonlinear process.

    Model:
        x[n+1] = x[n] + g * x[n] * (1 - x[n-1])

    The second point is bootstrapped from the base-model update, matching the
    notebook logic.
    """
    x = np.zeros(steps, dtype=float)
    x[0] = x0
    if steps == 1:
        return x

    x1 = x[0] + x[0] * g * (1.0 - x[0])
    x[1] = _apply_clip(x1, clip_min=clip_min, clip_max=clip_max)

    for i in range(1, steps - 1):
        x_next = x[i] + x[i] * g * (1.0 - x[i - 1])
        x[i + 1] = _apply_clip(x_next, clip_min=clip_min, clip_max=clip_max)
    return x


def generate_mixed_process(
    q: float,
    gamma: float,
    x0: float = 1e-4,
    steps: int = 200,
    clip_min: float = 0.0,
    clip_max: float | None = 5.0,
) -> np.ndarray:
    """Generate the mixed nonlinear process.

    Model:
        x[n+1] = x[n] + q * x[n] * (1 - x[n] - gamma * x[n-1])
    """
    x = np.zeros(steps, dtype=float)
    x[0] = x0
    if steps == 1:
        return x

    x1 = x[0] + x[0] * q * (1.0 - x[0] - gamma * x[0])
    x[1] = _apply_clip(x1, clip_min=clip_min, clip_max=clip_max)

    for i in range(1, steps - 1):
        term = 1.0 - x[i] - gamma * x[i - 1]
        x_next = x[i] + x[i] * q * term
        x[i + 1] = _apply_clip(x_next, clip_min=clip_min, clip_max=clip_max)
    return x


def theoretical_coeffs(q: float, gamma: float) -> dict[str, float]:
    """Return theoretical regression coefficients for the mixed-model growth rate."""
    return {
        "B_X_n_theory": -q,
        "B_Lag1_theory": -q * gamma,
    }


def fixed_point(gamma: float) -> float:
    """Return the non-zero stationary point for the mixed model."""
    if np.isclose(1.0 + gamma, 0.0):
        return float("nan")
    return 1.0 / (1.0 + gamma)
