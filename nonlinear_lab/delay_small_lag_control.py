from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.models import generate_delay_process
from nonlinear_lab.regression import stepwise_selection


@dataclass(frozen=True)
class DelayVariant:
    name: str
    label: str
    params: dict[str, float]
    one_step_rmse: float


def _fit_delay_variant(series: np.ndarray, features: list[str]) -> tuple[dict[str, float], float]:
    df = make_regression_df(series, lags=10)
    X = df[features]
    y = df["omega"]
    model = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    fitted = model.predict(sm.add_constant(X, has_constant="add"))
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    return {str(k): float(v) for k, v in model.params.items()}, rmse


def _recursive_linear_forecast(history: np.ndarray, params: dict[str, float], horizon: int) -> np.ndarray:
    values = list(map(float, history))
    for _ in range(horizon):
        x_t = values[-1]
        omega = float(params.get("const", 0.0))
        for name, coef in params.items():
            if name == "const":
                continue
            if name == "X_n":
                omega += coef * values[-1]
                continue
            if name.startswith("Lag_"):
                lag = int(name.split("_")[1])
                omega += coef * values[-1 - lag]
        values.append(x_t * (1.0 + omega))
    return np.asarray(values, dtype=float)


def _true_delay_continuation(history: np.ndarray, g: float, horizon: int) -> np.ndarray:
    values = list(map(float, history))
    for _ in range(horizon):
        x_prev = values[-2]
        x_cur = values[-1]
        values.append(x_cur + g * x_cur * (1.0 - x_prev))
    return np.asarray(values, dtype=float)


def _turning_points(x: np.ndarray) -> int:
    diff = np.diff(x)
    if len(diff) == 0:
        return 0
    signs = np.sign(diff)
    cleaned = []
    for sign in signs:
        if sign == 0:
            continue
        if not cleaned or cleaned[-1] != sign:
            cleaned.append(sign)
    return max(len(cleaned) - 1, 0)


def _phase_preserved(reference: np.ndarray, candidate: np.ndarray) -> bool:
    return abs(_turning_points(reference) - _turning_points(candidate)) <= 1


def _first_noticeable_divergence(errors: np.ndarray, threshold: float = 1e-6) -> int | None:
    for idx, value in enumerate(errors, start=1):
        if float(value) > threshold:
            return idx
    return None


def _summary_row(
    variant: DelayVariant,
    true_future: np.ndarray,
    predicted_future: np.ndarray,
    divergence_threshold: float = 1e-6,
) -> dict[str, Any]:
    errors = np.abs(predicted_future - true_future)
    return {
        "variant": variant.name,
        "label": variant.label,
        "one_step_rmse": variant.one_step_rmse,
        "rmse_h10": float(np.sqrt(np.mean((predicted_future[:10] - true_future[:10]) ** 2))),
        "rmse_h25": float(np.sqrt(np.mean((predicted_future[:25] - true_future[:25]) ** 2))),
        "rmse_h50": float(np.sqrt(np.mean((predicted_future[:50] - true_future[:50]) ** 2))),
        "rmse_h100": float(np.sqrt(np.mean((predicted_future[:100] - true_future[:100]) ** 2))),
        "max_abs_deviation": float(errors.max()),
        "first_noticeable_divergence": _first_noticeable_divergence(errors, threshold=divergence_threshold),
        "phase_preserved": _phase_preserved(true_future, predicted_future),
    }


def build_delay_small_lag_control(
    g: float = 0.8,
    steps: int = 150,
    horizon: int = 100,
    lags: int = 10,
    threshold: float = 1e-12,
    x0: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if lags < 9:
        raise ValueError("lags must be at least 9 for the control experiment")

    series = generate_delay_process(g=g, x0=x0, steps=steps, clip_max=None)
    regression_df = make_regression_df(series, lags=lags)
    X = regression_df.drop(columns=["omega"])
    y = regression_df["omega"]
    selected = stepwise_selection(X, y)

    truncated_params, truncated_rmse = _fit_delay_variant(series, ["Lag_1"])
    stepwise_params, stepwise_rmse = _fit_delay_variant(series, selected)
    thresholded_params = {
        key: (0.0 if key != "const" and abs(value) < threshold else value)
        for key, value in stepwise_params.items()
    }

    variants = [
        DelayVariant("true_nonlinear", "Эталон: истинная нелинейная delay-модель", {}, 0.0),
        DelayVariant("lag1_only", "Усечённая регрессия: только Lag_1", truncated_params, truncated_rmse),
        DelayVariant("stepwise_full", "Полная Stepwise-модель: Lag_1 + Lag_3 + Lag_9", stepwise_params, stepwise_rmse),
        DelayVariant("thresholded", f"Пороговая версия: зануление коэффициентов < {threshold:.0e}", thresholded_params, stepwise_rmse),
    ]

    history = series[-max(lags, 10) :]
    true_path = _true_delay_continuation(history=history, g=g, horizon=horizon)
    true_future = true_path[len(history) :]

    summary_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    coefficients_rows: list[dict[str, Any]] = []

    for step, value in enumerate(true_future, start=1):
        path_rows.append({"variant": "true_nonlinear", "step": step, "value": float(value)})

    for variant in variants[1:]:
        predicted_path = _recursive_linear_forecast(history=history, params=variant.params, horizon=horizon)
        predicted_future = predicted_path[len(history) :]
        summary_rows.append(_summary_row(variant, true_future=true_future, predicted_future=predicted_future))

        for step, value in enumerate(predicted_future, start=1):
            path_rows.append({"variant": variant.name, "step": step, "value": float(value)})

        for key, value in variant.params.items():
            coefficients_rows.append({"variant": variant.name, "coefficient": key, "value": float(value)})

    summary = pd.DataFrame(summary_rows)
    paths = pd.DataFrame(path_rows)
    coefficients = pd.DataFrame(coefficients_rows)
    meta = {
        "g": g,
        "steps": steps,
        "horizon": horizon,
        "selected_features": selected,
        "threshold": threshold,
        "history_length": len(history),
    }
    return summary, paths, {"coefficients": coefficients, "meta": meta}


def plot_delay_small_lag_control(paths: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)

    pivot = paths.pivot(index="step", columns="variant", values="value").sort_index()
    axes[0].plot(pivot.index, pivot["true_nonlinear"], label="Эталон", linewidth=2.2, color="#111827")
    colors = {
        "lag1_only": "#1d4ed8",
        "stepwise_full": "#d97706",
        "thresholded": "#059669",
    }
    labels = {
        "lag1_only": "Только Lag_1",
        "stepwise_full": "Полная Stepwise-модель",
        "thresholded": "Пороговая версия",
    }
    for key in ["lag1_only", "stepwise_full", "thresholded"]:
        axes[0].plot(pivot.index, pivot[key], label=labels[key], linewidth=1.8, color=colors[key])

    axes[0].set_ylabel("X_t")
    axes[0].set_title("Контроль по g=0.8: рекуррентное продолжение при сохранении малых лагов")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.2)

    for key in ["lag1_only", "stepwise_full", "thresholded"]:
        deviation = np.abs(pivot[key] - pivot["true_nonlinear"])
        axes[1].plot(pivot.index, deviation, label=labels[key], linewidth=1.8, color=colors[key])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Горизонт прогноза")
    axes[1].set_ylabel("|ошибка|")
    axes[1].grid(alpha=0.2)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

