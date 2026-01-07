"""Оркестрация экспериментов по ТЗ: симуляция, AR-модели, прогнозы, отчёт."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import compare_forecast, fit_phase_parabola
from regression import fit_segments
from simulation import (
    SimulationConfig,
    logistic_map,
    parameter_range,
    split_series,
)
from visualization import plot_forecast, plot_phase_portrait, plot_trajectory
from report import save_html_report


@dataclass
class ExperimentConfig:
    """Конфигурация полного эксперимента."""

    n_steps: int = 300
    x0: float = 0.0001
    noise_std: float = 0.0
    horizon: int = 150
    n_segments: int = 5
    step: float = 0.25  # шаг сетки A/K
    intervals: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))
    fixed_pairs: Tuple[Tuple[float, float], ...] = ((2.0, 2.0), (2.52, 2.52), (2.8, 2.8))
    max_lag: int = 10
    criterion: str = "bic"
    output_root: Path = field(default_factory=lambda: Path("outputs"))
    random_state: int | None = None


def make_param_grid(cfg: ExperimentConfig) -> List[Tuple[float, float]]:
    """Строит сетку по A/K для заданных интервалов и фиксированных точек."""
    values = []
    for lo, hi in cfg.intervals:
        values.extend(parameter_range(lo, hi, cfg.step))
    unique_vals = sorted(set(values))
    grid = [(A, K) for A in unique_vals for K in unique_vals]
    grid.extend(cfg.fixed_pairs)
    # Убираем дубликаты и сохраняем порядок
    seen = set()
    ordered = []
    for pair in grid:
        if pair not in seen:
            ordered.append(pair)
            seen.add(pair)
    return ordered


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig, path: Path) -> Path:
    path = path.with_suffix(".png")
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def run_single(
    A: float,
    K: float,
    cfg: ExperimentConfig,
    output_dir: Path,
) -> Dict:
    """Запускает один эксперимент по паре (A, K)."""
    sim_cfg = SimulationConfig(
        n_steps=cfg.n_steps,
        x0=cfg.x0,
        A=A,
        K=K,
        noise_std=cfg.noise_std,
        random_state=cfg.random_state,
    )
    df = logistic_map(sim_cfg)
    df = df[np.isfinite(df["X"])]
    if len(df) < 10:
        raise ValueError(f"Траектория деградировала (A={A}, K={K}).")

    # Траектория
    fig, ax = plt.subplots()
    plot_trajectory(df, ax=ax, label=f"A={A}, K={K}")
    traj_path = save_fig(fig, output_dir / f"trajectory_A{A}_K{K}")

    # Фазовый портрет + парабола
    parab = fit_phase_parabola(df["X"])
    fig, ax = plt.subplots()
    plot_phase_portrait(df["X"], parabola=parab, ax=ax)
    phase_path = save_fig(fig, output_dir / f"phase_A{A}_K{K}")

    # Сегменты и AR-модели
    segments = split_series(df["X"], n_segments=cfg.n_segments)
    models_step = fit_segments(segments, method="stepwise", max_lag=cfg.max_lag, criterion=cfg.criterion)
    models_enter = fit_segments(segments, method="enter", max_lag=cfg.max_lag)

    forecasts: List[Dict] = []
    for models in (models_step, models_enter):
        for model in models:
            seg = segments[model.segment_index]
            seg_start = model.segment_index * (len(df["X"]) // cfg.n_segments)
            seg_end = seg_start + len(seg)
            actual_future = df["X"].iloc[seg_end:].reset_index(drop=True)
            cmp_res = compare_forecast(
                model_params=model.params,
                lags=model.lags,
                history=seg,
                actual_future=actual_future,
                horizon=cfg.horizon,
                add_const=model.add_const,
            )

            # График прогноза
            fig, ax = plt.subplots()
            plot_forecast(
                history=seg,
                forecast=cmp_res["pred"],
                actual_future=actual_future,
                ax=ax,
            )
            fc_path = save_fig(
                fig,
                output_dir
                / f"forecast_A{A}_K{K}_seg{model.segment_index}_{model.method}",
            )

            forecasts.append(
                {
                    "method": model.method,
                    "segment": model.segment_index,
                    "lags": model.lags,
                    "adj_r2": model.adj_r2,
                    "mae": cmp_res["mae"],
                    "mse": cmp_res["mse"],
                    "forecast_path": fc_path.name,
                }
            )

    return {
        "A": A,
        "K": K,
        "trajectory_path": traj_path.name,
        "phase_path": phase_path.name,
        "parabola": parab,
        "forecasts": forecasts,
    }


def build_report(
    results: List[Dict],
    output_dir: Path,
    title: str,
) -> Path:
    """Формирует HTML-отчёт с основными графиками и метриками."""
    sections = []
    for res in results:
        forecasts_html = []
        for fc in res["forecasts"]:
            forecasts_html.append(
                f"<li>seg {fc['segment']} ({fc['method']}), lags={fc['lags']}, "
                f"adjR²={fc['adj_r2']:.3f}, MAE={fc['mae']:.4f}, "
                f"MSE={fc['mse']:.4f} — <a href='{fc['forecast_path']}'>forecast plot</a></li>"
            )
        parab = res["parabola"]
        section_body = f"""
        <p>A={res['A']}, K={res['K']}</p>
        <p>Парабола: a={parab['a']:.4f}, b={parab['b']:.4f}, c={parab['c']:.4f},
        RMSE={parab['rmse']:.4f}, R²={parab['r2']:.4f}</p>
        <p><a href="{res['trajectory_path']}">Траектория</a> | <a href="{res['phase_path']}">Фазовый портрет</a></p>
        <ul>{"".join(forecasts_html)}</ul>
        """
        sections.append({"title": f"A={res['A']}, K={res['K']}", "body": section_body})

    return save_html_report(title=title, sections=sections, output_path=output_dir / "report.html")


def run_experiments(cfg: ExperimentConfig) -> Path:
    """Запускает полный набор по сетке и строит отчёт."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(cfg.output_root / f"run_{timestamp}")

    grid = make_param_grid(cfg)
    results = []
    for idx, (A, K) in enumerate(grid, start=1):
        print(f"[{idx}/{len(grid)}] A={A}, K={K}")
        try:
            res = run_single(A, K, cfg, output_dir)
            results.append(res)
        except Exception as exc:
            print(f"Пропуск A={A}, K={K}: {exc}")

    report_path = build_report(results, output_dir, title="Отчёт по экспериментам")
    print(f"Отчёт сохранён: {report_path}")
    return report_path


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Запуск полного пайплайна по ТЗ.")
    parser.add_argument("--n-steps", type=int, default=300, help="Длина траектории.")
    parser.add_argument("--horizon", type=int, default=150, help="Длина прогноза AR.")
    parser.add_argument("--segments", type=int, default=5, help="Число сегментов ряда.")
    parser.add_argument("--step", type=float, default=0.25, help="Шаг сетки по A/K.")
    parser.add_argument("--max-lag", type=int, default=10, help="Максимальный лаг AR.")
    parser.add_argument("--criterion", type=str, default="bic", help="Критерий stepwise (aic/bic).")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Добавочный шум в симуляции.")
    parser.add_argument("--output", type=str, default="outputs", help="Каталог для результатов.")
    parser.add_argument("--random-state", type=int, default=None, help="Сид генератора.")
    args = parser.parse_args()
    return ExperimentConfig(
        n_steps=args.n_steps,
        horizon=args.horizon,
        n_segments=args.segments,
        step=args.step,
        max_lag=args.max_lag,
        criterion=args.criterion,
        noise_std=args.noise_std,
        output_root=Path(args.output),
        random_state=args.random_state,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_experiments(cfg)
