from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.lifecycle import find_lifecycle_stages, stagewise_analysis
from nonlinear_lab.plotting import (
    plot_phase_portrait,
    plot_rolling_coefficients,
    plot_series,
    plot_stepwise_frequency,
    save_figure,
)
from nonlinear_lab.regression import (
    fit_enter_with_beta,
    rolling_window_regression,
    stepwise_frequency,
    stepwise_selection,
)


def build_experiment_report(
    model_name: str,
    series: np.ndarray,
    lags: int = 10,
    window: int | None = None,
    include_lifecycle: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    df = make_regression_df(series, lags=lags)
    X = df.drop(columns=["omega"])
    y = df["omega"]
    enter_model, beta = fit_enter_with_beta(X, y)
    selected = stepwise_selection(X, y)

    report: dict[str, Any] = {
        "model_name": model_name,
        "metadata": metadata or {},
        "series": np.asarray(series, dtype=float),
        "regression_df": df,
        "enter_r2": float(enter_model.rsquared),
        "enter_params": {k: float(v) for k, v in enter_model.params.items()},
        "beta_params": {k: float(v) for k, v in beta.items()},
        "stepwise_selected": selected,
    }

    if window is not None:
        roll_enter = rolling_window_regression(series, window=window, lags=lags, method="enter")
        roll_step = rolling_window_regression(series, window=window, lags=lags, method="stepwise")
        report["rolling_enter"] = roll_enter
        report["rolling_step"] = roll_step
        report["stepwise_frequency"] = stepwise_frequency(roll_step, lags=lags)

    if include_lifecycle:
        stages = find_lifecycle_stages(series)
        report["lifecycle_stages"] = stages
        report["lifecycle_analysis"] = stagewise_analysis(series, stages, lags=lags) if stages else pd.DataFrame()

    return report


def save_experiment_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name": report["model_name"],
        "metadata": report.get("metadata", {}),
        "enter_r2": report["enter_r2"],
        "enter_params": report["enter_params"],
        "beta_params": report["beta_params"],
        "stepwise_selected": report["stepwise_selected"],
    }

    with (output / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    report["regression_df"].to_csv(output / "regression_data.csv", index=False)

    fig, _ = plot_series(report["series"], title=f"{report['model_name']} series")
    save_figure(fig, output / "series.png")

    fig, _ = plot_phase_portrait(report["series"], title=f"{report['model_name']} phase portrait")
    save_figure(fig, output / "phase_portrait.png")

    if "rolling_enter" in report:
        report["rolling_enter"].to_csv(output / "rolling_enter.csv", index=False)
        report["rolling_step"].to_csv(output / "rolling_step.csv", index=False)
        report["stepwise_frequency"].to_csv(output / "stepwise_frequency.csv", index=False)

        coefficient_columns = [col for col in ["B_X_n", "B_Lag_1"] if col in report["rolling_enter"].columns]
        if coefficient_columns:
            fig, _ = plot_rolling_coefficients(
                report["rolling_enter"],
                coefficient_columns=coefficient_columns,
                title=f"{report['model_name']} rolling coefficients",
            )
            save_figure(fig, output / "rolling_coefficients.png")

        fig, _ = plot_stepwise_frequency(report["stepwise_frequency"], title=f"{report['model_name']} stepwise frequency")
        save_figure(fig, output / "stepwise_frequency.png")

    if "lifecycle_analysis" in report:
        report["lifecycle_analysis"].to_csv(output / "lifecycle_analysis.csv", index=False)

    return output
