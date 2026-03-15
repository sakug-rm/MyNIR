from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.regression import fit_enter_with_beta, stepwise_selection


def find_lifecycle_stages(series: np.ndarray, k: float = 1.0, min_points: int = 5) -> dict[str, tuple[int, int] | None]:
    """Split an S-curve into startup, growth, maturity and plateau stages."""
    x = np.asarray(series, dtype=float)

    def bounds(indexes: np.ndarray) -> tuple[int, int] | None:
        if len(indexes) < min_points:
            return None
        return int(indexes[0]), int(indexes[-1] + 1)

    stages = {
        "1. Зарождение (<10%)": bounds(np.where(x < 0.1 * k)[0]),
        "2. Активный рост (10-50%)": bounds(np.where((x >= 0.1 * k) & (x < 0.5 * k))[0]),
        "3. Насыщение (50-95%)": bounds(np.where((x >= 0.5 * k) & (x < 0.95 * k))[0]),
        "4. Плато (>95%)": bounds(np.where(x >= 0.95 * k)[0]),
    }
    return {name: interval for name, interval in stages.items() if interval is not None}


def stagewise_analysis(series: np.ndarray, stages: dict[str, tuple[int, int] | None], lags: int = 10) -> pd.DataFrame:
    """Run ENTER and STEPWISE regression separately for each lifecycle stage."""
    rows: list[dict[str, object]] = []

    for stage_name, bounds in stages.items():
        if bounds is None:
            continue
        start, end = bounds
        df = make_regression_df(np.asarray(series)[start:end], lags=lags)
        if len(df) < 8:
            continue

        y = df["omega"]
        X_mat = df.drop(columns=["omega"])
        enter_model, enter_beta = fit_enter_with_beta(X_mat, y)

        selected = stepwise_selection(X_mat, y)
        step_r2 = np.nan
        if selected:
            step_model = sm.OLS(y, sm.add_constant(X_mat[selected], has_constant="add")).fit()
            step_r2 = step_model.rsquared

        rows.append(
            {
                "stage": stage_name,
                "interval": f"{start}-{end}",
                "n_obs": len(df),
                "ENTER_R2": enter_model.rsquared,
                "STEP_selected": selected,
                "STEP_R2": step_r2,
                "ENTER_B_Xn": enter_model.params.get("X_n", np.nan),
                "ENTER_B_Lag1": enter_model.params.get("Lag_1", np.nan),
                "ENTER_Beta_Xn": enter_beta.get("X_n", np.nan),
                "ENTER_Beta_Lag1": enter_beta.get("Lag_1", np.nan),
            }
        )

    return pd.DataFrame(rows)
