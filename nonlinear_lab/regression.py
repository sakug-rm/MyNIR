from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


def stepwise_selection(
    X: pd.DataFrame,
    y: Iterable[float],
    initial_list: list[str] | None = None,
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
    max_steps: int = 200,
) -> list[str]:
    """Run a simple SPSS-like stepwise variable selection by p-values."""
    included = list(initial_list or [])
    y_series = pd.Series(y)

    steps = 0
    while steps < max_steps:
        steps += 1
        changed = False

        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)

        for col in excluded:
            try:
                model = sm.OLS(y_series, sm.add_constant(X[included + [col]], has_constant="add")).fit()
                new_pval[col] = model.pvalues[col]
            except Exception:
                continue

        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                included.append(new_pval.idxmin())
                changed = True

        if included:
            try:
                model = sm.OLS(y_series, sm.add_constant(X[included], has_constant="add")).fit()
                pvals = model.pvalues.drop(labels="const", errors="ignore")
                if not pvals.empty and pvals.max() > threshold_out:
                    worst_feature = pvals.idxmax()
                    included.remove(worst_feature)
                    changed = True
            except Exception:
                pass

        if not changed:
            break

    return included


def fit_enter_with_beta(X_mat: pd.DataFrame, y: Iterable[float]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.Series]:
    """Fit an ENTER-style OLS model and standardized beta coefficients."""
    y_series = pd.Series(y)
    model = sm.OLS(y_series, sm.add_constant(X_mat, has_constant="add")).fit()

    std = X_mat.std(ddof=0)
    good_cols = std[std > 0].index.tolist()
    if not good_cols:
        return model, pd.Series(dtype=float)

    Xs = (X_mat[good_cols] - X_mat[good_cols].mean()) / X_mat[good_cols].std(ddof=0)
    y_std = y_series.std(ddof=0)
    ys = (y_series - y_series.mean()) / (y_std if y_std > 0 else 1.0)
    beta_model = sm.OLS(ys, Xs).fit()
    beta = pd.Series(beta_model.params, index=good_cols)
    return model, beta


def rolling_window_regression(
    series: np.ndarray,
    window: int = 25,
    lags: int = 10,
    method: str = "enter",
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
) -> pd.DataFrame:
    """Run rolling-window regression for a time series.

    Returns one row per valid window with R2, selected variables and estimated
    B/Beta coefficients.
    """
    from nonlinear_lab.features import make_regression_df

    x = np.asarray(series, dtype=float)
    rows: list[dict[str, object]] = []
    min_len = lags + 8
    if window < min_len:
        raise ValueError(f"window must be >= {min_len} for lags={lags}")

    for start in range(0, len(x) - window):
        end = start + window
        df = make_regression_df(x[start:end], lags=lags)
        if len(df) < 8:
            continue

        y = df["omega"]
        X_mat = df.drop(columns=["omega"])

        if method == "enter":
            model, beta = fit_enter_with_beta(X_mat, y)
            selected = list(X_mat.columns)
        elif method == "stepwise":
            selected = stepwise_selection(
                X_mat,
                y,
                threshold_in=threshold_in,
                threshold_out=threshold_out,
            )
            if not selected:
                continue
            model = sm.OLS(y, sm.add_constant(X_mat[selected], has_constant="add")).fit()

            std = X_mat[selected].std(ddof=0)
            good_cols = std[std > 0].index.tolist()
            if good_cols:
                Xs = (X_mat[good_cols] - X_mat[good_cols].mean()) / X_mat[good_cols].std(ddof=0)
                y_std = y.std(ddof=0)
                ys = (y - y.mean()) / (y_std if y_std > 0 else 1.0)
                beta_model = sm.OLS(ys, Xs).fit()
                beta = pd.Series(beta_model.params, index=good_cols)
            else:
                beta = pd.Series(dtype=float)
        else:
            raise ValueError("method must be 'enter' or 'stepwise'")

        row: dict[str, object] = {
            "start": start,
            "end": end,
            "R2": model.rsquared,
            "selected": selected,
        }
        for name, value in model.params.items():
            row[f"B_{name}"] = value
        for name, value in beta.items():
            row[f"Beta_{name}"] = value
        rows.append(row)

    out = pd.DataFrame(rows)
    for col, dtype in [("start", float), ("end", float), ("R2", float), ("selected", object)]:
        if col not in out.columns:
            out[col] = pd.Series(dtype=dtype)
    return out


def stepwise_frequency(roll_step: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Count how often each variable was selected across stepwise windows."""
    cols = ["X_n"] + [f"Lag_{idx}" for idx in range(1, lags + 1)]
    counts = {name: 0 for name in cols}

    if roll_step is None or len(roll_step) == 0 or "selected" not in roll_step.columns:
        return pd.DataFrame({"var": list(counts.keys()), "count": list(counts.values())}).sort_values(
            "count", ascending=False
        )

    for selected in roll_step["selected"].dropna():
        for feature in selected:
            if feature in counts:
                counts[feature] += 1

    return pd.DataFrame({"var": list(counts.keys()), "count": list(counts.values())}).sort_values(
        "count", ascending=False
    )
