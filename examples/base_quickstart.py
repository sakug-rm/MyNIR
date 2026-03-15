from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.models import generate_base_process
from nonlinear_lab.regression import fit_enter_with_beta, stepwise_selection


def main() -> None:
    series = generate_base_process(a=0.8, steps=120)
    df = make_regression_df(series, lags=10)
    X = df.drop(columns=["omega"])
    y = df["omega"]

    model, beta = fit_enter_with_beta(X, y)
    selected = stepwise_selection(X, y)

    print("Base quickstart")
    print(f"observations={len(df)}")
    print(f"R2={model.rsquared:.6f}")
    print("params:")
    print(model.params[["const", "X_n"]].to_string())
    print(f"stepwise_selected={selected}")
    print("beta:")
    print(beta.head().to_string())


if __name__ == "__main__":
    main()
