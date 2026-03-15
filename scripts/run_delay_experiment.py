from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.models import generate_delay_process
from nonlinear_lab.regression import fit_enter_with_beta, rolling_window_regression, stepwise_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a delay-model experiment.")
    parser.add_argument("--g", type=float, required=True, help="Delay-model intensity coefficient.")
    parser.add_argument("--x0", type=float, default=1e-4, help="Initial value.")
    parser.add_argument("--steps", type=int, default=150, help="Number of generated points.")
    parser.add_argument("--lags", type=int, default=10, help="Number of regression lags.")
    parser.add_argument("--window", type=int, default=25, help="Rolling-window size.")
    parser.add_argument("--with-windows", action="store_true", help="Compute rolling-window summary.")
    args = parser.parse_args()

    series = generate_delay_process(g=args.g, x0=args.x0, steps=args.steps)
    df = make_regression_df(series, lags=args.lags)
    X = df.drop(columns=["omega"])
    y = df["omega"]

    selected = stepwise_selection(X, y)
    model, beta = fit_enter_with_beta(X, y)

    print(f"Delay model | g={args.g} steps={args.steps} lags={args.lags}")
    print(f"observations={len(df)}")
    print(f"ENTER R2={model.rsquared:.6f}")
    print("ENTER params:")
    print(model.params.to_string())
    print("STEPWISE selected:")
    print(selected)
    if not beta.empty:
        print("Beta head:")
        print(beta.head().to_string())

    if args.with_windows:
        windows = rolling_window_regression(series, window=args.window, lags=args.lags, method="stepwise")
        print(f"rolling stepwise windows={len(windows)}")
        cols = [c for c in ["start", "end", "R2", "selected"] if c in windows.columns]
        print(windows[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
