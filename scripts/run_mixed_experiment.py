from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.features import make_regression_df
from nonlinear_lab.lifecycle import find_lifecycle_stages, stagewise_analysis
from nonlinear_lab.models import fixed_point, generate_mixed_process, theoretical_coeffs
from nonlinear_lab.regression import (
    fit_enter_with_beta,
    rolling_window_regression,
    stepwise_frequency,
    stepwise_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mixed-model experiment.")
    parser.add_argument("--q", type=float, required=True, help="Mixed-model intensity coefficient.")
    parser.add_argument("--gamma", type=float, required=True, help="Memory coefficient.")
    parser.add_argument("--x0", type=float, default=1e-4, help="Initial value.")
    parser.add_argument("--steps", type=int, default=150, help="Number of generated points.")
    parser.add_argument("--lags", type=int, default=10, help="Number of regression lags.")
    parser.add_argument("--window", type=int, default=25, help="Rolling-window size.")
    parser.add_argument("--with-windows", action="store_true", help="Compute rolling-window summary.")
    parser.add_argument("--with-lifecycle", action="store_true", help="Compute lifecycle-stage summary.")
    args = parser.parse_args()

    series = generate_mixed_process(q=args.q, gamma=args.gamma, x0=args.x0, steps=args.steps)
    df = make_regression_df(series, lags=args.lags)
    X = df.drop(columns=["omega"])
    y = df["omega"]

    selected = stepwise_selection(X, y)
    model, beta = fit_enter_with_beta(X, y)

    print(f"Mixed model | q={args.q} gamma={args.gamma} steps={args.steps} lags={args.lags}")
    print(f"fixed_point={fixed_point(args.gamma)}")
    print(f"theoretical={theoretical_coeffs(args.q, args.gamma)}")
    print(f"observations={len(df)}")
    print(f"ENTER R2={model.rsquared:.6f}")
    print("ENTER params:")
    print(model.params.to_string())
    print("STEPWISE selected:")
    print(selected)
    if not beta.empty:
        print("Beta head:")
        print(beta.head().to_string())

    if args.with_lifecycle:
        stages = find_lifecycle_stages(series)
        if stages:
            stage_df = stagewise_analysis(series, stages, lags=args.lags)
            if not stage_df.empty:
                print("Lifecycle analysis:")
                print(stage_df.to_string(index=False))
            else:
                print("Lifecycle analysis: no valid stage regressions.")
        else:
            print("Lifecycle analysis: no stages detected.")

    if args.with_windows:
        roll_step = rolling_window_regression(series, window=args.window, lags=args.lags, method="stepwise")
        print(f"rolling stepwise windows={len(roll_step)}")
        cols = [c for c in ["start", "end", "R2", "selected"] if c in roll_step.columns]
        print(roll_step[cols].head().to_string(index=False))
        freq = stepwise_frequency(roll_step, lags=args.lags)
        print("Selection frequency:")
        print(freq.head().to_string(index=False))


if __name__ == "__main__":
    main()
