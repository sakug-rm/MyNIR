from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.plan_a_experiment import run_plan_a_experiment, run_plan_a_noise_sweep, save_plan_a_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Plan A: direct identification vs structural regression.")
    parser.add_argument("--steps", type=int, default=180, help="Number of time steps per synthetic trajectory.")
    parser.add_argument("--regression-window", type=int, default=25, help="Window length for structural regression.")
    parser.add_argument("--noise-sigma", type=float, default=0.02, help="Lognormal noise sigma for observed trajectories.")
    parser.add_argument("--noise-seed", type=int, default=7, help="Base seed for observation noise.")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="*",
        default=[0.0, 0.005, 0.01, 0.02, 0.05],
        help="Noise levels for the sensitivity sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/planA"),
        help="Directory where CSV summaries will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels\.regression\.linear_model")
    args = parse_args()
    results = run_plan_a_experiment(
        steps=args.steps,
        regression_window=args.regression_window,
        noise_sigma=args.noise_sigma,
        noise_seed=args.noise_seed,
    )
    output_dir = save_plan_a_results(results, args.output_dir)
    sweep = run_plan_a_noise_sweep(
        noise_levels=args.noise_levels,
        steps=args.steps,
        regression_window=args.regression_window,
        noise_seed=args.noise_seed,
    )
    sweep.to_csv(output_dir / "noise_sweep.csv", index=False)
    print(f"Saved Plan A results to {output_dir}")
    print(results["overall_summary"].to_string(index=False))
    print()
    print("Noise sweep:")
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
