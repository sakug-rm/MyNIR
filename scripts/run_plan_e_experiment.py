from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.plan_e_experiment import run_plan_e_experiment, save_plan_e_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plan E: Stepwise p-value sensitivity.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "planE")
    args = parser.parse_args()

    results = run_plan_e_experiment(
        steps=args.steps,
        window=args.window,
        lags=args.lags,
        alphas=args.alphas,
    )
    output_dir = save_plan_e_results(results, args.output_dir)
    print(f"Saved Plan E results to {output_dir}")


if __name__ == "__main__":
    main()
