from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.plan_h_experiment import run_plan_h_experiment, save_plan_h_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plan H: dispersion, scale and condition-number diagnostics.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--threshold-in", type=float, default=0.01)
    parser.add_argument("--threshold-out", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "planH")
    args = parser.parse_args()

    results = run_plan_h_experiment(
        steps=args.steps,
        window=args.window,
        lags=args.lags,
        threshold_in=args.threshold_in,
        threshold_out=args.threshold_out,
    )
    output_dir = save_plan_h_results(results, args.output_dir)
    print(f"Saved Plan H results to {output_dir}")


if __name__ == "__main__":
    main()
