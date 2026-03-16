from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.plan_c_experiment import run_plan_c_experiment, save_plan_c_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plan C: Beta coefficients vs raw B coefficients.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--threshold-in", type=float, default=0.01)
    parser.add_argument("--threshold-out", type=float, default=0.05)
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "planC")
    args = parser.parse_args()

    results = run_plan_c_experiment(
        steps=args.steps,
        window=args.window,
        lags=args.lags,
        threshold_in=args.threshold_in,
        threshold_out=args.threshold_out,
        corr_threshold=args.corr_threshold,
    )
    output_dir = save_plan_c_results(results, args.output_dir)
    print(f"Saved Plan C results to {output_dir}")


if __name__ == "__main__":
    main()
