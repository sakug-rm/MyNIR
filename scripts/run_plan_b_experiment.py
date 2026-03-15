from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.plan_b_experiment import run_plan_b_experiment, save_plan_b_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plan B: false lags vs window length and cycle structure.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 15, 20, 25, 30, 40, 50])
    parser.add_argument("--lags", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--threshold-in", type=float, default=0.01)
    parser.add_argument("--threshold-out", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "planB")
    args = parser.parse_args()

    results = run_plan_b_experiment(
        steps=args.steps,
        window_sizes=args.windows,
        lag_options=args.lags,
        threshold_in=args.threshold_in,
        threshold_out=args.threshold_out,
    )
    output_dir = save_plan_b_results(results, args.output_dir)
    print(f"Saved Plan B results to {output_dir}")


if __name__ == "__main__":
    main()
