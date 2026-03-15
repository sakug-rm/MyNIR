from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.plan_g_experiment import run_plan_g_experiment, save_plan_g_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plan G: regime diagnostics before regression.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "planG")
    args = parser.parse_args()

    results = run_plan_g_experiment(steps=args.steps, window=args.window, lags=args.lags)
    output_dir = save_plan_g_results(results, args.output_dir)
    print(f"Saved Plan G results to {output_dir}")


if __name__ == "__main__":
    main()
