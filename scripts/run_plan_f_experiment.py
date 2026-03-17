from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.plan_f_experiment import run_plan_f_experiment, save_plan_f_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plan F: PCA / PCR in lag space.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--validation-share", type=float, default=0.2)
    parser.add_argument("--active-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "planF")
    args = parser.parse_args()

    results = run_plan_f_experiment(
        steps=args.steps,
        window=args.window,
        lags=args.lags,
        rho=args.rho,
        validation_share=args.validation_share,
        active_threshold=args.active_threshold,
    )
    output_dir = save_plan_f_results(results, args.output_dir)
    print(f"Saved Plan F results to {output_dir}")


if __name__ == "__main__":
    main()
