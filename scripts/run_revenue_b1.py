from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.revenue_b1 import run_b1_experiment, save_b1_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B1 experiment on revenue medians.")
    parser.add_argument("--revenue-wide", default="real_data/processed/revenue_wide.csv")
    parser.add_argument("--revenue-long", default="real_data/processed/revenue_long.csv")
    parser.add_argument("--output-dir", default="outputs/real_data/revenue_b1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revenue_wide = pd.read_csv(args.revenue_wide, low_memory=False)
    revenue_long = pd.read_csv(args.revenue_long, low_memory=False)
    results = run_b1_experiment(revenue_wide, revenue_long)
    output_dir = save_b1_results(results, args.output_dir)
    print(results["group_summary"].to_string(index=False))
    print()
    print(results["interval_summary"].to_string(index=False))
    print()
    print(results["window_summary"].to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
