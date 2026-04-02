from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.revenue_b2 import run_b2_experiment, save_b2_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B2 sector revenue experiment.")
    parser.add_argument("--revenue-wide", default="real_data/processed/revenue_wide.csv")
    parser.add_argument("--revenue-long", default="real_data/processed/revenue_long.csv")
    parser.add_argument("--output-dir", default="outputs/real_data/revenue_b2")
    parser.add_argument("--min-firms", type=int, default=100)
    parser.add_argument("--focus-sectors", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wide = pd.read_csv(args.revenue_wide, low_memory=False)
    long = pd.read_csv(args.revenue_long, low_memory=False)
    results = run_b2_experiment(wide, long, min_firms=args.min_firms, focus_sectors=tuple(args.focus_sectors))
    out = save_b2_results(results, args.output_dir)
    print(results["coverage_summary"].to_string(index=False))
    print()
    print(results["sector_summary"][["sector_2d", "firms", "sector_name", "included"]].head(20).to_string(index=False))
    print()
    print(results["interval_summary"].head(20).to_string(index=False))
    print()
    print(results["window_summary"].head(20).to_string(index=False))
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
