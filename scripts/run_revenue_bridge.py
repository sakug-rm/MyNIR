from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.revenue_bridge import run_bridge_case, save_bridge_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bridge-case between annual IPP and sector revenue.")
    parser.add_argument("--ipp-long", default="real_data/processed/ipp_long.csv")
    parser.add_argument("--b2-sector-yearly", default="outputs/real_data/revenue_b2/sector_yearly.csv")
    parser.add_argument("--output-dir", default="outputs/real_data/revenue_bridge")
    parser.add_argument("--ipp-variant", default="adj_unsmoothed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ipp_long = pd.read_csv(args.ipp_long, parse_dates=["date"])
    b2_yearly = pd.read_csv(args.b2_sector_yearly, low_memory=False)
    results = run_bridge_case(ipp_long, b2_yearly, ipp_variant=args.ipp_variant)
    out = save_bridge_results(results, args.output_dir)
    print(results["bridge_mapping"].to_string(index=False))
    print()
    print(results["bridge_comparison"].to_string(index=False))
    print()
    print(results["shock_comparison"].to_string(index=False))
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
