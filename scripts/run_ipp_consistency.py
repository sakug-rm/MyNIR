from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.real_data_consistency import run_ipp_consistency_analysis, save_ipp_consistency_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consistency analysis over saved IPP routing/structural outputs.")
    parser.add_argument("--routing-csv", default="outputs/real_data/ipp_routing_adj_unsmoothed_expanded/window_features.csv")
    parser.add_argument("--structural-csv", default="outputs/real_data/ipp_structural_adj_unsmoothed_expanded/window_structures.csv")
    parser.add_argument("--output-dir", default="outputs/real_data/ipp_consistency_adj_unsmoothed_expanded")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routing = pd.read_csv(args.routing_csv)
    structural = pd.read_csv(args.structural_csv)
    results = run_ipp_consistency_analysis(routing, structural)
    out = save_ipp_consistency_results(results, args.output_dir)
    print(results["overall_summary"].to_string(index=False))
    print()
    print(results["variant_consistency"].to_string(index=False))
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
