from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.real_data_cases import DEFAULT_CASE_CODES, run_ipp_case_cards, save_ipp_case_card_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build IPP case cards from saved routing and structural windows.")
    parser.add_argument(
        "--ipp-data",
        default="real_data/processed/ipp_long.csv",
        help="Path to prepared IPP long-format data.",
    )
    parser.add_argument(
        "--hierarchy",
        default="real_data/processed/ipp_hierarchy.csv",
        help="Path to prepared IPP hierarchy file.",
    )
    parser.add_argument(
        "--routing",
        default="outputs/real_data/ipp_routing_adj_unsmoothed_expanded/window_features.csv",
        help="Path to saved routing window features.",
    )
    parser.add_argument(
        "--structural",
        default="outputs/real_data/ipp_structural_adj_unsmoothed_expanded/window_structures.csv",
        help="Path to saved structural window features.",
    )
    parser.add_argument(
        "--variant",
        default="adj_unsmoothed",
        help="Variant for case-card structural reading.",
    )
    parser.add_argument(
        "--codes",
        nargs="*",
        default=list(DEFAULT_CASE_CODES),
        help="Series codes to include in case cards.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/real_data/ipp_case_cards",
        help="Directory where case-card artifacts will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ipp_long = pd.read_csv(Path(args.ipp_data), parse_dates=["date"])
    hierarchy = pd.read_csv(Path(args.hierarchy))
    routing = pd.read_csv(Path(args.routing), low_memory=False)
    structural = pd.read_csv(Path(args.structural), low_memory=False)

    results = run_ipp_case_cards(
        ipp_long,
        hierarchy,
        routing,
        structural,
        case_codes=tuple(args.codes),
        variant=args.variant,
    )
    output_dir = save_ipp_case_card_results(results, Path(args.output_dir))
    print(f"Saved case cards to {output_dir}")


if __name__ == "__main__":
    main()
