from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.real_data import load_ipp_long
from nonlinear_lab.real_data_routing import run_ipp_routing_experiment
from nonlinear_lab.real_data_structural import run_ipp_structural_reading, save_ipp_structural_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural reading on routed IPP windows.")
    parser.add_argument("--ipp-path", default="real_data/февраль 2026 OKVED.XLS")
    parser.add_argument("--output-dir", default="outputs/real_data/ipp_structural_adj_unsmoothed")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--variant", default="adj_unsmoothed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ipp_long, _ = load_ipp_long(args.ipp_path)
    routing = run_ipp_routing_experiment(
        ipp_long,
        windows=[args.window],
        lags=args.lags,
        variant_filter=[args.variant],
    )
    results = run_ipp_structural_reading(ipp_long, routing["window_features"], lags=args.lags)
    out = save_ipp_structural_results(results, args.output_dir)
    print(results["mode_summary"].to_string(index=False))
    print()
    print(results["series_summary"].head(15).to_string(index=False))
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
