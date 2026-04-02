from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.real_data import load_ipp_long
from nonlinear_lab.real_data_routing import run_ipp_routing_experiment, save_ipp_routing_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data routing and interpretability analysis on IPP windows.")
    parser.add_argument("--ipp-path", default="real_data/февраль 2026 OKVED.XLS")
    parser.add_argument("--output-dir", default="outputs/real_data/ipp_routing_baseline")
    parser.add_argument("--windows", nargs="+", type=int, default=[24])
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--variants", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ipp_long, _ = load_ipp_long(args.ipp_path)
    results = run_ipp_routing_experiment(
        ipp_long,
        windows=args.windows,
        lags=args.lags,
        variant_filter=args.variants,
    )
    out = save_ipp_routing_results(results, args.output_dir)
    print(results["regime_summary"].to_string(index=False))
    print()
    print(results["interpretability_summary"].to_string(index=False))
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()
