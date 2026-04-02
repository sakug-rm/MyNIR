from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.real_data import load_ipp_long
from nonlinear_lab.real_data_experiment import run_ipp_variant_qc, save_ipp_variant_qc_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QC across smoothed, unsmoothed and raw IPP variants.")
    parser.add_argument("--ipp-path", default="real_data/февраль 2026 OKVED.XLS")
    parser.add_argument("--output-dir", default="outputs/real_data/ipp_variant_qc")
    parser.add_argument("--windows", nargs="+", type=int, default=[24, 36, 48])
    parser.add_argument("--lags", nargs="+", type=int, default=[3, 6])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ipp_long, _ = load_ipp_long(args.ipp_path)
    results = run_ipp_variant_qc(ipp_long, windows=args.windows, lag_options=args.lags)
    save_ipp_variant_qc_results(results, args.output_dir)
    print(results["variant_summary"].to_string(index=False))
    print(f"\nSaved outputs to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
