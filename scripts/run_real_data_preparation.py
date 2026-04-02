from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.real_data import load_ipp_long, load_revenue_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IPP and revenue corpora from source workbooks.")
    parser.add_argument("--ipp-path", default="real_data/февраль 2026 OKVED.XLS")
    parser.add_argument("--revenue-path", default="real_data/Данные по выручке 2003-2022.xlsx")
    parser.add_argument("--output-dir", default="real_data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ipp_long, ipp_hierarchy = load_ipp_long(args.ipp_path)
    ipp_long.to_csv(out / "ipp_long.csv", index=False)
    ipp_hierarchy.to_csv(out / "ipp_hierarchy.csv", index=False)

    revenue_wide, revenue_long = load_revenue_corpus(args.revenue_path)
    revenue_wide.to_csv(out / "revenue_wide.csv", index=False)
    revenue_long.to_csv(out / "revenue_long.csv", index=False)

    print(f"Saved {len(ipp_long)} IPP rows to {out / 'ipp_long.csv'}")
    print(f"Saved {len(ipp_hierarchy)} IPP hierarchy rows to {out / 'ipp_hierarchy.csv'}")
    print(f"Saved {len(revenue_wide)} revenue-wide rows to {out / 'revenue_wide.csv'}")
    print(f"Saved {len(revenue_long)} revenue-long rows to {out / 'revenue_long.csv'}")


if __name__ == "__main__":
    main()
