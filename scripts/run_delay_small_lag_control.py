from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.delay_small_lag_control import (  # noqa: E402
    build_delay_small_lag_control,
    plot_delay_small_lag_control,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the g=0.8 small-lag control experiment.")
    parser.add_argument("--g", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-12)
    parser.add_argument("--x0", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "review" / "delay_small_lag_control",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, paths, extra = build_delay_small_lag_control(
        g=args.g,
        steps=args.steps,
        horizon=args.horizon,
        lags=args.lags,
        threshold=args.threshold,
        x0=args.x0,
    )

    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paths.to_csv(args.output_dir / "paths.csv", index=False)
    extra["coefficients"].to_csv(args.output_dir / "coefficients.csv", index=False)
    with (args.output_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(extra["meta"], fh, ensure_ascii=False, indent=2)

    plot_delay_small_lag_control(paths, args.output_dir / "forecast_comparison.png")


if __name__ == "__main__":
    main()

