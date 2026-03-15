from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.models import generate_base_process, generate_delay_process, generate_mixed_process
from nonlinear_lab.reporting import build_experiment_report, save_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a saved experiment report with tables and plots.")
    parser.add_argument("--model", choices=["base", "delay", "mixed"], required=True, help="Model family.")
    parser.add_argument("--a", type=float, help="Base-model intensity.")
    parser.add_argument("--g", type=float, help="Delay-model intensity.")
    parser.add_argument("--q", type=float, help="Mixed-model intensity.")
    parser.add_argument("--gamma", type=float, help="Mixed-model memory coefficient.")
    parser.add_argument("--steps", type=int, default=150, help="Number of generated points.")
    parser.add_argument("--lags", type=int, default=10, help="Number of regression lags.")
    parser.add_argument("--window", type=int, default=25, help="Rolling-window size.")
    parser.add_argument("--x0", type=float, default=1e-4, help="Initial value.")
    parser.add_argument("--with-lifecycle", action="store_true", help="Include lifecycle-stage analysis.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where the report will be written.")
    args = parser.parse_args()

    if args.model == "base":
        if args.a is None:
            parser.error("--a is required for base model")
        series = generate_base_process(a=args.a, x0=args.x0, steps=args.steps)
        metadata = {"a": args.a}
    elif args.model == "delay":
        if args.g is None:
            parser.error("--g is required for delay model")
        series = generate_delay_process(g=args.g, x0=args.x0, steps=args.steps)
        metadata = {"g": args.g}
    else:
        if args.q is None or args.gamma is None:
            parser.error("--q and --gamma are required for mixed model")
        series = generate_mixed_process(q=args.q, gamma=args.gamma, x0=args.x0, steps=args.steps)
        metadata = {"q": args.q, "gamma": args.gamma}

    report = build_experiment_report(
        model_name=args.model,
        series=series,
        lags=args.lags,
        window=args.window,
        include_lifecycle=args.with_lifecycle,
        metadata=metadata,
    )
    output_path = save_experiment_report(report, args.output_dir)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
