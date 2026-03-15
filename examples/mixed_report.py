from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.models import generate_mixed_process
from nonlinear_lab.reporting import build_experiment_report, save_experiment_report


def main() -> None:
    output_dir = Path("outputs/example_mixed_report")
    series = generate_mixed_process(q=1.5, gamma=0.5, steps=120)

    report = build_experiment_report(
        model_name="mixed",
        series=series,
        lags=10,
        window=25,
        include_lifecycle=True,
        metadata={"q": 1.5, "gamma": 0.5},
    )
    save_experiment_report(report, output_dir)
    print(f"Saved report to: {output_dir}")


if __name__ == "__main__":
    main()
