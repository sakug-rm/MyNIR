from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonlinear_lab.models import generate_delay_process
from nonlinear_lab.regression import rolling_window_regression, stepwise_frequency


def main() -> None:
    series = generate_delay_process(g=0.8, steps=120)
    roll_step = rolling_window_regression(series, window=25, lags=10, method="stepwise")
    freq = stepwise_frequency(roll_step, lags=10)

    print("Delay rolling-window example")
    print(f"windows={len(roll_step)}")
    print("head:")
    print(roll_step[["start", "end", "R2", "selected"]].head().to_string(index=False))
    print("selection frequency:")
    print(freq.head().to_string(index=False))


if __name__ == "__main__":
    main()
