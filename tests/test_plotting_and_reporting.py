import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from nonlinear_lab.models import generate_base_process
from nonlinear_lab.plotting import (
    plot_phase_portrait,
    plot_series,
    plot_stepwise_frequency,
)
from nonlinear_lab.reporting import build_experiment_report, save_experiment_report


class PlottingTests(unittest.TestCase):
    def test_plot_series_returns_figure_and_axes(self) -> None:
        series = generate_base_process(a=0.8, steps=20)
        fig, ax = plot_series(series, title="Series")
        self.assertEqual(ax.get_title(), "Series")
        self.assertEqual(len(ax.lines), 1)
        fig.clf()

    def test_plot_phase_portrait_returns_scatter_plot(self) -> None:
        series = generate_base_process(a=0.8, steps=20)
        fig, ax = plot_phase_portrait(series, title="Phase")
        self.assertEqual(ax.get_title(), "Phase")
        self.assertEqual(len(ax.collections), 1)
        fig.clf()

    def test_plot_stepwise_frequency_draws_bars(self) -> None:
        freq = pd.DataFrame({"var": ["X_n", "Lag_1"], "count": [3, 1]})
        fig, ax = plot_stepwise_frequency(freq, title="Freq")
        self.assertEqual(ax.get_title(), "Freq")
        self.assertEqual(len(ax.patches), 2)
        fig.clf()


class ReportingTests(unittest.TestCase):
    def test_build_experiment_report_contains_expected_sections(self) -> None:
        report = build_experiment_report(model_name="base", series=generate_base_process(a=0.8, steps=40), lags=5)
        self.assertIn("series", report)
        self.assertIn("enter_params", report)
        self.assertIn("stepwise_selected", report)

    def test_save_experiment_report_writes_outputs(self) -> None:
        report = build_experiment_report(model_name="base", series=generate_base_process(a=0.8, steps=40), lags=5)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            save_experiment_report(report, output_dir)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "regression_data.csv").exists())
            self.assertTrue((output_dir / "series.png").exists())
            self.assertTrue((output_dir / "phase_portrait.png").exists())


if __name__ == "__main__":
    unittest.main()
