# nonlinear-lab

Reusable Python library for the nonlinear-process experiments in this repository.

It extracts the durable computation layer from the notebooks:
- process generation for `base`, `delay`, and `mixed` models;
- feature engineering for growth-rate regression;
- ENTER and STEPWISE regression helpers;
- lifecycle and rolling-window analysis.
- plotting helpers for reusable figures;
- report generation that saves tables and figures to disk.

The notebooks remain useful as research reports. The library is the stable base for future experiments.

## Installation

Use the existing virtual environment or install the package in editable mode:

```bash
pip install -e .
```

## Quick Example

```python
from nonlinear_lab.models import generate_base_process
from nonlinear_lab.features import make_regression_df
from nonlinear_lab.regression import fit_enter_with_beta

series = generate_base_process(a=0.8, steps=120)
df = make_regression_df(series, lags=10)
model, beta = fit_enter_with_beta(df.drop(columns=["omega"]), df["omega"])

print(model.params[["const", "X_n"]])
print(beta.head())
```

## Examples

Ready-to-run examples are available in [`examples/`](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/examples):

- `python examples/base_quickstart.py`
- `python examples/delay_windows.py`
- `python examples/mixed_report.py`

See [examples/README.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/examples/README.md) for a short guide.

## CLI Scripts

Simple entry points are provided in [`scripts/`](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/scripts):

- `python scripts/run_base_experiment.py --a 0.8 --steps 120 --lags 10`
- `python scripts/run_delay_experiment.py --g 0.8 --steps 150 --lags 10`
- `python scripts/run_mixed_experiment.py --q 2.8 --gamma 0.5 --steps 150 --lags 10`
- `python scripts/run_experiment_report.py --model mixed --q 1.5 --gamma 0.5 --steps 150 --lags 10 --window 25 --with-lifecycle --output-dir outputs/mixed_q15_g05`

Each script prints a compact regression summary and, optionally, rolling-window diagnostics.

The report script writes:
- `summary.json`
- `regression_data.csv`
- `series.png`
- `phase_portrait.png`
- optional rolling-window CSV/PNG files
- optional lifecycle CSV

## Documentation

See:
- [docs/usage.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/docs/usage.md)
- [docs/api.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/docs/api.md)
- [examples/README.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/examples/README.md)

## Tests

```bash
python -m unittest discover -s tests -v
```
