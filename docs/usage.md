# Usage Guide

## What the library does

`nonlinear_lab` is built for the workflow used across the notebooks:

1. generate a synthetic nonlinear time series;
2. convert it to growth-rate regression data;
3. fit linear models to test structural identification;
4. inspect coefficients on full samples, lifecycle segments, or rolling windows.
5. save reusable plots and experiment reports.

## Core workflow

If you want the fastest entry point, start with the runnable examples in
[examples/README.md](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/examples/README.md).

### 1. Generate a process

```python
from nonlinear_lab.models import generate_base_process

series = generate_base_process(a=0.8, k=1.0, x0=1e-4, steps=150)
```

For other models:

```python
from nonlinear_lab.models import generate_delay_process, generate_mixed_process

delay_series = generate_delay_process(g=0.8, steps=150)
mixed_series = generate_mixed_process(q=2.8, gamma=0.5, steps=150)
```

### 2. Build regression features

```python
from nonlinear_lab.features import make_regression_df

df = make_regression_df(series, lags=10)
```

Result:
- `omega`: target growth rate;
- `X_n`: current level;
- `Lag_1 ... Lag_k`: lagged levels.

### 3. Fit ENTER regression

```python
from nonlinear_lab.regression import fit_enter_with_beta

model, beta = fit_enter_with_beta(df.drop(columns=["omega"]), df["omega"])
print(model.rsquared)
print(model.params)
print(beta)
```

Use this when you want the full linear specification without variable selection.

### 4. Fit STEPWISE selection

```python
from nonlinear_lab.regression import stepwise_selection

selected = stepwise_selection(df.drop(columns=["omega"]), df["omega"])
print(selected)
```

Use this when you want to see which variables the linear selection procedure treats as significant.

## Rolling-window analysis

```python
from nonlinear_lab.regression import rolling_window_regression, stepwise_frequency

roll_enter = rolling_window_regression(series, window=25, lags=10, method="enter")
roll_step = rolling_window_regression(series, window=25, lags=10, method="stepwise")
freq = stepwise_frequency(roll_step, lags=10)
```

This is the direct library version of the “short data” experiments from the notebooks.

Useful columns in the returned table:
- `start`, `end`: window bounds;
- `R2`: model fit for the window;
- `selected`: variables chosen in stepwise mode;
- `B_*`: raw regression coefficients;
- `Beta_*`: standardized coefficients.

## Lifecycle analysis

For S-curves and similar monotone-growth trajectories:

```python
from nonlinear_lab.lifecycle import find_lifecycle_stages, stagewise_analysis

stages = find_lifecycle_stages(series, k=1.0, min_points=5)
stage_df = stagewise_analysis(series, stages, lags=10)
print(stage_df)
```

This is useful for stable-growth scenarios. It is less meaningful for strongly chaotic trajectories.

## Clipping behavior

The notebooks used technical clipping to avoid numerical explosions or collapse outside a convenient plotting range. The library keeps this behavior explicit:

- `clip_min` defaults to `0.0`;
- `clip_max` depends on the model;
- set `clip_max=None` to disable upper clipping.

For reproducible comparison with notebook logic, keep clipping enabled. For pure mathematical exploration, consider turning it off.

## Recommended usage pattern

Use the library in three layers:

1. `models.py` for series generation;
2. `features.py` and `regression.py` for analysis;
3. `plotting.py` and `reporting.py` for outputs;
4. notebooks or custom scripts for interpretation.

That keeps the experiment logic reusable while letting each notebook stay focused on narrative and plots.

## Plotting

The plotting layer is intentionally simple and reusable:

```python
from nonlinear_lab.plotting import plot_series, plot_phase_portrait

fig, ax = plot_series(series, title="Base model trajectory")
fig, ax = plot_phase_portrait(series, title="Base model phase portrait")
```

Available helpers:
- `plot_series`
- `plot_phase_portrait`
- `plot_rolling_coefficients`
- `plot_stepwise_frequency`
- `save_figure`

Use them from notebooks, scripts, or larger batch pipelines.

## Saved experiment reports

If you want saved outputs rather than only in-memory objects, use the reporting layer:

```python
from nonlinear_lab.reporting import build_experiment_report, save_experiment_report

report = build_experiment_report(
    model_name="mixed",
    series=mixed_series,
    lags=10,
    window=25,
    include_lifecycle=True,
    metadata={"q": 1.5, "gamma": 0.5},
)
save_experiment_report(report, "outputs/mixed_q15_g05")
```

Generated files typically include:
- `summary.json`
- `regression_data.csv`
- `series.png`
- `phase_portrait.png`
- `rolling_enter.csv`
- `rolling_step.csv`
- `stepwise_frequency.csv`
- `rolling_coefficients.png`
- `stepwise_frequency.png`
- `lifecycle_analysis.csv`

Exact outputs depend on whether rolling-window and lifecycle analysis were requested.

## One-command report generation

For repeatable experiments, use the CLI report script:

```bash
python scripts/run_experiment_report.py \
  --model mixed \
  --q 1.5 \
  --gamma 0.5 \
  --steps 150 \
  --lags 10 \
  --window 25 \
  --with-lifecycle \
  --output-dir outputs/mixed_q15_g05
```

This is the recommended path when you want a stable artifact directory for comparison across runs.

## Recommended onboarding path

For a new user of the library, the shortest path is:

1. run `python examples/base_quickstart.py`
2. run `python examples/delay_windows.py`
3. run `python examples/mixed_report.py`
4. then move to the CLI scripts in [`scripts/`](/Users/v.l.gukasyan/Desktop/DIPLOM/experiments/scripts)
5. then build custom notebooks or batch runs on top of the library modules
