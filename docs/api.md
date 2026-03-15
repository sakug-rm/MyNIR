# API Reference

## `nonlinear_lab.models`

### `generate_base_process(a, k=1.0, x0=1e-4, steps=200, clip_min=0.0, clip_max=None)`
Generates the base model:

\[
x_{n+1} = x_n + a x_n (k - x_n)
\]

### `generate_delay_process(g, x0=1e-4, steps=200, clip_min=0.0, clip_max=2.0)`
Generates the delayed model:

\[
x_{n+1} = x_n + g x_n (1 - x_{n-1})
\]

### `generate_mixed_process(q, gamma, x0=1e-4, steps=200, clip_min=0.0, clip_max=5.0)`
Generates the mixed model:

\[
x_{n+1} = x_n + q x_n (1 - x_n - \gamma x_{n-1})
\]

### `fixed_point(gamma)`
Returns the non-zero stationary point for the mixed model:

\[
X^* = \frac{1}{1 + \gamma}
\]

### `theoretical_coeffs(q, gamma)`
Returns the theoretical coefficients for the mixed-model growth-rate regression:
- `B_X_n_theory = -q`
- `B_Lag1_theory = -q * gamma`

## `nonlinear_lab.features`

### `growth_rate(series, zero_guard=1e-12)`
Computes:

\[
\omega_{n+1} = \frac{x_{n+1} - x_n}{x_n}
\]

### `make_regression_df(series, lags=10, zero_guard=1e-12)`
Builds a `pandas.DataFrame` with:
- `omega`
- `X_n`
- `Lag_1 ... Lag_k`

## `nonlinear_lab.regression`

### `stepwise_selection(X, y, initial_list=None, threshold_in=0.01, threshold_out=0.05, max_steps=200)`
Greedy p-value based variable selection similar to the notebook logic.

Returns a list of selected column names.

### `fit_enter_with_beta(X_mat, y)`
Fits OLS on the full design matrix and also returns standardized beta coefficients.

Returns:
- `model`: `statsmodels` fitted result
- `beta`: `pandas.Series`

### `rolling_window_regression(series, window=25, lags=10, method="enter", threshold_in=0.01, threshold_out=0.05)`
Runs repeated regressions on sliding windows.

Valid `method` values:
- `"enter"`
- `"stepwise"`

Returns a table with fit metrics and coefficients per window.

### `stepwise_frequency(roll_step, lags=10)`
Counts variable-selection frequency across the output of `rolling_window_regression(..., method="stepwise")`.

## `nonlinear_lab.lifecycle`

### `find_lifecycle_stages(series, k=1.0, min_points=5)`
Splits a trajectory into:
- startup `< 10%`
- active growth `10-50%`
- maturity `50-95%`
- plateau `> 95%`

### `stagewise_analysis(series, stages, lags=10)`
Runs ENTER and STEPWISE analysis separately for each lifecycle stage.

Returns one row per stage with:
- interval;
- sample size;
- ENTER fit;
- STEPWISE selection;
- key raw and standardized coefficients.

## `nonlinear_lab.plotting`

### `plot_series(series, title="Time Series", xlabel="n", ylabel="X")`
Builds a line chart for the generated trajectory.

### `plot_phase_portrait(series, title="Phase Portrait", xlabel="X_n", ylabel="X_{n+1}")`
Builds a scatter phase portrait from consecutive states.

### `plot_rolling_coefficients(windows, coefficient_columns, title="Rolling Coefficients", ...)`
Plots selected coefficient columns from a rolling-window result table.

### `plot_stepwise_frequency(freq, title="Stepwise Frequency", ...)`
Plots variable-selection counts as a bar chart.

### `save_figure(fig, path)`
Writes a figure to disk and closes it.

## `nonlinear_lab.reporting`

### `build_experiment_report(model_name, series, lags=10, window=None, include_lifecycle=False, metadata=None)`
Builds a structured in-memory report containing:
- raw series;
- regression dataframe;
- ENTER metrics;
- standardized beta coefficients;
- STEPWISE selected variables;
- optional rolling-window outputs;
- optional lifecycle analysis.

### `save_experiment_report(report, output_dir)`
Saves the report to disk as CSV, JSON and PNG files.
