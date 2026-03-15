# Examples

These examples are the fastest way to start using `nonlinear_lab`.

## 1. Base quickstart

```bash
python examples/base_quickstart.py
```

Shows the minimal workflow:
- generate a base process;
- build regression features;
- run ENTER and STEPWISE analysis.

## 2. Delay rolling windows

```bash
python examples/delay_windows.py
```

Shows the short-data workflow:
- generate a delay process;
- run rolling-window STEPWISE regression;
- summarize variable-selection frequency.

## 3. Mixed full report

```bash
python examples/mixed_report.py
```

Creates a saved output directory with CSV, JSON and PNG artifacts.
