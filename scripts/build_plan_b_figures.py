from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_false_lag_vs_rho(rho_curve: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, model in zip(axes, ["base", "delay", "mixed"]):
        subset = rho_curve[rho_curve["model"] == model].copy()
        for lags, color in [(3, "#1f5aa6"), (5, "#b33a3a"), (10, "#26734d")]:
            line = subset[subset["lags"] == lags].sort_values("mean_rho")
            if line.empty:
                continue
            ax.plot(line["mean_rho"], line["false_lag_rate"], marker="o", linewidth=2, color=color, label=f"lags={lags}")
        ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1)
        ax.axvline(2.0, color="#999999", linestyle=":", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("W / T")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("false lag rate")
    axes[0].legend(frameon=False)
    _save(fig, output_dir / "planB_false_lag_vs_rho.png")


def build_false_lag_heatmap(case_summary: pd.DataFrame, output_dir: Path) -> None:
    target_cases = [
        "base_a_2_52",
        "delay_g_1_25",
        "mixed_q_2_8_gamma_0_5",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, case in zip(axes, target_cases):
        part = case_summary[case_summary["case"] == case].copy()
        pivot = part.pivot(index="lags", columns="window", values="false_lag_rate").sort_index()
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(0.5, float(np.nanmax(pivot.values))))
        ax.set_title(case)
        ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        ax.set_xlabel("window")
        ax.set_ylabel("lags")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="false lag rate")
    _save(fig, output_dir / "planB_false_lag_heatmap.png")


def build_bsum_stability(rho_curve: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, model in zip(axes, ["base", "delay", "mixed"]):
        subset = rho_curve[rho_curve["model"] == model].copy()
        for lags, color in [(3, "#1f5aa6"), (5, "#b33a3a"), (10, "#26734d")]:
            line = subset[subset["lags"] == lags].sort_values("mean_rho")
            if line.empty:
                continue
            ax.plot(line["mean_rho"], line["std_b_sum"], marker="o", linewidth=2, color=color, label=f"lags={lags}")
        ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1)
        ax.axvline(2.0, color="#999999", linestyle=":", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("W / T")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("std(B_sum)")
    axes[0].legend(frameon=False)
    _save(fig, output_dir / "planB_bsum_stability_vs_rho.png")


def build_no_model_share(rho_curve: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, model in zip(axes, ["base", "delay", "mixed"]):
        subset = rho_curve[rho_curve["model"] == model].copy()
        for lags, color in [(3, "#1f5aa6"), (5, "#b33a3a"), (10, "#26734d")]:
            line = subset[subset["lags"] == lags].sort_values("mean_rho")
            if line.empty:
                continue
            ax.plot(line["mean_rho"], line["no_model_share"], marker="o", linewidth=2, color=color, label=f"lags={lags}")
        ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1)
        ax.axvline(2.0, color="#999999", linestyle=":", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("W / T")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("no model share")
    axes[0].legend(frameon=False)
    _save(fig, output_dir / "planB_no_model_share_vs_rho.png")


def main() -> None:
    output_dir = ROOT / "outputs" / "planB"
    case_summary = pd.read_csv(output_dir / "case_summary.csv")
    rho_curve = pd.read_csv(output_dir / "rho_curve.csv")

    build_false_lag_vs_rho(rho_curve, output_dir)
    build_false_lag_heatmap(case_summary, output_dir)
    build_bsum_stability(rho_curve, output_dir)
    build_no_model_share(rho_curve, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
