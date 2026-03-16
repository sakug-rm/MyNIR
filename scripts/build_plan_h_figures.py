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


def build_dispersion_condition_scatter(window_level: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "stable": "#26734d",
        "low_dispersion": "#d48a00",
        "collinearity": "#b33a3a",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for failure_type, part in window_level.groupby("failure_type", as_index=False):
        ax.scatter(
            part["log10_var_omega"],
            part["log10_cond_scaled"],
            s=24,
            alpha=0.65,
            label=failure_type,
            color=colors.get(str(failure_type), "#555555"),
        )
    ax.set_xlabel("log10 Var(omega)")
    ax.set_ylabel("log10 cond(X_scaled)")
    ax.set_title("Failure types in the dispersion-condition plane")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "planH_dispersion_condition_scatter.png")


def build_risk_grid(risk_grid: pd.DataFrame, output_dir: Path) -> None:
    pivot = risk_grid.pivot(index="var_omega_band", columns="cond_scaled_band", values="degraded_share")
    pivot = pivot.reindex(index=["Q1", "Q2", "Q3", "Q4"], columns=["Q1", "Q2", "Q3", "Q4"])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("cond(X_scaled) risk band")
    ax.set_ylabel("Var(omega) risk band")
    ax.set_title("Share of degraded windows")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="#222222", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, output_dir / "planH_risk_grid.png")


def build_condition_comparison(window_level: pd.DataFrame, output_dir: Path) -> None:
    labels = ["stable", "low_dispersion", "collinearity"]
    raw_data = [
        window_level.loc[window_level["failure_type"] == label, "cond_raw"]
        .replace([np.inf], np.nan)
        .dropna()
        .clip(lower=1e-12)
        for label in labels
    ]
    scaled_data = [
        window_level.loc[window_level["failure_type"] == label, "cond_scaled"]
        .replace([np.inf], np.nan)
        .dropna()
        .clip(lower=1e-12)
        for label in labels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, data, title in [
        (axes[0], raw_data, "raw condition number"),
        (axes[1], scaled_data, "scaled condition number"),
    ]:
        ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            boxprops={"facecolor": "#d9e6f5"},
            medianprops={"color": "#333333", "linewidth": 1.5},
        )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("condition number (log scale)")
    _save(fig, output_dir / "planH_condition_comparison.png")


def build_case_profile(case_summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = case_summary.sort_values(["model", "case"]).reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)

    axes[0].bar(x - width / 2, ordered["degraded_share"], width=width, color="#b33a3a", label="degraded")
    axes[0].bar(x + width / 2, ordered["no_model_share"], width=width, color="#d48a00", label="no model")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, ordered["case"], rotation=35, ha="right")
    axes[0].set_title("Window degradation by case")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].bar(
        x - width / 2,
        ordered["median_cond_scaled"].clip(lower=1e-12),
        width=width,
        color="#1f5aa6",
        label="median cond scaled",
    )
    axes[1].bar(
        x + width / 2,
        ordered["mean_var_omega"].clip(lower=1e-12),
        width=width,
        color="#26734d",
        label="mean var omega",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, ordered["case"], rotation=35, ha="right")
    axes[1].set_title("Scaled conditioning versus signal variance")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(frameon=False)

    _save(fig, output_dir / "planH_case_profile.png")


def main() -> None:
    output_dir = ROOT / "outputs" / "planH"
    window_level = pd.read_csv(output_dir / "window_level.csv")
    case_summary = pd.read_csv(output_dir / "case_summary.csv")
    risk_grid = pd.read_csv(output_dir / "risk_grid.csv")

    build_dispersion_condition_scatter(window_level, output_dir)
    build_risk_grid(risk_grid, output_dir)
    build_condition_comparison(window_level, output_dir)
    build_case_profile(case_summary, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
