from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_method_tradeoff(overall_summary: pd.DataFrame, output_dir: Path) -> None:
    overall = overall_summary[overall_summary["scope"] == "overall"].copy()
    enter = overall[overall["method"] == "enter"].iloc[0]
    pcr = overall[overall["method"] == "pcr"].iloc[0]
    metrics = ["validation_r2", "true_mass_share", "pairwise_score", "false_lag_rate"]
    labels = ["validation_r2", "true_mass_share", "pairwise_score", "false_lag_rate"]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, [enter[m] for m in metrics], width=width, color="#1f5aa6", label="ENTER")
    ax.bar(x + width / 2, [pcr[m] for m in metrics], width=width, color="#26734d", label="PCR")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fit retention vs structural interpretability")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, output_dir / "planF_method_tradeoff.png")


def build_condition_support(window_level: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(
        window_level["cond_original"],
        window_level["retained_components"],
        s=28,
        alpha=0.7,
        color="#1f5aa6",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("cond(X_scaled)")
    axes[0].set_ylabel("retained_components")
    axes[0].set_title("How many PCs are needed as collinearity grows")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        window_level["cond_reduction_ratio"],
        window_level["pcr_support_90"],
        s=28,
        alpha=0.7,
        color="#b33a3a",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("cond_reduction_ratio")
    axes[1].set_ylabel("PCR support_90")
    axes[1].set_title("Stabilization does not imply sparse back-projection")
    axes[1].grid(alpha=0.3)

    _save(fig, output_dir / "planF_condition_support.png")


def build_rho_tradeoff(rho_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(rho_summary["rho"], rho_summary["retained_components"], marker="o", linewidth=2, color="#1f5aa6")
    axes[0].plot(rho_summary["rho"], rho_summary["support_90"], marker="s", linewidth=2, color="#b33a3a")
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("count")
    axes[0].set_title("More explained variance means denser interpretation")
    axes[0].grid(alpha=0.3)
    axes[0].legend(["retained_components", "support_90"], frameon=False)

    axes[1].plot(rho_summary["rho"], rho_summary["validation_r2"], marker="o", linewidth=2, color="#26734d")
    axes[1].plot(rho_summary["rho"], rho_summary["true_mass_share"], marker="s", linewidth=2, color="#7f3c8d")
    axes[1].set_xlabel("rho")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Fit remains stable while true-mass share changes")
    axes[1].grid(alpha=0.3)
    axes[1].legend(["validation_r2", "true_mass_share"], frameon=False)

    _save(fig, output_dir / "planF_rho_tradeoff.png")


def build_loading_heatmap(component_summary: pd.DataFrame, output_dir: Path) -> None:
    hard_case = "mixed_q_2_8_gamma_0_5"
    subset = component_summary[(component_summary["case"] == hard_case) & (component_summary["component"] <= 3)].copy()
    pivot = subset.pivot(index="variable", columns="component", values="loading_share").fillna(0.0)
    order = ["X_n"] + [f"Lag_{idx}" for idx in range(1, 11) if f"Lag_{idx}" in pivot.index]
    pivot = pivot.reindex([name for name in order if name in pivot.index])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"PC{idx}" for idx in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Average loading shares in the hard mixed case")
    fig.colorbar(image, ax=ax, shrink=0.85)
    _save(fig, output_dir / "planF_loading_heatmap.png")


def main() -> None:
    output_dir = ROOT / "outputs" / "planF"
    overall_summary = pd.read_csv(output_dir / "overall_summary.csv")
    window_level = pd.read_csv(output_dir / "window_level.csv")
    rho_summary = pd.read_csv(output_dir / "rho_summary.csv")
    component_summary = pd.read_csv(output_dir / "component_summary.csv")

    build_method_tradeoff(overall_summary, output_dir)
    build_condition_support(window_level, output_dir)
    build_rho_tradeoff(rho_summary, output_dir)
    build_loading_heatmap(component_summary, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
