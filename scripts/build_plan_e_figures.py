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


def build_alpha_tradeoff(alpha_summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = alpha_summary.sort_values("alpha")
    x = np.arange(len(ordered))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, ordered["false_lag_rate"], width=width, color="#b33a3a", label="false_lag_rate")
    ax.bar(x, ordered["miss_rate"], width=width, color="#1f5aa6", label="miss_rate")
    ax.bar(x + width, ordered["no_model_share"], width=width, color="#7f3c8d", label="no_model_share")
    ax.set_xticks(x, [f"{alpha:.2f}" for alpha in ordered["alpha"]])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("alpha")
    ax.set_ylabel("share")
    ax.set_title("Changing alpha redistributes errors, not their mechanism")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, output_dir / "planE_alpha_tradeoff.png")


def build_exact_recovery(alpha_summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = alpha_summary.sort_values("alpha")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ordered["alpha"], ordered["exact_recovery"], marker="o", linewidth=2.5, color="#26734d")
    ax.plot(ordered["alpha"], ordered["mean_selected_count"], marker="s", linewidth=2.0, color="#e07a22")
    ax.set_xscale("linear")
    ax.set_xticks(list(ordered["alpha"]))
    ax.set_xticklabels([f"{alpha:.2f}" for alpha in ordered["alpha"]])
    ax.set_xlabel("alpha")
    ax.set_title("Exact recovery stays limited while support size expands")
    ax.grid(alpha=0.3)
    ax.legend(["exact_recovery", "mean_selected_count"], frameon=False)
    _save(fig, output_dir / "planE_exact_recovery.png")


def build_interpretability_split(interpretability_summary: pd.DataFrame, output_dir: Path) -> None:
    labels = ["interpretable", "collinearity_heavy", "low_dispersion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    palette = {
        "interpretable": "#26734d",
        "collinearity_heavy": "#b33a3a",
        "low_dispersion": "#7f3c8d",
    }

    for label in labels:
        part = interpretability_summary[interpretability_summary["interpretability"] == label].sort_values("alpha")
        if part.empty:
            continue
        axes[0].plot(part["alpha"], part["false_lag_rate"], marker="o", linewidth=2, color=palette[label], label=label)
        axes[1].plot(part["alpha"], part["exact_recovery"], marker="o", linewidth=2, color=palette[label], label=label)

    for ax in axes:
        ax.set_xticks(sorted(interpretability_summary["alpha"].unique()))
        ax.set_xticklabels([f"{alpha:.2f}" for alpha in sorted(interpretability_summary["alpha"].unique())])
        ax.grid(alpha=0.3)
        ax.set_xlabel("alpha")
    axes[0].set_ylabel("false_lag_rate")
    axes[0].set_title("False-lag burden by interpretability class")
    axes[1].set_ylabel("exact_recovery")
    axes[1].set_title("Exact structural recovery by interpretability class")
    axes[1].legend(frameon=False)
    _save(fig, output_dir / "planE_interpretability_split.png")


def build_model_profile(model_summary: pd.DataFrame, output_dir: Path) -> None:
    pivot = model_summary.pivot(index="alpha", columns="model", values="false_lag_rate").sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    palette = {"base": "#1f5aa6", "delay": "#26734d", "mixed": "#b33a3a"}
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model], marker="o", linewidth=2.2, color=palette.get(model, "#555555"), label=model)
    ax.set_xticks(list(pivot.index))
    ax.set_xticklabels([f"{alpha:.2f}" for alpha in pivot.index])
    ax.set_xlabel("alpha")
    ax.set_ylabel("false_lag_rate")
    ax.set_title("Model-specific response to p-value softening")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, output_dir / "planE_model_profile.png")


def main() -> None:
    output_dir = ROOT / "outputs" / "planE"
    alpha_summary = pd.read_csv(output_dir / "alpha_summary.csv")
    interpretability_summary = pd.read_csv(output_dir / "interpretability_summary.csv")
    model_summary = pd.read_csv(output_dir / "model_summary.csv")

    build_alpha_tradeoff(alpha_summary, output_dir)
    build_exact_recovery(alpha_summary, output_dir)
    build_interpretability_split(interpretability_summary, output_dir)
    build_model_profile(model_summary, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
