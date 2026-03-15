from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PRIMARY_PARAM = {
    "base": "a_mae",
    "delay": "g_mae",
    "mixed": "q_mae",
}

PRIMARY_LABEL = {
    "base": "MAE(a)",
    "delay": "MAE(g)",
    "mixed": "MAE(q)",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_noise_sweep_mae(noise_sweep: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, model in zip(axes, ["base", "delay", "mixed"]):
        subset = noise_sweep[(noise_sweep["model"] == model) & (noise_sweep["segment"] == "early")].copy()
        param = PRIMARY_PARAM[model]
        for method, color in [("direct", "#b33a3a"), ("regression", "#1f5aa6")]:
            line = subset[subset["method"] == method].sort_values("noise_level")
            ax.plot(line["noise_level"], line[param], marker="o", linewidth=2, color=color, label=method)
        ax.set_title(model)
        ax.set_xlabel("noise sigma")
        ax.set_ylabel(PRIMARY_LABEL[model])
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    _save(fig, output_dir / "planA_noise_sweep_mae.png")


def build_noise_sweep_valid_share(noise_sweep: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, model in zip(axes, ["base", "delay", "mixed"]):
        subset = noise_sweep[(noise_sweep["model"] == model) & (noise_sweep["segment"] == "early")].copy()
        for method, color in [("direct", "#b33a3a"), ("regression", "#1f5aa6")]:
            line = subset[subset["method"] == method].sort_values("noise_level")
            ax.plot(line["noise_level"], line["valid_share"], marker="o", linewidth=2, color=color, label=method)
        ax.set_title(model)
        ax.set_xlabel("noise sigma")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("valid share")
    axes[0].legend(frameon=False)
    _save(fig, output_dir / "planA_noise_sweep_valid_share.png")


def build_family_accuracy(diagnostic: pd.DataFrame, output_dir: Path) -> None:
    clean = diagnostic[diagnostic["observation"] == "clean"].copy()
    early = clean[clean["segment"] == "early"].copy()

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    width = 0.33
    direct = early[early["method"] == "direct"].set_index("model").reindex(["base", "delay", "mixed"])
    regression = early[early["method"] == "regression"].set_index("model").reindex(["base", "delay", "mixed"])
    ax.bar(x - width / 2, direct["family_accuracy"], width=width, color="#b33a3a", label="direct")
    ax.bar(x + width / 2, regression["family_accuracy"], width=width, color="#1f5aa6", label="regression")
    ax.set_xticks(x, ["base", "delay", "mixed"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("family accuracy")
    ax.set_title("Structural family recognition on clean early windows")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, output_dir / "planA_family_accuracy_clean_early.png")


def build_degenerate_recall(diagnostic: pd.DataFrame, output_dir: Path) -> None:
    clean = diagnostic[diagnostic["observation"] == "clean"].copy()
    target = clean[clean["segment"].isin(["early", "middle", "late"])].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, model in zip(axes, ["base", "delay", "mixed"]):
        part = target[target["model"] == model]
        x = np.arange(3)
        width = 0.33
        direct = part[part["method"] == "direct"].set_index("segment").reindex(["early", "middle", "late"])
        regression = part[part["method"] == "regression"].set_index("segment").reindex(["early", "middle", "late"])
        ax.bar(x - width / 2, direct["degenerate_recall"], width=width, color="#b33a3a", label="direct")
        ax.bar(x + width / 2, regression["degenerate_recall"], width=width, color="#1f5aa6", label="regression")
        ax.set_xticks(x, ["early", "middle", "late"])
        ax.set_title(model)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("degenerate recall")
    axes[0].legend(frameon=False)
    _save(fig, output_dir / "planA_degenerate_recall_clean.png")


def main() -> None:
    output_dir = ROOT / "outputs" / "planA"
    noise_sweep = pd.read_csv(output_dir / "noise_sweep.csv")
    diagnostic = pd.read_csv(output_dir / "diagnostic_overall.csv")

    build_noise_sweep_mae(noise_sweep, output_dir)
    build_noise_sweep_valid_share(noise_sweep, output_dir)
    build_family_accuracy(diagnostic, output_dir)
    build_degenerate_recall(diagnostic, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
