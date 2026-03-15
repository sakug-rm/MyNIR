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


def build_confusion_heatmap(confusion: pd.DataFrame, output_dir: Path) -> None:
    matrix = confusion.set_index("true_label")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_title("Predicted regime vs true regime")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="windows")
    _save(fig, output_dir / "planG_confusion_matrix.png")


def build_downstream_false_lags(downstream: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    order = ["all_windows", "predicted_filtered", "oracle_filtered"]
    labels = ["all", "predicted", "oracle"]
    for ax, model in zip(axes, ["base", "delay", "mixed"]):
        part = downstream[downstream["model"] == model].set_index("scope").reindex(order)
        ax.bar(labels, part["false_lag_rate"], color=["#999999", "#1f5aa6", "#26734d"])
        ax.set_title(model)
        ax.set_ylim(0, max(0.5, float(np.nanmax(downstream["false_lag_rate"])) * 1.1))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("false lag rate")
    _save(fig, output_dir / "planG_false_lag_reduction.png")


def build_downstream_no_model(downstream: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    order = ["all_windows", "predicted_filtered", "oracle_filtered"]
    labels = ["all", "predicted", "oracle"]
    for ax, model in zip(axes, ["base", "delay", "mixed"]):
        part = downstream[downstream["model"] == model].set_index("scope").reindex(order)
        ax.bar(labels, part["no_model_share"], color=["#999999", "#1f5aa6", "#26734d"])
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("no model share")
    _save(fig, output_dir / "planG_no_model_share.png")


def build_tool_routing(tool_summary: pd.DataFrame, output_dir: Path) -> None:
    pivot = tool_summary.pivot(index="true_tool", columns="recommended_tool", values="windows").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="OrRd")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Tool routing agreement")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="windows")
    _save(fig, output_dir / "planG_tool_routing.png")


def main() -> None:
    output_dir = ROOT / "outputs" / "planG"
    confusion = pd.read_csv(output_dir / "confusion_matrix.csv")
    downstream = pd.read_csv(output_dir / "downstream_summary.csv")
    tool_summary = pd.read_csv(output_dir / "tool_summary.csv")

    build_confusion_heatmap(confusion, output_dir)
    build_downstream_false_lags(downstream, output_dir)
    build_downstream_no_model(downstream, output_dir)
    build_tool_routing(tool_summary, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
