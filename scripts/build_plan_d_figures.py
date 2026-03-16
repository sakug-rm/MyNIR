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


def build_overall_tradeoff(overall_summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = overall_summary.copy()
    x = np.arange(len(ordered))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    axes[0].bar(x - width / 2, ordered["hit_rate"], width=width, color="#1f5aa6", label="hit_rate")
    axes[0].bar(x + width / 2, ordered["false_lag_rate"], width=width, color="#b33a3a", label="false_lag_rate")
    axes[0].set_xticks(x, ordered["method"], rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("share")
    axes[0].set_title("Structure recovery vs false-lag burden")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, ordered["sign_correct_share"], width=width, color="#26734d", label="sign_correct_share")
    axes[1].bar(x + width / 2, ordered["coef_mae_true"], width=width, color="#7f3c8d", label="coef_mae_true")
    axes[1].set_xticks(x, ordered["method"], rotation=25, ha="right")
    axes[1].set_title("Sign stability and true-coefficient error")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(frameon=False)

    _save(fig, output_dir / "planD_overall_tradeoff.png")


def build_case_false_lags(case_summary: pd.DataFrame, output_dir: Path) -> None:
    pivot = case_summary.pivot(index="case", columns="method", values="false_lag_rate").fillna(0.0)
    methods = [name for name in ["stepwise", "lasso", "elastic_net", "enter", "ridge"] if name in pivot.columns]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(pivot))
    width = 0.15
    palette = {
        "stepwise": "#1f5aa6",
        "lasso": "#26734d",
        "elastic_net": "#b33a3a",
        "enter": "#7f3c8d",
        "ridge": "#e07a22",
    }
    for idx, method in enumerate(methods):
        ax.bar(x + (idx - (len(methods) - 1) / 2) * width, pivot[method], width=width, label=method, color=palette[method])
    ax.set_xticks(x, pivot.index, rotation=35, ha="right")
    ax.set_ylabel("false_lag_rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("False-lag burden by case and method")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    _save(fig, output_dir / "planD_case_false_lags.png")


def build_mixed_tradeoff(case_summary: pd.DataFrame, output_dir: Path) -> None:
    mixed = case_summary[case_summary["model"] == "mixed"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {
        "stepwise": "#1f5aa6",
        "lasso": "#26734d",
        "elastic_net": "#b33a3a",
        "enter": "#7f3c8d",
        "ridge": "#e07a22",
    }
    for method, part in mixed.groupby("method"):
        ax.scatter(
            part["false_lag_rate"],
            part["hit_rate"],
            s=70,
            color=palette.get(method, "#555555"),
            label=method,
            alpha=0.9,
        )
        for _, row in part.iterrows():
            ax.annotate(row["case"].replace("mixed_", ""), (row["false_lag_rate"], row["hit_rate"]), fontsize=8, alpha=0.8)
    ax.set_xlabel("false_lag_rate")
    ax.set_ylabel("hit_rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_title("Mixed-model trade-off: keeping true lags vs suppressing false ones")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, output_dir / "planD_mixed_tradeoff.png")


def build_stage_profile(stage_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    stage_order = ["early", "middle", "late"]
    methods = ["stepwise", "lasso", "elastic_net", "enter", "ridge"]
    palette = {
        "stepwise": "#1f5aa6",
        "lasso": "#26734d",
        "elastic_net": "#b33a3a",
        "enter": "#7f3c8d",
        "ridge": "#e07a22",
    }
    stage_positions = np.arange(len(stage_order))

    for method in methods:
        part = stage_summary[stage_summary["method"] == method].set_index("stage").reindex(stage_order)
        axes[0].plot(stage_positions, part["false_lag_rate"], marker="o", linewidth=2, label=method, color=palette[method])
        axes[1].plot(stage_positions, part["coef_mae"], marker="o", linewidth=2, label=method, color=palette[method])

    axes[0].set_xticks(stage_positions, stage_order)
    axes[0].set_ylabel("false_lag_rate")
    axes[0].set_title("False-lag burden across trajectory stages")
    axes[0].grid(alpha=0.3)

    axes[1].set_xticks(stage_positions, stage_order)
    axes[1].set_ylabel("coef_mae")
    axes[1].set_title("All-coefficient error across trajectory stages")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False, ncol=2)

    _save(fig, output_dir / "planD_stage_profile.png")


def main() -> None:
    output_dir = ROOT / "outputs" / "planD"
    overall_summary = pd.read_csv(output_dir / "overall_summary.csv")
    case_summary = pd.read_csv(output_dir / "case_summary.csv")
    stage_summary = pd.read_csv(output_dir / "stage_summary.csv")

    build_overall_tradeoff(overall_summary, output_dir)
    build_case_false_lags(case_summary, output_dir)
    build_mixed_tradeoff(case_summary, output_dir)
    build_stage_profile(stage_summary, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
