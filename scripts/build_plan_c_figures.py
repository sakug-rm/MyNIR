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


def build_ranking_accuracy(case_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    x = np.arange(len(case_summary))
    labels = case_summary["case"].tolist()
    width = 0.38

    axes[0].bar(x - width / 2, case_summary["problem_correct_top_b"], width=width, color="#b33a3a", label="|B|")
    axes[0].bar(x + width / 2, case_summary["problem_correct_top_beta"], width=width, color="#1f5aa6", label="|Beta|")
    axes[0].set_title("Correct top predictor in problem windows")
    axes[0].set_ylabel("share")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, case_summary["problem_topk_b"], width=width, color="#b33a3a", label="|B|")
    axes[1].bar(x + width / 2, case_summary["problem_topk_beta"], width=width, color="#1f5aa6", label="|Beta|")
    axes[1].set_title("Core-structure recovery in problem windows")
    axes[1].set_ylabel("top-k recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    _save(fig, output_dir / "planC_ranking_accuracy.png")


def build_pairwise_separation(case_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ordered = case_summary.sort_values(["model", "case"]).reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.38
    ax.bar(x - width / 2, ordered["problem_pairwise_b"], width=width, color="#b33a3a", label="|B|")
    ax.bar(x + width / 2, ordered["problem_pairwise_beta"], width=width, color="#1f5aa6", label="|Beta|")
    ax.set_ylabel("pairwise true-vs-false score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, ordered["case"], rotation=35, ha="right")
    ax.set_title("True-vs-false predictor separation in problem windows")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir / "planC_pairwise_separation.png")


def build_true_false_boxplot(window_variable_level: pd.DataFrame, output_dir: Path) -> None:
    problem = window_variable_level[window_variable_level["problem_window"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, column, title in [
        (axes[0], "abs_B", "|B| in problem windows"),
        (axes[1], "abs_Beta", "|Beta| in problem windows"),
    ]:
        true_values = problem.loc[problem["is_true_predictor"] == 1, column].to_numpy()
        false_values = problem.loc[problem["is_true_predictor"] == 0, column].to_numpy()
        ax.boxplot(
            [true_values, false_values],
            tick_labels=["true", "false"],
            patch_artist=True,
            boxprops={"facecolor": "#d9e6f5"},
            medianprops={"color": "#333333", "linewidth": 1.5},
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("absolute score")
    _save(fig, output_dir / "planC_true_false_boxplot.png")


def build_frequency_alignment(frequency_summary: pd.DataFrame, output_dir: Path) -> None:
    problem = frequency_summary.copy()
    rows = []
    for (model, case), part in problem.groupby(["model", "case"], as_index=False):
        true_predictors = set(part.loc[part["is_true_predictor"] == 1, "variable"])
        k = len(true_predictors)
        for metric in ["problem_mean_abs_B", "problem_mean_abs_Beta", "problem_selection_rate"]:
            ranked = set(part.sort_values(metric, ascending=False)["variable"].head(k))
            rows.append(
                {
                    "case": case,
                    "metric": metric,
                    "topk_recall": len(ranked & true_predictors) / max(k, 1),
                }
            )
    chart = pd.DataFrame(rows)
    pivot = chart.pivot(index="case", columns="metric", values="topk_recall").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(pivot))
    width = 0.24
    mapping = [
        ("problem_mean_abs_B", "|B|", "#b33a3a"),
        ("problem_mean_abs_Beta", "|Beta|", "#1f5aa6"),
        ("problem_selection_rate", "frequency", "#26734d"),
    ]
    for idx, (column, label, color) in enumerate(mapping):
        ax.bar(x + (idx - 1) * width, pivot[column], width=width, label=label, color=color)
    ax.set_ylabel("top-k recall by case")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, pivot.index, rotation=35, ha="right")
    ax.set_title("Case-level core recovery: local ranking vs selection frequency")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir / "planC_frequency_alignment.png")


def main() -> None:
    output_dir = ROOT / "outputs" / "planC"
    case_summary = pd.read_csv(output_dir / "case_summary.csv")
    window_variable_level = pd.read_csv(output_dir / "window_variable_level.csv")
    frequency_summary = pd.read_csv(output_dir / "frequency_summary.csv")

    build_ranking_accuracy(case_summary, output_dir)
    build_pairwise_separation(case_summary, output_dir)
    build_true_false_boxplot(window_variable_level, output_dir)
    build_frequency_alignment(frequency_summary, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
