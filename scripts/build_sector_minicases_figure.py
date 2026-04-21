from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


YEARLY_PATH = Path("outputs/real_data/revenue_b2/sector_yearly.csv")
WINDOW_LEVEL_PATH = Path("outputs/real_data/revenue_b2/window_level.csv")
OUTPUT_PATH = Path("outputs/real_data/revenue_b2/sector_minicases_overview.png")
SECTORS = [63, 29, 52]
LABELS = {
    63: "63: хранение зерна",
    29: "29: пищевое машиностроение",
    52: "52: розничная торговля",
}
COLORS = {63: "#1b6ca8", 29: "#c55a11", 52: "#6b8e23"}
TOOL_COLORS = {
    "structural_m1_m2": "#4c9f70",
    "phase_trajectory": "#d18f1b",
    "do_not_read_regression": "#b85450",
}
TOOL_LABELS = {
    "structural_m1_m2": "структурное чтение",
    "phase_trajectory": "фазовое чтение",
    "do_not_read_regression": "окно закрыто",
}


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Serif"
    yearly = pd.read_csv(YEARLY_PATH)
    windows = pd.read_csv(WINDOW_LEVEL_PATH)

    fig = plt.figure(figsize=(12.0, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.55], wspace=0.16)
    ax_line = fig.add_subplot(gs[0, 0])
    ax_diag = fig.add_subplot(gs[0, 1])

    for sector in SECTORS:
        part = yearly[(yearly["sector_2d_code"] == sector) & (yearly["year"].between(2014, 2022))].copy()
        part = part.sort_values("year")
        idx = part["median_revenue"].astype(float) / float(part["median_revenue"].iloc[0]) * 100.0
        ax_line.plot(
            part["year"],
            idx,
            marker="o",
            linewidth=2.4,
            color=COLORS[sector],
            label=LABELS[sector],
        )
    for shock_year in [2020, 2022]:
        ax_line.axvline(shock_year, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
    ax_line.set_title("Индекс медианной выручки, 2014 = 100", fontsize=11)
    ax_line.set_xlabel("Год")
    ax_line.set_ylabel("Индекс")
    ax_line.grid(alpha=0.25)
    ax_line.legend(frameon=False, fontsize=9, loc="upper left")

    ax_diag.set_xlim(0, 3)
    ax_diag.set_ylim(0, 3)
    ax_diag.set_xticks([0.5, 1.5, 2.5])
    ax_diag.set_xticklabels(["2014–2020", "2015–2021", "2016–2022"], fontsize=9)
    ax_diag.set_yticks([2.5, 1.5, 0.5])
    ax_diag.set_yticklabels([LABELS[63], LABELS[29], LABELS[52]], fontsize=9)
    ax_diag.set_title("Короткие окна W = 7: допустимый контур чтения", fontsize=11)

    sector_to_y = {63: 2, 29: 1, 52: 0}
    for sector in SECTORS:
        diag = windows[
            (windows["sector_2d"] == sector)
            & (windows["window"] == 7)
            & (windows["start_year"].isin([2014, 2015, 2016]))
        ].sort_values("start_year")
        y = sector_to_y[sector]
        for x, (_, rec) in enumerate(diag.iterrows()):
            color = TOOL_COLORS.get(rec["tool_label"], "#cccccc")
            rect = patches.FancyBboxPatch(
                (x + 0.04, y + 0.08),
                0.92,
                0.84,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                linewidth=0.8,
                edgecolor="#666666",
                facecolor=color,
                alpha=0.95,
            )
            ax_diag.add_patch(rect)
            short = TOOL_LABELS.get(rec["tool_label"], rec["tool_label"]).replace(" чтение", "")
            ax_diag.text(
                x + 0.08,
                y + 0.60,
                short,
                ha="left",
                va="center",
                fontsize=8.8,
                color="white",
                fontweight="bold",
            )
            ax_diag.text(
                x + 0.08,
                y + 0.30,
                f"$R^2_{{adj}}={rec['adj_r2_best']:.2f}$",
                ha="left",
                va="center",
                fontsize=8.6,
                color="white",
            )

    ax_diag.text(
        0.02,
        -0.12,
        "63 остаётся читаемым во всех поздних окнах; 29 сохраняет читаемость, но слабее; 52 систематически закрыт.",
        transform=ax_diag.transAxes,
        fontsize=8.8,
    )
    fig.suptitle("Три мини-кейса отраслевой выручки", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
