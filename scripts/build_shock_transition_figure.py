#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "real_data" / "revenue_b2"
WINDOW_LEVEL = OUT_DIR / "window_level.csv"
SECTOR_YEARLY = OUT_DIR / "sector_yearly.csv"
FIG_PATH = OUT_DIR / "shock_transition_map.png"


STATUS_ORDER = ["structural", "phase", "closed"]
STATUS_RU = {
    "structural": "структурный",
    "phase": "фазовый",
    "closed": "закрыт",
}
STATUS_COLOR = {
    "structural": "#1f77b4",
    "phase": "#ff7f0e",
    "closed": "#c44e52",
}


def compact_label(text: str) -> str:
    prefix, _, rest = text.partition(" — ")
    short = rest.strip()
    replacements = {
        "Полиграфическая деятельность, не включенная в другие группировки": "Полиграфия",
        "Производство прочих фармацевтических продуктов и изделий медицинского назначения": "Фармацевтика",
        "Производство пара и горячей воды (тепловой энергии) котельными": "Теплоснабжение",
        "Производство машин и оборудования для изготовления пищевых продуктов, включая напитки, и табачных изделий": "Пищевое оборудование",
        "Предоставление услуг по монтажу, ремонту и техническому обслуживанию подъемно транспортного оборудования": "Монтаж и сервис",
        "Деятельность в области электросвязи": "Электросвязь",
    }
    return f"{prefix} — {replacements.get(short, short)}"


def make_status(row: pd.Series) -> str:
    if row["tool_label"] == "structural_m1_m2" and row["interpretability_label"] == "interpretable":
        return "structural"
    if row["tool_label"] == "phase_trajectory" and row["interpretability_label"] == "interpretable":
        return "phase"
    return "closed"


def build_transition_frame(window_level: pd.DataFrame, pre: tuple[int, int], hit: tuple[int, int]) -> pd.DataFrame:
    pre_df = (
        window_level[(window_level["start_year"] == pre[0]) & (window_level["end_year"] == pre[1])]
        [["sector_2d", "sector_label", "status", "regime_label", "adj_r2_best"]]
        .rename(
            columns={
                "status": "pre_status",
                "regime_label": "pre_regime",
                "adj_r2_best": "pre_r2",
            }
        )
    )
    hit_df = (
        window_level[(window_level["start_year"] == hit[0]) & (window_level["end_year"] == hit[1])]
        [["sector_2d", "sector_label", "status", "regime_label", "adj_r2_best"]]
        .rename(
            columns={
                "status": "hit_status",
                "regime_label": "hit_regime",
                "adj_r2_best": "hit_r2",
            }
        )
    )
    return pre_df.merge(hit_df, on=["sector_2d", "sector_label"], how="inner")


def plot_transition_matrix(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    mat = (
        pd.crosstab(frame["pre_status"], frame["hit_status"])
        .reindex(index=STATUS_ORDER, columns=STATUS_ORDER, fill_value=0)
        .to_numpy()
    )
    vmax = max(1, int(mat.max()))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
    del im
    ax.set_xticks(range(len(STATUS_ORDER)))
    ax.set_xticklabels([STATUS_RU[s] for s in STATUS_ORDER], rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(len(STATUS_ORDER)))
    ax.set_yticklabels([STATUS_RU[s] for s in STATUS_ORDER], fontsize=9)
    ax.set_xlabel("Окно, содержащее шок")
    ax.set_ylabel("Последнее окно до шока")
    ax.set_title(title, fontsize=11, fontweight="bold")
    total = len(frame)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            count = int(mat[i, j])
            share = count / total if total else 0.0
            text_color = "white" if count >= vmax * 0.45 else "black"
            ax.text(j, i, f"{count}\n({share:.0%})", ha="center", va="center", fontsize=9, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(STATUS_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(STATUS_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_event_panel(
    ax: plt.Axes,
    yearly: pd.DataFrame,
    sectors: list[int],
    years: list[int],
    base_year: int,
    title: str,
    transition_map: dict[int, str],
) -> None:
    subset = yearly[(yearly["sector_2d"].isin(sectors)) & (yearly["year"].isin(years))].copy()
    pivot = subset.pivot_table(index="year", columns="sector_2d", values="median_revenue")
    for sector in sectors:
        values = pivot[sector] / pivot.loc[base_year, sector] * 100.0
        label = compact_label(
            subset.loc[subset["sector_2d"] == sector, "sector_label"].iloc[0]
        )
        line_label = f"{label} ({transition_map[sector]})"
        ax.plot(
            values.index,
            values.values,
            marker="o",
            linewidth=2.2,
            label=line_label,
        )
        ax.text(values.index[-1] + 0.04, values.values[-1], label, fontsize=8, va="center")
    ax.axvline(base_year + 1, color="#555555", linestyle="--", linewidth=1.3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Индекс медианной выручки\n({base_year} = 100)")
    ax.set_xlim(min(years), max(years) + 0.5)
    ax.grid(alpha=0.25)
    ax.set_xticks(years)


def main() -> None:
    window_level = pd.read_csv(WINDOW_LEVEL)
    window_level = window_level[window_level["window"] == 7].copy()
    window_level["status"] = window_level.apply(make_status, axis=1)

    yearly = pd.read_csv(SECTOR_YEARLY)

    frame_2020 = build_transition_frame(window_level, (2013, 2019), (2014, 2020))
    frame_2022 = build_transition_frame(window_level, (2015, 2021), (2016, 2022))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_transition_matrix(
        ax1,
        frame_2020,
        "Шок 2020 года: переходы аналитического статуса\n2013–2019 → 2014–2020",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    plot_transition_matrix(
        ax2,
        frame_2022,
        "Шок 2022 года: переходы аналитического статуса\n2015–2021 → 2016–2022",
    )

    ax3 = fig.add_subplot(gs[1, 0])
    plot_event_panel(
        ax3,
        yearly,
        sectors=[22, 24, 40],
        years=[2018, 2019, 2020, 2021],
        base_year=2019,
        title="2020: восстановление, усиление и закрытие окна",
        transition_map={
            22: "закрыт → структурный",
            24: "закрыт → структурный",
            40: "структурный → закрыт",
        },
    )

    ax4 = fig.add_subplot(gs[1, 1])
    plot_event_panel(
        ax4,
        yearly,
        sectors=[29, 31, 64],
        years=[2019, 2020, 2021, 2022],
        base_year=2021,
        title="2022: сохранение, фазовый сдвиг и потеря читаемости",
        transition_map={
            29: "структурный → структурный",
            31: "структурный → фазовый",
            64: "структурный → закрыт",
        },
    )

    handles = [
        plt.Line2D([0], [0], color=STATUS_COLOR["structural"], linewidth=0),
    ]
    del handles
    fig.suptitle(
        "Шоки 2020 и 2022 годов меняют не только уровень выручки, но и допустимый контур её анализа",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
