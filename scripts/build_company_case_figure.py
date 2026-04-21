from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


DATA_PATH = Path("real_data/processed/revenue_wide.csv")
OUTPUT_PATH = Path("outputs/real_data/revenue_b2/company_case_anchors.png")
YEARS = list(range(2014, 2023))
SECTOR_COLORS = {"63": "#1b6ca8", "29": "#c55a11", "52": "#6b8e23"}
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
REGIME_LABELS = {
    "oscillatory": "колебательный",
    "shock_transition": "шоковый переход",
    "plateau_like": "низкодисперсионный",
    "monotone_growth": "монотонный рост",
    "collapse_like": "деградационный",
}
CASES = [
    {
        "sector": "63",
        "sector_label": "63: хранение зерна",
        "company_name": '"ЕЛАНСКИЙ ЭЛЕВАТОР", ОТКРЫТОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО',
        "company_label": "ОАО «Еланский элеватор»",
    },
    {
        "sector": "29",
        "sector_label": "29: пищевое машиностроение",
        "company_name": '"ТЭСМО", АКЦИОНЕРНОЕ ОБЩЕСТВО',
        "company_label": "АО «ТЭСМО»",
    },
    {
        "sector": "52",
        "sector_label": "52: розничная торговля",
        "company_name": "ШЕСТИХИНСКОЕ ПОТРЕБИТЕЛЬСКОЕ ОБЩЕСТВО",
        "company_label": "Шестихинское потребительское общество",
    },
]


def growth_rate(series: np.ndarray) -> np.ndarray:
    return np.asarray([(series[i + 1] - series[i]) / series[i] for i in range(len(series) - 1)], dtype=float)


def make_reg_df(series: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"omega": growth_rate(series[1:]), "X_n": series[1:-1], "Lag_1": series[:-2]})


def interval_regime(block: np.ndarray) -> str:
    omega = growth_rate(block)
    diffs = np.diff(block)
    if len(diffs) == 0:
        return "plateau_like"
    turning_rate = float(np.mean(np.sign(diffs[1:]) != np.sign(diffs[:-1]))) if len(diffs) >= 2 else 0.0
    rel_range = float((np.max(block) - np.min(block)) / max(np.mean(block), 1e-12))
    var_omega = float(np.var(omega, ddof=0)) if len(omega) else 0.0
    median_abs_omega = float(np.median(np.abs(omega))) if len(omega) else 0.0
    max_abs_omega = float(np.max(np.abs(omega))) if len(omega) else 0.0
    net_change = float(block[-1] / block[0] - 1.0) if abs(block[0]) > 1e-12 else 0.0
    if rel_range < 0.08 or var_omega < 0.0025:
        return "plateau_like"
    if block[-1] / max(np.max(block), 1e-12) < 0.72 and net_change < -0.1:
        return "collapse_like"
    if max_abs_omega > max(0.25, 2.5 * max(median_abs_omega, 1e-12)):
        return "shock_transition"
    if turning_rate >= 0.4:
        return "oscillatory"
    if net_change >= 0.08:
        return "monotone_growth"
    if net_change <= -0.08:
        return "monotone_decline"
    return "shock_transition"


def cond_and_corr(reg_df: pd.DataFrame) -> tuple[float, float]:
    X = reg_df.drop(columns=["omega"]).to_numpy(dtype=float)
    scaled = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    scaled = scaled / std
    cond = float(np.linalg.cond(scaled))
    corr = np.corrcoef(scaled, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return cond, float(np.nanmean(np.abs(upper))) if upper.size else 0.0


def interpretability(regime_label: str, var_omega: float, cond_scaled: float, mean_abs_corr: float) -> str:
    if regime_label in {"plateau_like", "collapse_like"}:
        return "low_dispersion"
    if not np.isfinite(var_omega) or var_omega < 0.0025:
        return "low_dispersion"
    if np.isfinite(cond_scaled) and cond_scaled > 80:
        return "collinearity_heavy"
    if np.isfinite(mean_abs_corr) and mean_abs_corr > 0.92:
        return "collinearity_heavy"
    return "interpretable"


def tool_label(regime_label: str, interpretability_label: str) -> str:
    if interpretability_label == "low_dispersion":
        return "do_not_read_regression"
    if regime_label == "shock_transition":
        return "phase_trajectory"
    if interpretability_label == "collinearity_heavy":
        return "enter_beta_bsum"
    return "structural_m1_m2"


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Serif"
    wide = pd.read_csv(DATA_PATH, low_memory=False)
    revenue_cols = [f"revenue_{year}" for year in YEARS]

    fig = plt.figure(figsize=(12.2, 8.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1.05], hspace=0.42, wspace=0.18)

    for row_idx, case in enumerate(CASES):
        sector = wide[wide["okved_old"].astype(str).str.startswith(case["sector"])].copy()
        company = wide[wide["company_name"] == case["company_name"]].iloc[0]
        sector_median = sector[revenue_cols].median().astype(float)
        company_series = company[revenue_cols].astype(float)
        sector_index = sector_median / float(sector_median.iloc[0]) * 100.0
        company_index = company_series / float(company_series.iloc[0]) * 100.0

        ax_line = fig.add_subplot(gs[row_idx, 0])
        ax_line.plot(YEARS, sector_index.values, marker="o", linewidth=2.2, color=SECTOR_COLORS[case["sector"]], label=f"Медиана сектора {case['sector']}")
        ax_line.plot(YEARS, company_index.values, marker="s", linewidth=1.9, color="#444444", label=case["company_label"])
        for shock_year in [2020, 2022]:
            ax_line.axvline(shock_year, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_line.set_title(case["sector_label"], loc="left", fontsize=11, fontweight="bold")
        ax_line.set_ylabel("Индекс, 2014 = 100")
        ax_line.set_xlabel("Год")
        ax_line.grid(alpha=0.25)
        if row_idx == 0:
            ax_line.legend(frameon=False, fontsize=8.8, ncol=2, loc="upper left")

        ax_diag = fig.add_subplot(gs[row_idx, 1])
        ax_diag.set_xlim(2013.7, 2022.3)
        ax_diag.set_ylim(0, 3)
        ax_diag.set_xticks(YEARS)
        ax_diag.set_yticks([2.5, 1.5, 0.5])
        ax_diag.set_yticklabels(["2014–2020", "2015–2021", "2016–2022"], fontsize=8.7)
        ax_diag.grid(axis="x", alpha=0.18)
        if row_idx == 0:
            ax_diag.set_title("Сдвиг окна W = 7 и допустимый контур чтения", fontsize=10.5, pad=10)
        ax_diag.set_xlabel("Год")
        values = company_series.to_numpy(dtype=float)
        windows = [(2014, 2020), (2015, 2021), (2016, 2022)]
        full_regime = interval_regime(values)
        reg = make_reg_df(values)
        cond, corr = cond_and_corr(reg)
        interp = interpretability(full_regime, float(np.var(growth_rate(values), ddof=0)), cond, corr)
        full_tool = tool_label(full_regime, interp)
        ax_diag.text(
            0.00,
            0.98,
            f"Полный интервал: {REGIME_LABELS.get(full_regime, full_regime)}; {TOOL_LABELS.get(full_tool, full_tool)}",
            transform=ax_diag.transAxes,
            fontsize=8.6,
            va="top",
        )

        for idx, (start, end) in enumerate(windows):
            s_idx = YEARS.index(start)
            block = values[s_idx:s_idx + 7]
            regime = interval_regime(block)
            reg = make_reg_df(block)
            cond, corr = cond_and_corr(reg)
            interp = interpretability(regime, float(np.var(growth_rate(block), ddof=0)), cond, corr)
            tool = tool_label(regime, interp)
            y = 2 - idx
            rect = patches.FancyBboxPatch(
                (start - 0.45, y + 0.12),
                (end - start) + 0.9,
                0.80,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                linewidth=0.8,
                edgecolor="#666666",
                facecolor=TOOL_COLORS.get(tool, "#cccccc"),
                alpha=0.95,
            )
            ax_diag.add_patch(rect)
            ax_diag.text(
                start - 0.20,
                y + 0.50,
                f"{REGIME_LABELS.get(regime, regime)}; {TOOL_LABELS.get(tool, tool)}",
                va="center",
                ha="left",
                fontsize=8.7,
                color="white",
                fontweight="bold",
            )
        for shock_year in [2020, 2022]:
            ax_diag.axvline(shock_year, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)

    fig.suptitle("Три фирменных якоря внутри отраслевых мини-кейсов", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
