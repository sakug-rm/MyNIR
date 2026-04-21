from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("real_data/processed/revenue_wide.csv")
OUTPUT_PATH = Path("outputs/real_data/revenue_b2/company_case_tesmo.png")
COMPANY_NAME = '"ТЭСМО", АКЦИОНЕРНОЕ ОБЩЕСТВО'
SECTOR_PREFIX = "29"
SUBSECTOR_CODE = "29.53"


def main() -> None:
    wide = pd.read_csv(DATA_PATH, low_memory=False)
    year_range = list(range(2014, 2023))
    revenue_cols = [f"revenue_{year}" for year in year_range]

    sector = wide[wide["okved_old"].astype(str).str.startswith(SECTOR_PREFIX)].copy()
    company = wide[wide["company_name"] == COMPANY_NAME].copy()
    if company.empty:
        raise SystemExit(f"Company not found: {COMPANY_NAME}")

    company = company.iloc[0]
    sector_median = sector[revenue_cols].median().astype(float)
    company_series = company[revenue_cols].astype(float)

    sector_index = sector_median / float(sector_median.iloc[0]) * 100.0
    company_index = company_series / float(company_series.iloc[0]) * 100.0

    sector_growth = sector_median.pct_change() * 100.0
    company_growth = company_series.pct_change() * 100.0

    plt.rcParams["font.family"] = "DejaVu Serif"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    axes[0].plot(year_range, sector_index.values, marker="o", linewidth=2.2, color="#1b6ca8", label="Медиана сектора 29")
    axes[0].plot(year_range, company_index.values, marker="s", linewidth=2.0, color="#c55a11", label="АО «ТЭСМО»")
    axes[0].set_title("Индекс выручки, 2014 = 100")
    axes[0].set_xlabel("Год")
    axes[0].set_ylabel("Индекс")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    growth_years = year_range[1:]
    axes[1].axhline(0.0, color="#666666", linewidth=1.0, alpha=0.8)
    axes[1].plot(growth_years, sector_growth.iloc[1:].values, marker="o", linewidth=2.2, color="#1b6ca8", label="Медиана сектора 29")
    axes[1].plot(growth_years, company_growth.iloc[1:].values, marker="s", linewidth=2.0, color="#c55a11", label="АО «ТЭСМО»")
    axes[1].set_title("Годовой темп прироста выручки, %")
    axes[1].set_xlabel("Год")
    axes[1].set_ylabel("Темп, %")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Иллюстративный фирменный кейс внутри сектора 29.53\n"
        "Производство машин и оборудования для изготовления пищевых продуктов, включая напитки, и табачных изделий",
        fontsize=11,
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT_PATH}")
    print(f"Selected company: {company['company_name']} | INN {company['inn']} | OKVED {SUBSECTOR_CODE}")


if __name__ == "__main__":
    main()
