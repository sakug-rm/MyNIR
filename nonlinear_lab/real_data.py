from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


IPP_VARIANTS = ("adj_smoothed", "adj_unsmoothed", "raw")
REVENUE_STATIC_COLUMNS = [
    "rank",
    "company_name",
    "address",
    "inn",
    "region",
    "okved_old",
    "industry_text",
]


def _coerce_dates(columns: list[Any]) -> list[pd.Timestamp]:
    parsed = pd.to_datetime(columns, errors="coerce")
    if pd.isna(parsed).any():
        raise ValueError("IPP date columns could not be parsed as timestamps.")
    return [pd.Timestamp(value) for value in parsed]


def parse_ipp_label(label: str) -> tuple[str, str]:
    text = str(label).strip()
    if text == "Промышленность - всего":
        return "ПРОМ", text

    match = re.match(r"^([A-ZА-ЯЁ]|\d+(?:\.\d+)?)\s+(.+)$", text)
    if match:
        return match.group(1), match.group(2).strip()
    return text, text


def _ipp_series_prefix_length(labels: pd.Series) -> int:
    clean = labels.fillna("").astype(str).str.strip().tolist()
    usable_len = 0
    for idx in range(0, len(clean), 3):
        chunk = clean[idx : idx + 3]
        if len(chunk) < 3:
            break
        if any(not value for value in chunk):
            break
        if chunk[0] != chunk[1] or chunk[1] != chunk[2]:
            break
        usable_len = idx + 3
    return usable_len


def reshape_ipp_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["date", "series_code", "series_name", "series_label", "variant", "index_value"]
        )

    label_col = df.columns[0]
    usable_len = _ipp_series_prefix_length(df[label_col])
    if usable_len == 0:
        raise ValueError("Could not detect triplet-structured IPP rows.")

    core = df.iloc[:usable_len].copy()
    dates = _coerce_dates(list(core.columns[1:]))
    rows: list[dict[str, Any]] = []

    for idx in range(0, len(core), 3):
        block = core.iloc[idx : idx + 3]
        label = str(block.iloc[0, 0]).strip()
        series_code, series_name = parse_ipp_label(label)
        for variant, (_, row) in zip(IPP_VARIANTS, block.iterrows(), strict=True):
            values = pd.to_numeric(row.iloc[1:], errors="coerce").to_numpy(dtype=float)
            for date, value in zip(dates, values, strict=True):
                rows.append(
                    {
                        "date": date,
                        "series_code": series_code,
                        "series_name": series_name,
                        "series_label": label,
                        "variant": variant,
                        "index_value": value,
                    }
                )

    return pd.DataFrame(rows).dropna(subset=["index_value"]).reset_index(drop=True)


def build_ipp_hierarchy(series_df: pd.DataFrame) -> pd.DataFrame:
    unique = series_df[["series_code", "series_name"]].drop_duplicates().reset_index(drop=True)
    code_to_name = dict(zip(unique["series_code"], unique["series_name"]))
    codes = unique["series_code"].tolist()

    current_section: str | None = None
    section_codes = {"B", "C", "D", "E"}
    section_parent: dict[str, str] = {}
    for code in codes:
        if code == "ПРОМ":
            current_section = None
            continue
        if code in section_codes:
            current_section = code
            continue
        if "." not in str(code) and re.match(r"^\d+$", str(code)) and current_section is not None:
            section_parent[code] = current_section

    rows: list[dict[str, Any]] = []
    for code in codes:
        parent_code: str | None = None
        if code == "ПРОМ":
            parent_code = None
        elif code in section_codes:
            parent_code = "ПРОМ" if "ПРОМ" in code_to_name else None
        elif "." in str(code):
            prefix = str(code).rsplit(".", 1)[0]
            parent_code = prefix if prefix in code_to_name else None
        else:
            parent_code = section_parent.get(code)

        rows.append(
            {
                "series_code": code,
                "series_name": code_to_name[code],
                "parent_code": parent_code,
                "parent_name": code_to_name.get(parent_code),
            }
        )

    return pd.DataFrame(rows)


def load_ipp_long(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    workbook = pd.read_excel(path, sheet_name="OKVED")
    long_df = reshape_ipp_wide_to_long(workbook)
    hierarchy = build_ipp_hierarchy(long_df[["series_code", "series_name"]])
    return long_df, hierarchy


def normalize_revenue_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    header_idx = None
    for idx in range(min(len(raw), 10)):
        row_values = raw.iloc[idx].astype(str).tolist()
        if "Название" in row_values and "ИНН" in row_values and "ОКВЭД" in row_values:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Failed to locate revenue header row.")

    header_row = raw.iloc[header_idx].tolist()
    year_by_position: dict[int, int] = {}
    for idx, value in enumerate(header_row):
        match = re.search(r"(19\d{2}|20\d{2})", str(value))
        if match:
            year_by_position[idx] = int(match.group(1))

    next_row_values = raw.iloc[header_idx + 1].astype(str).tolist() if header_idx + 1 < len(raw) else []
    has_subheader = "Код" in next_row_values or "Название" in next_row_values
    data_start = header_idx + 2 if has_subheader else header_idx + 1
    data = raw.iloc[data_start:].reset_index(drop=True).copy()
    renamed = []
    for idx in range(len(data.columns)):
        if idx < len(REVENUE_STATIC_COLUMNS):
            renamed.append(REVENUE_STATIC_COLUMNS[idx])
        elif idx in year_by_position:
            renamed.append(f"revenue_{year_by_position[idx]}")
        else:
            renamed.append(f"extra_{idx}")
    data.columns = renamed

    keep = REVENUE_STATIC_COLUMNS + [name for name in data.columns if name.startswith("revenue_")]
    wide = data[keep].copy()
    wide["rank"] = pd.to_numeric(wide["rank"], errors="coerce")
    for column in [name for name in wide.columns if name.startswith("revenue_")]:
        wide[column] = pd.to_numeric(wide[column], errors="coerce")

    normalized_name = (
        wide["company_name"]
        .astype(str)
        .str.lower()
        .str.replace('"', "", regex=False)
        .str.replace("«", "", regex=False)
        .str.replace("»", "", regex=False)
        .str.strip()
    )
    summary_mask = normalized_name.isin({"медиана", "median", "итог", "итого"})
    wide["is_summary_row"] = summary_mask | wide["rank"].isna()
    return wide.reset_index(drop=True)


def reshape_revenue_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    revenue_cols = [column for column in df.columns if column.startswith("revenue_")]
    long_df = df.melt(
        id_vars=REVENUE_STATIC_COLUMNS + ["is_summary_row"],
        value_vars=revenue_cols,
        var_name="revenue_year",
        value_name="revenue",
    )
    long_df["year"] = long_df["revenue_year"].str.replace("revenue_", "", regex=False).astype(int)
    long_df["okved_missing"] = long_df["okved_old"].isna() | (long_df["okved_old"].astype(str).str.strip() == "")
    return long_df.drop(columns=["revenue_year"]).reset_index(drop=True)


def load_revenue_corpus(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path, header=None)
    wide = normalize_revenue_sheet(raw)
    long_df = reshape_revenue_wide_to_long(wide)
    return wide, long_df
