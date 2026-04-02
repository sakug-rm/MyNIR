from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nonlinear_lab.plotting import save_figure

DEFAULT_CASE_CODES = (
    "ПРОМ",
    "06",
    "06.1",
    "10",
    "35.1+35.3",
    "21",
    "26",
    "38",
)

SHOCK_DATES = {
    "2020_shock": pd.Timestamp("2020-04-01"),
    "2022_shock": pd.Timestamp("2022-03-01"),
}

REGIME_COLORS = {
    "growth_no_memory": "#6baed6",
    "memory_like_growth": "#74c476",
    "oscillatory_informative": "#fd8d3c",
    "turbulent_informative": "#ef3b2c",
    "plateau_degenerate": "#9e9ac8",
    "collapse": "#636363",
}

INTERPRETABILITY_COLORS = {
    "interpretable": "#74c476",
    "collinearity_heavy": "#fd8d3c",
    "low_dispersion": "#9e9ac8",
    "plateau_degenerate": "#756bb1",
    "collapse": "#525252",
}

READING_MODE_COLORS = {
    "phase_caution": "#ef3b2c",
    "do_not_read": "#636363",
    "beta_bsum_caution": "#fd8d3c",
    "beta_bsum_then_stepwise": "#fdae6b",
    "memory_structure": "#31a354",
    "current_state": "#3182bd",
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-zА-Яа-я._+-]+", "_", str(value)).strip("_")
    return slug or "case"


def _display_code(value: str) -> str:
    value_str = str(value)
    return value_str.split(" ", 1)[0]


def _normalize_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].replace(".", "").isdigit():
        return text[:-2]
    return text


def _dominant_name(values: pd.Series) -> str:
    usable = values.dropna()
    if usable.empty:
        return ""
    return str(usable.value_counts().idxmax())


def _resolve_case_codes(case_codes: tuple[str, ...] | list[str], available_codes: pd.Series) -> tuple[str, ...]:
    available = [_normalize_code(code) for code in pd.Series(available_codes).dropna().astype(str).drop_duplicates().tolist()]
    resolved: list[str] = []
    for requested in case_codes:
        requested_str = _normalize_code(requested)
        if requested_str in available:
            resolved.append(requested_str)
            continue
        matched = [code for code in available if code.startswith(requested_str)]
        if len(matched) == 1:
            resolved.append(matched[0])
    return tuple(dict.fromkeys(resolved))


def _window_center_dates(dates: pd.Series, starts: pd.Series, window: int) -> pd.Series:
    indices = starts.astype(int) + max(window // 2, 0)
    indices = indices.clip(lower=0, upper=len(dates) - 1)
    lookup = dates.reset_index(drop=True)
    return indices.map(lambda idx: lookup.iloc[int(idx)])


def _find_nearest_index(dates: pd.Series, target: pd.Timestamp) -> int:
    distances = (dates - target).abs()
    return int(distances.idxmin())


def _mode_by_shock(group: pd.DataFrame, dates: pd.Series, *, mode_col: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for label, target in SHOCK_DATES.items():
        target_idx = _find_nearest_index(dates.reset_index(drop=True), target)
        hit = group[(group["start"] <= target_idx) & (group["end"] > target_idx)]
        out[label] = _dominant_name(hit[mode_col]) if not hit.empty else ""
    return out


def _shock_value_summary(series: pd.Series, dates: pd.Series) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    aligned_dates = dates.reset_index(drop=True)
    aligned_values = series.reset_index(drop=True).astype(float)
    for label, target in SHOCK_DATES.items():
        idx = _find_nearest_index(aligned_dates, target)
        prev_idx = max(idx - 6, 0)
        next_idx = min(idx + 6, len(aligned_values) - 1)
        prev_level = float(aligned_values.iloc[prev_idx])
        level = float(aligned_values.iloc[idx])
        next_level = float(aligned_values.iloc[next_idx])
        six_month_change_before = (level / prev_level - 1.0) if abs(prev_level) > 1e-12 else np.nan
        six_month_change_after = (next_level / level - 1.0) if abs(level) > 1e-12 else np.nan
        rows.append(
            {
                "shock_label": label,
                "shock_date": aligned_dates.iloc[idx].strftime("%Y-%m-%d"),
                "shock_level": level,
                "six_month_change_before": float(six_month_change_before),
                "six_month_change_after": float(six_month_change_after),
            }
        )
    return rows


def _build_case_summary(
    case_codes: tuple[str, ...],
    hierarchy: pd.DataFrame,
    series_panel: pd.DataFrame,
    routing_windows: pd.DataFrame,
    structural_windows: pd.DataFrame,
    *,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hierarchy_norm = hierarchy.copy()
    hierarchy_norm["series_code"] = hierarchy_norm["series_code"].map(_normalize_code)
    hierarchy_norm["parent_code"] = hierarchy_norm["parent_code"].map(_normalize_code)
    hierarchy_map = hierarchy_norm.set_index("series_code")
    rows: list[dict[str, object]] = []
    shock_rows: list[dict[str, object]] = []
    hierarchy_rows: list[dict[str, object]] = []

    for code in case_codes:
        case_series = series_panel[(series_panel["series_code"] == code) & (series_panel["variant"] == variant)].copy()
        if case_series.empty:
            continue
        case_series = case_series.sort_values("date")
        case_routing = routing_windows[routing_windows["series_code"] == code].copy()
        case_structural = structural_windows[structural_windows["series_code"] == code].copy()
        if case_routing.empty or case_structural.empty:
            continue

        routing_w24 = case_routing[case_routing["window"] == 24].copy()
        structural_w24 = case_structural[case_structural["window"] == 24].copy()
        series_name = str(case_series["series_name"].iloc[0])
        parent_code = ""
        parent_name = ""
        if code in hierarchy_map.index:
            parent_code = str(hierarchy_map.loc[code, "parent_code"])
            parent_name = str(hierarchy_map.loc[code, "parent_name"])

        case_row = {
            "series_code": code,
            "series_name": series_name,
            "parent_code": parent_code,
            "parent_name": parent_name,
            "interpretable_share": float((case_routing["interpretability_label"] == "interpretable").mean()),
            "plateau_share": float((case_routing["interpretability_label"] == "plateau_degenerate").mean()),
            "collinearity_share": float((case_routing["interpretability_label"] == "collinearity_heavy").mean()),
            "low_dispersion_share": float((case_routing["interpretability_label"] == "low_dispersion").mean()),
            "dominant_regime": _dominant_name(case_routing["regime_label"]),
            "dominant_tool": _dominant_name(case_routing["preferred_tool"]),
            "dominant_mode": _dominant_name(case_structural["reading_mode"]),
            "dominant_top_beta": _dominant_name(case_structural["top_beta_predictor"]),
            "dominant_top_b": _dominant_name(case_structural["top_b_predictor"]),
            "mean_r2": float(case_structural["R2_enter"].mean()),
            "mean_b_sum": float(case_structural["b_sum"].mean()),
            "mean_selected_count": float(case_structural["selected_count"].mean()),
        }
        case_row.update({f"shock_{k}_regime": v for k, v in _mode_by_shock(routing_w24, case_series["date"], mode_col="regime_label").items()})
        case_row.update(
            {f"shock_{k}_mode": v for k, v in _mode_by_shock(structural_w24, case_series["date"], mode_col="reading_mode").items()}
        )
        rows.append(case_row)

        for shock in _shock_value_summary(case_series["index_value"], case_series["date"]):
            routing_hit = routing_w24[
                (routing_w24["start"] <= _find_nearest_index(case_series["date"].reset_index(drop=True), pd.Timestamp(shock["shock_date"])))
                & (routing_w24["end"] > _find_nearest_index(case_series["date"].reset_index(drop=True), pd.Timestamp(shock["shock_date"])))
            ]
            structural_hit = structural_w24[
                (structural_w24["start"] <= _find_nearest_index(case_series["date"].reset_index(drop=True), pd.Timestamp(shock["shock_date"])))
                & (structural_w24["end"] > _find_nearest_index(case_series["date"].reset_index(drop=True), pd.Timestamp(shock["shock_date"])))
            ]
            shock_rows.append(
                {
                    "series_code": code,
                    "series_name": series_name,
                    **shock,
                    "shock_regime_w24": _dominant_name(routing_hit["regime_label"]) if not routing_hit.empty else "",
                    "shock_interpretability_w24": _dominant_name(routing_hit["interpretability_label"]) if not routing_hit.empty else "",
                    "shock_mode_w24": _dominant_name(structural_hit["reading_mode"]) if not structural_hit.empty else "",
                }
            )

        if parent_code:
            parent_routing = routing_windows[routing_windows["series_code"] == parent_code]
            parent_structural = structural_windows[structural_windows["series_code"] == parent_code]
            if not parent_routing.empty and not parent_structural.empty:
                hierarchy_rows.append(
                    {
                        "series_code": code,
                        "series_name": series_name,
                        "parent_code": parent_code,
                        "parent_name": parent_name,
                        "child_interpretable_share": float((case_routing["interpretability_label"] == "interpretable").mean()),
                        "parent_interpretable_share": float((parent_routing["interpretability_label"] == "interpretable").mean()),
                        "child_plateau_share": float((case_routing["interpretability_label"] == "plateau_degenerate").mean()),
                        "parent_plateau_share": float((parent_routing["interpretability_label"] == "plateau_degenerate").mean()),
                        "child_dominant_mode": _dominant_name(case_structural["reading_mode"]),
                        "parent_dominant_mode": _dominant_name(parent_structural["reading_mode"]),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(shock_rows), pd.DataFrame(hierarchy_rows)


def run_ipp_case_cards(
    ipp_long: pd.DataFrame,
    hierarchy: pd.DataFrame,
    routing_windows: pd.DataFrame,
    structural_windows: pd.DataFrame,
    *,
    case_codes: tuple[str, ...] | list[str] = DEFAULT_CASE_CODES,
    variant: str = "adj_unsmoothed",
) -> dict[str, pd.DataFrame]:
    resolved_codes = _resolve_case_codes(case_codes, ipp_long["series_code"])
    series_panel = ipp_long[ipp_long["series_code"].isin(resolved_codes)].copy()
    series_panel["date"] = pd.to_datetime(series_panel["date"])

    routing = routing_windows[
        (routing_windows["series_code"].isin(resolved_codes)) & (routing_windows["variant"] == variant)
    ].copy()
    routing["center_date"] = pd.NaT
    for (code, window), idx in routing.groupby(["series_code", "window"]).groups.items():
        case_dates = (
            series_panel[(series_panel["series_code"] == code) & (series_panel["variant"] == variant)]
            .sort_values("date")["date"]
        )
        routing.loc[idx, "center_date"] = _window_center_dates(case_dates, routing.loc[idx, "start"], int(window)).to_numpy()

    structural = structural_windows[
        (structural_windows["series_code"].isin(resolved_codes)) & (structural_windows["variant"] == variant)
    ].copy()
    structural["center_date"] = pd.NaT
    for (code, window), idx in structural.groupby(["series_code", "window"]).groups.items():
        case_dates = (
            series_panel[(series_panel["series_code"] == code) & (series_panel["variant"] == variant)]
            .sort_values("date")["date"]
        )
        structural.loc[idx, "center_date"] = _window_center_dates(case_dates, structural.loc[idx, "start"], int(window)).to_numpy()

    case_summary, shock_summary, hierarchy_summary = _build_case_summary(
        tuple(resolved_codes),
        hierarchy,
        series_panel,
        routing,
        structural,
        variant=variant,
    )

    return {
        "series_panel": series_panel,
        "routing_windows": routing,
        "structural_windows": structural,
        "case_summary": case_summary.sort_values("series_code").reset_index(drop=True),
        "shock_summary": shock_summary.sort_values(["series_code", "shock_label"]).reset_index(drop=True),
        "hierarchy_summary": (
            hierarchy_summary.sort_values("series_code").reset_index(drop=True)
            if not hierarchy_summary.empty
            else pd.DataFrame(
                columns=[
                    "series_code",
                    "series_name",
                    "parent_code",
                    "parent_name",
                    "child_interpretable_share",
                    "parent_interpretable_share",
                    "child_plateau_share",
                    "parent_plateau_share",
                    "child_dominant_mode",
                    "parent_dominant_mode",
                ]
            )
        ),
    }


def _plot_case_figure(
    series_panel: pd.DataFrame,
    routing_windows: pd.DataFrame,
    structural_windows: pd.DataFrame,
    *,
    series_code: str,
    window: int = 24,
):
    case_series = series_panel[series_panel["series_code"] == series_code].sort_values(["variant", "date"])
    case_routing = routing_windows[(routing_windows["series_code"] == series_code) & (routing_windows["window"] == window)].sort_values("center_date")
    case_structural = structural_windows[(structural_windows["series_code"] == series_code) & (structural_windows["window"] == window)].sort_values("center_date")
    series_name = str(case_series["series_name"].iloc[0])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.7, 0.8])
    ax_series = fig.add_subplot(gs[0, :])
    ax_regime = fig.add_subplot(gs[1, 0])
    ax_mode = fig.add_subplot(gs[1, 1])
    ax_struct = fig.add_subplot(gs[2, 0])
    ax_top = fig.add_subplot(gs[2, 1])

    for variant, subset in case_series.groupby("variant", sort=False):
        ax_series.plot(subset["date"], subset["index_value"], label=variant, linewidth=1.5)
    for label, date in SHOCK_DATES.items():
        ax_series.axvline(date, color="#444444", linestyle="--", linewidth=1.0)
        ax_series.text(date, ax_series.get_ylim()[1], label.replace("_", " "), rotation=90, va="top", ha="right", fontsize=8)
    ax_series.set_title(f"{series_code} — {series_name}")
    ax_series.set_ylabel("Index level")
    ax_series.legend(fontsize=8, ncol=3)

    if not case_routing.empty:
        regime_categories = list(dict.fromkeys(case_routing["regime_label"]))
        regime_map = {name: idx for idx, name in enumerate(regime_categories)}
        colors = [REGIME_COLORS.get(name, "#cccccc") for name in regime_categories]
        values = np.array([[regime_map[name] for name in case_routing["regime_label"]]])
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        ax_regime.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
        ax_regime.set_title(f"Regime Timeline (W={window})")
        tick_idx = np.linspace(0, len(case_routing) - 1, min(6, len(case_routing)), dtype=int)
        ax_regime.set_xticks(tick_idx)
        ax_regime.set_xticklabels(case_routing.iloc[tick_idx]["center_date"].dt.strftime("%Y-%m"), rotation=45, ha="right")
        ax_regime.set_yticks([])

    if not case_structural.empty:
        mode_categories = list(dict.fromkeys(case_structural["reading_mode"]))
        mode_map = {name: idx for idx, name in enumerate(mode_categories)}
        colors = [READING_MODE_COLORS.get(name, "#cccccc") for name in mode_categories]
        values = np.array([[mode_map[name] for name in case_structural["reading_mode"]]])
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        ax_mode.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
        ax_mode.set_title(f"Reading Mode Timeline (W={window})")
        tick_idx = np.linspace(0, len(case_structural) - 1, min(6, len(case_structural)), dtype=int)
        ax_mode.set_xticks(tick_idx)
        ax_mode.set_xticklabels(case_structural.iloc[tick_idx]["center_date"].dt.strftime("%Y-%m"), rotation=45, ha="right")
        ax_mode.set_yticks([])

        ax_struct.plot(case_structural["center_date"], case_structural["b_sum"], color="#1f77b4", label="B_sum")
        ax_struct.set_ylabel("B_sum", color="#1f77b4")
        ax_struct.tick_params(axis="y", labelcolor="#1f77b4")
        ax_struct2 = ax_struct.twinx()
        ax_struct2.plot(case_structural["center_date"], case_structural["R2_enter"], color="#d62728", alpha=0.8, label="R2")
        ax_struct2.set_ylabel("R2 enter", color="#d62728")
        ax_struct2.tick_params(axis="y", labelcolor="#d62728")
        ax_struct.set_title(f"Structural Metrics (W={window})")

        top_counts = (
            case_structural["top_beta_predictor"].replace("", np.nan).dropna().value_counts().head(6)
        )
        ax_top.bar(top_counts.index.astype(str), top_counts.values, color="#6baed6")
        ax_top.set_title("Top beta predictors")
        ax_top.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig


def _plot_case_overview(case_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(case_summary["interpretable_share"], case_summary["plateau_share"], s=60, color="#3182bd")
    for _, row in case_summary.iterrows():
        ax.text(row["interpretable_share"] + 0.005, row["plateau_share"] + 0.005, str(row["series_code"]), fontsize=8)
    ax.set_xlabel("Interpretable Share")
    ax.set_ylabel("Plateau Share")
    ax.set_title("Case Cards: Interpretable vs Plateau Shares")
    fig.tight_layout()
    return fig


def _build_case_cards_markdown(case_summary: pd.DataFrame, output_dir: Path) -> str:
    lines = ["# Кейсовые отраслевые карточки", ""]
    lines.append("| Код | Серия | Интерпрет. | Plateau | Дом. режим | Дом. контур | Дом. beta |")
    lines.append("|---|---|---:|---:|---|---|---|")
    for _, row in case_summary.iterrows():
        display_code = _display_code(str(row["series_code"]))
        lines.append(
            f"| {display_code} | {row['series_name']} | {row['interpretable_share']:.3f} | {row['plateau_share']:.3f} | {row['dominant_regime']} | {row['dominant_mode']} | {row['dominant_top_beta']} |"
        )
    lines.append("")
    for _, row in case_summary.iterrows():
        slug = _slugify(str(row["series_code"]))
        display_code = _display_code(str(row["series_code"]))
        lines.append(f"## {display_code} — {row['series_name']}")
        lines.append("")
        parent_prefix = ""
        if pd.notna(row["parent_code"]) and str(row["parent_code"]).lower() != "nan":
            parent_prefix = f"Родитель: `{row['parent_code']}` {row['parent_name']}; "
        lines.append(
            f"{parent_prefix}интерпретируемая доля `{row['interpretable_share']:.3f}`, plateau `{row['plateau_share']:.3f}`, доминирующий контур `{row['dominant_mode']}`."
        )
        lines.append("")
        lines.append(f"![{display_code}](./card_{slug}.png)")
        lines.append("")
    return "\n".join(lines) + "\n"


def save_ipp_case_card_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)

    case_summary = results["case_summary"]
    if not case_summary.empty:
        save_figure(_plot_case_overview(case_summary), out / "case_overview.png")
        for _, row in case_summary.iterrows():
            slug = _slugify(str(row["series_code"]))
            fig = _plot_case_figure(
                results["series_panel"],
                results["routing_windows"],
                results["structural_windows"],
                series_code=str(row["series_code"]),
            )
            save_figure(fig, out / f"card_{slug}.png")

    markdown = _build_case_cards_markdown(case_summary, out)
    (out / "case_cards.md").write_text(markdown, encoding="utf-8")
    summary = {
        "case_count": int(len(case_summary)),
        "case_codes": [str(code) for code in case_summary["series_code"].tolist()],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
