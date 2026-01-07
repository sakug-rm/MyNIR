"""Lightweight report generation utilities (HTML-first)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping


def save_html_report(title: str, sections: Iterable[Mapping[str, str]], output_path: str | Path) -> Path:
    """Render a simple static HTML report.

    Parameters
    ----------
    title : str
        Report title.
    sections : iterable of mappings
        Each mapping should contain "title" and "body" (HTML snippets).
    output_path : str or Path
        Where to write the report.
    """
    parts: List[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;} h1,h2{color:#222;} .section{margin-bottom:32px;} .section pre{background:#f7f7f7;padding:12px;}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
    ]
    for section in sections:
        sec_title = section.get("title", "")
        body = section.get("body", "")
        parts.append(f"<div class='section'><h2>{sec_title}</h2>{body}</div>")
    parts.append("</body></html>")

    output_path = Path(output_path)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def save_pdf_placeholder(output_path: str | Path) -> None:
    """Placeholder for PDF export (extend with weasyprint/reportlab later)."""
    output_path = Path(output_path)
    output_path.write_text(
        "PDF export is not implemented yet. Use HTML report or wire a renderer (weasyprint/pdfkit).",
        encoding="utf-8",
    )
