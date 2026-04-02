from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_MD = ROOT / "diploma.md"
OUTPUT_DIR = ROOT / "tex"
CHAPTERS_DIR = OUTPUT_DIR / "chapters"


@dataclass
class Node:
    level: int
    title: str
    children: list["Node"] = field(default_factory=list)
    body: list[str] = field(default_factory=list)


def slugify(text: str) -> str:
    text = re.sub(r"^[0-9. ]+", "", text).strip()
    text = text.lower()
    translit = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
    text = "".join(translit.get(ch, ch) for ch in text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "section"


def strip_numbering(title: str) -> str:
    title = re.sub(r"^ГЛАВА\s+\d+\.\s*", "", title).strip()
    title = re.sub(r"^ПРИЛОЖЕНИЕ\s+[А-ЯA-Z]\.?\s*", "", title).strip()
    title = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", title).strip()
    return title


def parse_markdown(text: str) -> Node:
    root = Node(level=0, title="root")
    stack = [root]
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            node = Node(level=level, title=title)
            while stack and stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].body.append(line)
    return root


def convert_markdown_fragment(markdown: str) -> str:
    fragment = markdown.strip()
    if not fragment:
        return ""
    result = subprocess.run(
        [
            "pandoc",
            "--from=markdown",
            "--to=latex",
            "--wrap=none",
        ],
        input=fragment,
        text=True,
        capture_output=True,
        check=True,
    )
    latex = result.stdout
    latex = latex.replace("\\begin{figure}", "\\begin{figure}[htbp]")
    latex = latex.replace("\\begin{table}", "\\begin{table}[htbp]")
    return latex.strip() + "\n\n"


def render_node(node: Node, level_map: dict[int, str]) -> str:
    parts: list[str] = []
    if node.level in level_map:
        command = level_map[node.level]
        title = strip_numbering(node.title)
        label_prefix = "app" if node.level == 2 and node.title.startswith("Приложение") else "sec"
        parts.append(f"\\{command}{{{title}}}\n\\label{{{label_prefix}:{slugify(title)}}}\n\n")
    body = "\n".join(node.body).strip()
    if body:
        parts.append(convert_markdown_fragment(body))
    for child in node.children:
        parts.append(render_node(child, level_map))
    return "".join(parts)


def build_chapter_file(node: Node, filename: str, chapter_number: int | None = None) -> None:
    title = strip_numbering(node.title)
    label = f"chap:{slugify(title)}"
    parts = []
    if chapter_number is not None:
        parts.append(f"\\setcounter{{chapter}}{{{chapter_number - 1}}}\n")
    parts.append(f"\\chapter{{{title}}}\n\\label{{{label}}}\n\n")
    body = "\n".join(node.body).strip()
    if body:
        parts.append(convert_markdown_fragment(body))
    for child in node.children:
        parts.append(render_node(child, {2: "section", 3: "subsection", 4: "subsubsection"}))
    (CHAPTERS_DIR / filename).write_text("".join(parts))


def build_appendix_file(node: Node, filename: str) -> None:
    title = strip_numbering(node.title)
    label = f"app:{slugify(title)}"
    parts = [f"\\chapter{{{title}}}\n\\label{{{label}}}\n\n"]
    body = "\n".join(node.body).strip()
    if body:
        parts.append(convert_markdown_fragment(body))
    for child in node.children:
        parts.append(render_node(child, {3: "section", 4: "subsection"}))
    (CHAPTERS_DIR / filename).write_text("".join(parts))


def write_main_template() -> None:
    thesis = textwrap.dedent(
        r"""
        \documentclass[14pt,a4paper]{extreport}

        \usepackage{fontspec}
        \setmainfont{Times New Roman}
        \setsansfont{Times New Roman}
        \setmonofont{Courier New}
        \newfontfamily\cyrillicfonttt{Times New Roman}

        \usepackage{polyglossia}
        \setdefaultlanguage{russian}
        \setotherlanguage{english}

        \usepackage[left=30mm,right=15mm,top=20mm,bottom=20mm]{geometry}
        \usepackage{setspace}
        \onehalfspacing
        \usepackage{indentfirst}
        \setlength{\parindent}{1.25cm}
        \setlength{\parskip}{0pt}
        \usepackage{graphicx}
        \graphicspath{{../}}
        \usepackage{float}
        \usepackage{calc}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{array}
        \usepackage{amsmath,amssymb}
        \usepackage{caption}
        \captionsetup{justification=centering,labelsep=endash}
        \usepackage[hidelinks]{hyperref}
        \usepackage{enumitem}
        \setlist{nosep}
        \usepackage{titlesec}
        \usepackage{placeins}
        \usepackage{etoolbox}
        \AtBeginEnvironment{longtable}{\small}
        \makeatletter
        \def\fps@figure{htbp}
        \def\fps@table{htbp}
        \makeatother
        \newcounter{none}
        \providecommand{\tightlist}{%
          \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
        \providecommand{\pandocbounded}[1]{#1}

        \titleformat{\chapter}[block]{\bfseries\filcenter}{\thechapter}{1em}{}
        \titleformat{\section}[block]{\bfseries}{\thesection}{1em}{}
        \titleformat{\subsection}[block]{\bfseries}{\thesubsection}{1em}{}

        \begin{document}

        \pagenumbering{arabic}
        \setcounter{page}{1}

        \tableofcontents
        \clearpage

        \input{chapters/02_modeling}
        \input{chapters/03_real_data}

        \appendix
        \input{chapters/appendix_a}
        \input{chapters/appendix_b}
        \input{chapters/appendix_v}

        \end{document}
        """
    ).strip() + "\n"
    (OUTPUT_DIR / "thesis.tex").write_text(thesis)
    (OUTPUT_DIR / "references.bib").write_text("% Заполнить библиографию при переносе ссылок на литературу.\n")


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    root = parse_markdown(SOURCE_MD.read_text())
    top = {node.title: node for node in root.children}

    build_chapter_file(top["ГЛАВА 2. ЧИСЛЕННОЕ МОДЕЛИРОВАНИЕ И ИССЛЕДОВАНИЕ ВОЗМОЖНОСТЕЙ РЕГРЕССИОННОГО АНАЛИЗА ДЛЯ ИДЕНТИФИКАЦИИ НЕЛИНЕЙНЫХ ПРОЦЕССОВ"], "02_modeling.tex", 2)
    build_chapter_file(top["ГЛАВА 3. ПЕРЕНОС РЕЖИМНО-ЗАВИСИМОГО АППАРАТА НА РЕАЛЬНЫЕ ДАННЫЕ"], "03_real_data.tex", None)

    appendices = top["ПРИЛОЖЕНИЯ"].children
    appendix_map = {
        "Приложение А. Дополнительные synthetic-эксперименты и проверки устойчивости": "appendix_a.tex",
        "Приложение Б. Расширенная линия промышленных индексов": "appendix_b.tex",
        "Приложение В. Дополнительные таблицы по линии выручки": "appendix_v.tex",
    }
    for appendix in appendices:
        build_appendix_file(appendix, appendix_map[appendix.title])

    write_main_template()


if __name__ == "__main__":
    main()
