# LaTeX Version Of The Thesis

Main source file: `thesis.tex`

Build from the `tex/` directory:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build thesis.tex
```

Resulting PDF:

- `build/thesis.pdf`

The LaTeX sources are generated from the current `diploma.md` by:

```bash
python3 scripts/build_thesis_tex.py
```
