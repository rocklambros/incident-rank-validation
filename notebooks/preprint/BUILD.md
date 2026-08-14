# Preprint build — environment & commands

The preprint is built **notebook-first**: update the notebook → execute → export markdown → pandoc + custom LaTeX template → PDF. This file records the exact, reproducible build environment (hard-won during Task 4/4.5).

## Kernel (IMPORTANT)
The notebook-execution stack (`nbconvert`, `ipykernel`) is a declared extra — install it (with the chart deps) before registering the kernel:

```bash
uv sync --extra narrative --extra preprint   # or: uv sync --all-extras
```

Then register the **`preprint-build`** kernel from the project `uv` env:

```bash
uv run python -m ipykernel install --user --name preprint-build --display-name "preprint-build"
```

**Do NOT use the anaconda `python3` kernel** — its kernelspec is hand-crippled: it stubs out `kaleido`/`tensorflow`/`jax` (`sys.modules[m]=None`) and rewrites `sys.argv`, which (a) breaks under `nbconvert --execute` (`--IPKernelApp.connection_file: expected one argument` → "Kernel died"), and (b) would break the 3 plotly charts (they need kaleido for PNG export). The `preprint-build` kernel (uv env) has the full stack **plus a working kaleido and the importable `engine` package**.

## LaTeX packages
`arxiv-template.latex` requires `fancyhdr`, `wrapfig`, `float`, and `newunicodechar`, none guaranteed in a minimal local TeX tree:

```bash
tlmgr install fancyhdr wrapfig float newunicodechar
```

- `wrapfig`/`float` — figure placement (the `figure-layout.lua` filter emits `wrapfigure` and `figure[H/htbp]`).
- `newunicodechar` — maps the Greek/math glyphs used in prose and captions (κ λ ρ ≈ × −) to math mode; the Latin Modern text font lacks them and xelatex would otherwise drop them silently.

(arXiv's TeXLive has all four, so LaTeX-source submission compiles there.)

## Build command
```bash
python tools/build_preprint.py \
  --notebook notebooks/2026_top_10_llm_update_what_the_data_says.ipynb \
  --out-dir notebooks/preprint \
  --output-name Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026
```
This executes a COPY of the notebook (never mutates the committed `.ipynb`), exports markdown with outputs stripped, prepends the YAML front-matter, and compiles via `pandoc --template=notebooks/preprint/arxiv-template.latex --pdf-engine=xelatex`. Emits `<output-name>.{md,pdf,tex}`.

## arXiv submission

Submit the generated LaTeX **source**, not PDF-only, so arXiv can recompile.

Upload a single `.tar.gz`. arXiv's uploader takes one archive and unpacks it, which preserves the `figures/` directory the `\includegraphics` paths depend on. Uploading files one at a time would flatten them and break every figure reference.

The archive holds exactly two things, with no wrapper directory:

1. `<output-name>.tex` at the archive root
2. `figures/` with the 19 PNGs the `.tex` references

Do **not** include `<output-name>.pdf` (arXiv rejects `foo.pdf` shipped alongside `foo.tex`), `<output-name>.md`, `arxiv-template.latex`, `document-metadata.latex`, `figure-layout.lua`, `front_matter.md`, `_stub.md`, or `figures/rank_change_2025_2026.png` (retired slopegraph, unreferenced). No `.bbl`/`.bib` is needed; the document has no bibliography environment.

Build the archive from `notebooks/preprint/`:

```bash
STEM=<output-name>
STAGE=$(mktemp -d) && mkdir -p "$STAGE/figures"
cp "$STEM.tex" "$STAGE/"
grep -o 'includegraphics\[[^]]*\]{figures/[a-z0-9_]*\.png}' "$STEM.tex" \
  | sed 's|.*{figures/||;s/}//' | sort -u \
  | while read -r f; do cp "figures/$f" "$STAGE/figures/$f"; done
# COPYFILE_DISABLE stops macOS tar from writing ._ AppleDouble xattr sidecars
( cd "$STAGE" && COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' \
    -czf "$OLDPWD/arxiv-submission.tar.gz" "$STEM.tex" figures )
```

Verify before uploading: `tar -tzf arxiv-submission.tar.gz` should list 21 entries (one `.tex`, `figures/`, 19 PNGs) and nothing else. Then extract to an empty directory and run `xelatex` three times; it must reach 23 pages with no `File ... not found`.

Engine: arXiv runs TeX Live 2025 and has supported `xelatex` since November 2025. The `.tex` also compiles under `pdflatex` (the `\ifPDFTeX` branch plus the `\newunicodechar` maps cover every non-ASCII glyph), so either processor selection works.

## T4.5 outcome (env gate)
- Kernel: use `preprint-build` (uv env); anaconda `python3` kernel is crippled (kaleido/jax/tf stubbed).
- Installed: jinja2 (pandas .style). fancyhdr via tlmgr.
- Current notebook executes through data-load + matplotlib cells; the 3 INLINE plotly cells fail on plotly-6/kaleido-1 API change. FIX = remediation #1 (refactor chart cells to narrative_charts.py `_plotly_write_image`, which works) in T8. Env otherwise clean.

## Toolchain versions used for the adopted build (2026-07-05)

- pandoc: pandoc 3.8.2
- xelatex: XeTeX 3.141592653-2.6-0.999997 (TeX Live 2025)
- Build shim: `notebooks/preprint/document-metadata.latex` supplies the pandoc partial that pandoc 3.8.2 no longer ships (reproduces the empty output of the prior build).
