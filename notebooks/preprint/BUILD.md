# Preprint build — environment & commands

The preprint is built **notebook-first**: update the notebook → execute → export markdown → pandoc + custom LaTeX template → PDF. This file records the exact, reproducible build environment (hard-won during Task 4/4.5).

## Kernel (IMPORTANT)
Execute the notebook with the **`preprint-build`** kernel, registered from the project `uv` env:

```bash
uv run python -m ipykernel install --user --name preprint-build --display-name "preprint-build"
```

**Do NOT use the anaconda `python3` kernel** — its kernelspec is hand-crippled: it stubs out `kaleido`/`tensorflow`/`jax` (`sys.modules[m]=None`) and rewrites `sys.argv`, which (a) breaks under `nbconvert --execute` (`--IPKernelApp.connection_file: expected one argument` → "Kernel died"), and (b) would break the 3 plotly charts (they need kaleido for PNG export). The `preprint-build` kernel (uv env) has the full stack **plus a working kaleido and the importable `engine` package**.

## LaTeX packages
`fancyhdr` is required by `arxiv-template.latex` and is NOT pre-installed in the local TeX tree:

```bash
tlmgr install fancyhdr
```

(arXiv's own TeXLive has `fancyhdr`, so LaTeX-source submission compiles there.)

## Build command
```bash
python tools/build_preprint.py \
  --notebook notebooks/2026_top_10_llm_update_what_the_data_says.ipynb \
  --out-dir notebooks/preprint
```
This executes a COPY of the notebook (never mutates the committed `.ipynb`), exports markdown with outputs stripped, prepends the YAML front-matter, and compiles via `pandoc --template=notebooks/preprint/arxiv-template.latex --pdf-engine=xelatex`.

## arXiv submission
Submit the generated LaTeX **source** (`.tex` + `figures/`), not PDF-only, so arXiv can recompile.

## T4.5 outcome (env gate)
- Kernel: use `preprint-build` (uv env); anaconda `python3` kernel is crippled (kaleido/jax/tf stubbed).
- Installed: jinja2 (pandas .style). fancyhdr via tlmgr.
- Current notebook executes through data-load + matplotlib cells; the 3 INLINE plotly cells fail on plotly-6/kaleido-1 API change. FIX = remediation #1 (refactor chart cells to narrative_charts.py `_plotly_write_image`, which works) in T8. Env otherwise clean.
