"""Notebook → markdown → PDF build pipeline for the arXiv preprint.

Usage (CLI)::

    python tools/build_preprint.py \\
        --notebook notebooks/2026_top_10_llm_update_what_the_data_says.ipynb \\
        --out-dir notebooks/preprint

Steps executed by :func:`build_preprint`:

1. Copy the source notebook to *out_dir* (never mutate the committed ``.ipynb``).
2. Execute the copy with the ``preprint-build`` kernel so figure-producing cells run.
3. Clear **all** cell outputs from the executed copy (strips base64 blobs, DataFrames,
   stdout) so the subsequent markdown export contains only prose and figure refs.
4. Export the cleaned copy to Markdown via ``jupyter nbconvert --no-input`` (code inputs
   hidden; prose cells and ``![…](figures/…){width=85%}`` refs preserved).
5. Prepend *front_matter_md* (the YAML front-matter block) to the exported ``.md``.
6. Compile to PDF via ``pandoc --pdf-engine=xelatex --template=<template> --toc``,
   and to a standalone ``.tex`` via ``pandoc --standalone``, both routed through the
   ``figure-layout.lua`` filter (or *lua_filter* override) so figures wrap/center.

Emits ``<stem>.md``, ``<stem>.pdf``, and ``<stem>.tex`` where ``stem`` is *output_name*
(falling back to the notebook's stem). Returns the path to the generated PDF.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def build_preprint(
    notebook: Path,
    out_dir: Path,
    front_matter_md: Path,
    template: Path,
    output_name: str | None = None,
    lua_filter: Path | None = None,
) -> Path:
    """Execute notebook copy, export stripped markdown, prepend front-matter, compile PDF.

    Args:
        notebook: Path to the source ``.ipynb`` (never mutated).
        out_dir: Output directory; receives the executed copy, markdown, PDF, and TeX.
        front_matter_md: Markdown file whose content (YAML ``---`` block) is prepended to
            the nbconvert output before pandoc compilation.
        template: LaTeX template passed to ``pandoc --template``.
        output_name: Stem for the emitted ``.md``/``.pdf``/``.tex`` files. Defaults to
            ``notebook.stem`` when omitted.
        lua_filter: Pandoc Lua filter applied to both the PDF and ``.tex`` builds.
            Defaults to ``figure-layout.lua`` alongside *template*.

    Returns:
        Path to the generated PDF file.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create figures/ so notebook code cells can savefig without mkdir calls
    (out_dir / "figures").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Execute a COPY of the notebook (never mutate the source)
    # ------------------------------------------------------------------
    copy = out_dir / (notebook.stem + "_executed.ipynb")
    shutil.copy2(notebook, copy)

    # The copy lives in out_dir; nbconvert sets kernel CWD to the notebook's
    # parent → out_dir.  Code cells that call savefig('figures/…') therefore
    # write into out_dir/figures/, which is where pandoc will look for them.
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.kernel_name=preprint-build",
            "--ExecutePreprocessor.startup_timeout=90",
            "--ExecutePreprocessor.timeout=1800",
            str(copy),
        ],
        check=True,
        cwd=str(out_dir),
    )

    # ------------------------------------------------------------------
    # Step 2 — Clear ALL cell outputs (base64 images, DataFrames, stdout)
    # ------------------------------------------------------------------
    with open(copy) as fh:
        nb_json: dict[str, object] = json.load(fh)

    cells = nb_json.get("cells", [])
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    with open(copy, "w") as fh:
        json.dump(nb_json, fh)

    # ------------------------------------------------------------------
    # Step 3 — Export to Markdown (no code inputs, no outputs)
    # ------------------------------------------------------------------
    stem = output_name or notebook.stem
    md_name = stem + ".md"
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            "--no-input",
            str(copy),
            "--output-dir",
            str(out_dir),
            "--output",
            md_name,
        ],
        check=True,
        cwd=str(out_dir),
    )

    # ------------------------------------------------------------------
    # Step 4 — Prepend YAML front-matter block
    # ------------------------------------------------------------------
    md_path = out_dir / md_name
    front_matter_text = front_matter_md.read_text()
    existing_md = md_path.read_text()
    md_path.write_text(front_matter_text + "\n" + existing_md)

    # ------------------------------------------------------------------
    # Step 5 — pandoc → PDF, pandoc → TeX (both via the shared Lua filter)
    # ------------------------------------------------------------------
    pdf_path = out_dir / (stem + ".pdf")
    tex_path = out_dir / (stem + ".tex")
    lf = lua_filter or (template.parent / "figure-layout.lua")
    common = [
        "--from=markdown+yaml_metadata_block",
        f"--template={template}",
        f"--lua-filter={lf}",
        # Resolve [@key] citations against the bibliography named in the front-matter.
        # citeproc renders the reference list into the output itself, so the .tex is
        # self-contained: arXiv needs no .bib and no .bbl alongside it.
        "--citeproc",
        "--toc",
        "--toc-depth=2",
    ]
    subprocess.run(
        ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=xelatex", *common],
        check=True,
        cwd=str(out_dir),
    )
    subprocess.run(
        ["pandoc", str(md_path), "-o", str(tex_path), "--standalone", *common],
        check=True,
        cwd=str(out_dir),
    )

    return pdf_path


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Jupyter notebook into a preprint PDF via pandoc + xelatex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--notebook",
        required=True,
        type=Path,
        metavar="NOTEBOOK",
        help="Path to the source notebook (.ipynb).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory (receives PDF, markdown, executed copy).",
    )
    parser.add_argument(
        "--front-matter",
        type=Path,
        default=Path("notebooks/preprint/front_matter.md"),
        metavar="FILE",
        help="YAML front-matter file (default: notebooks/preprint/front_matter.md).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("notebooks/preprint/arxiv-template.latex"),
        metavar="FILE",
        help="LaTeX template (default: notebooks/preprint/arxiv-template.latex).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        metavar="STEM",
        help="Stem for the emitted .md/.pdf/.tex files (default: notebook stem).",
    )
    args = parser.parse_args()
    pdf = build_preprint(
        notebook=args.notebook.resolve(),
        out_dir=args.out_dir.resolve(),
        front_matter_md=args.front_matter.resolve(),
        template=args.template.resolve(),
        output_name=args.output_name,
    )
    print(f"PDF written to: {pdf}")


if __name__ == "__main__":
    _cli()
