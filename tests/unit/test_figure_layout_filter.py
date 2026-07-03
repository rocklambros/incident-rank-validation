from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PRE = Path(__file__).resolve().parents[2] / "notebooks" / "preprint"
LUA = PRE / "figure-layout.lua"
TEMPLATE = PRE / "arxiv-template.latex"

# pandoc is a system tool, not a Python dep — skip the tests that invoke it when
# it is absent (e.g. CI runners without pandoc), matching test_preprint_build.py.
_requires_pandoc = pytest.mark.skipif(
    shutil.which("pandoc") is None, reason="pandoc not available"
)


def _pandoc_latex(md: str) -> str:
    r = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex", f"--lua-filter={LUA}"],
        input=md,
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout


@_requires_pandoc
def test_wrap_becomes_wrapfigure() -> None:
    out = _pandoc_latex("![Cap with 50% sign](figures/x.png){width=42% wrap=right}\n")
    assert "\\begin{wrapfigure}{R}" in out
    assert "0.42\\textwidth" in out
    assert "50\\%" in out  # % escaped, not swallowed


@_requires_pandoc
def test_center_becomes_figure_htbp() -> None:
    out = _pandoc_latex("![Latent incidence (λ)](figures/y.png){width=72%}\n")
    assert "\\begin{figure}[htbp]" in out
    assert "0.72\\textwidth" in out


def test_template_maps_greek_math_glyphs() -> None:
    """The Latin Modern text font lacks these glyphs; the template must map each
    via \\newunicodechar so xelatex does not silently drop them (regression guard
    for the dropped Cohen's κ / Spearman ρ / latent incidence λ in the abstract)."""
    tpl = TEMPLATE.read_text(encoding="utf-8")
    for glyph in ("κ", "λ", "ρ", "≈", "×", "−"):
        assert f"\\newunicodechar{{{glyph}}}" in tpl, f"missing \\newunicodechar for {glyph!r}"
