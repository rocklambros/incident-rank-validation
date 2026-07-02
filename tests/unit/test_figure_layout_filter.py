from __future__ import annotations

import subprocess
from pathlib import Path

PRE = Path(__file__).resolve().parents[2] / "notebooks" / "preprint"
LUA = PRE / "figure-layout.lua"


def _pandoc_latex(md: str) -> str:
    r = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex", f"--lua-filter={LUA}"],
        input=md,
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout


def test_wrap_becomes_wrapfigure() -> None:
    out = _pandoc_latex("![Cap with 50% sign](figures/x.png){width=42% wrap=right}\n")
    assert "\\begin{wrapfigure}{R}" in out
    assert "0.42\\textwidth" in out
    assert "50\\%" in out  # % escaped, not swallowed


def test_center_becomes_figure_htbp() -> None:
    out = _pandoc_latex("![Latent incidence (λ)](figures/y.png){width=72%}\n")
    assert "\\begin{figure}[htbp]" in out
    assert "0.72\\textwidth" in out
