"""Tests for the notebook→PDF preprint build pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.build_preprint import build_preprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
PRE = _REPO_ROOT / "notebooks" / "preprint"


def _kernel_available(name: str) -> bool:
    try:
        r = subprocess.run(
            ["jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        return name in r.stdout
    except Exception:
        return False


_TOOLCHAIN_OK = (
    shutil.which("pandoc") is not None
    and shutil.which("xelatex") is not None
    and _kernel_available("preprint-build")
)

_FIXTURE_NB: dict[str, object] = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "preprint-build",
            "language": "python",
            "name": "preprint-build",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": [
        {
            "cell_type": "markdown",
            "id": "a1b2c3d4-aaaa-bbbb-cccc-000000000001",
            "metadata": {},
            "source": (
                "## Introduction\n\n"
                "This fixture validates the build pipeline.\n\n"
                "![Test chart](figures/test.png){width=85%}"
            ),
        },
        {
            "cell_type": "code",
            "id": "a1b2c3d4-aaaa-bbbb-cccc-000000000002",
            "metadata": {},
            "source": (
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n"
                "import os\n"
                "os.makedirs('figures', exist_ok=True)\n"
                "fig, ax = plt.subplots()\n"
                "ax.plot([1, 2, 3], [4, 5, 6])\n"
                "fig.savefig('figures/test.png', dpi=72)\n"
                "plt.close('all')"
            ),
            "outputs": [],
            "execution_count": None,
        },
        {
            "cell_type": "markdown",
            "id": "a1b2c3d4-aaaa-bbbb-cccc-000000000003",
            "metadata": {},
            "source": "## Conclusion\n\nTest pipeline complete.",
        },
    ],
}

_FRONT_MATTER = """\
---
title: "Test Preprint"
date: "2026"
numbersections: true
---
"""


# ---------------------------------------------------------------------------
# Existing Task-4 test (stub compiles via pandoc + xelatex)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("pandoc") is None or shutil.which("xelatex") is None,
    reason="toolchain (pandoc and/or xelatex) not available",
)
def test_stub_compiles_with_template(tmp_path: Path) -> None:
    out = tmp_path / "stub.pdf"
    r = subprocess.run(
        [
            "pandoc",
            str(PRE / "_stub.md"),
            "-o",
            str(out),
            "--pdf-engine=xelatex",
            f"--template={PRE / 'arxiv-template.latex'}",
            "--from=markdown+yaml_metadata_block",
            "--toc",
        ],
        capture_output=True,
        text=True,
        cwd=str(PRE),
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.stat().st_size > 10 * 1024


# ---------------------------------------------------------------------------
# Task-5 test: build_preprint pipeline on a 3-cell fixture notebook
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _TOOLCHAIN_OK,
    reason="toolchain (pandoc/xelatex/preprint-build kernel) not available",
)
def test_build_preprint_fixture(tmp_path: Path) -> None:
    """Execute a 3-cell fixture notebook through the full pipeline; assert PDF > 10 KB."""
    # Arrange: fixture notebook in its own subdirectory
    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    fixture_nb = nb_dir / "fixture.ipynb"
    fixture_nb.write_text(json.dumps(_FIXTURE_NB))

    out_dir = tmp_path / "out"

    front_matter = tmp_path / "front_matter.md"
    front_matter.write_text(_FRONT_MATTER)

    template = PRE / "arxiv-template.latex"

    # Act
    pdf = build_preprint(
        notebook=fixture_nb,
        out_dir=out_dir,
        front_matter_md=front_matter,
        template=template,
    )

    # Assert
    assert pdf.exists(), f"PDF not produced at {pdf}"
    size = pdf.stat().st_size
    assert size > 10 * 1024, f"PDF too small: {size} bytes"
