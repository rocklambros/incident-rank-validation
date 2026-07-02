"""Smoke-test: pandoc + xelatex compile the preprint stub through the arXiv template."""

import shutil
import subprocess
from pathlib import Path

import pytest

# Resolve absolute path from project root so paths survive any pytest invocation cwd.
_REPO_ROOT = Path(__file__).parent.parent.parent
PRE = _REPO_ROOT / "notebooks" / "preprint"


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
