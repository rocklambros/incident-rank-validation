"""Regression test: corpus B must never enter the inference module.

HANDOFF §4 Corpus B role: 'Not a modeled Bayesian channel.'
HANDOFF §5.4: 'Corpus B is not in the likelihood.'
"""
from __future__ import annotations

import ast
from pathlib import Path

_INFERENCE_PATH = Path(__file__).resolve().parents[2] / "engine" / "model" / "inference.py"

_CORPUS_B_MARKERS = frozenset({
    "owasp_asi",
    "corpus_b",
    "corroboration",
    "asi_agentic",
    "ASI_Agentic",
    "ASIB-",
})


def test_inference_has_no_corpus_b_imports() -> None:
    """Assert inference.py does not import any corpus B module."""
    source = _INFERENCE_PATH.read_text()
    tree = ast.parse(source, filename=str(_INFERENCE_PATH))

    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.append(node.module)
            for alias in node.names:
                imported_names.append(alias.name)

    for name in imported_names:
        for marker in _CORPUS_B_MARKERS:
            assert marker not in name, (
                f"inference.py imports '{name}' which contains corpus B marker "
                f"'{marker}'. Corpus B must NEVER enter the likelihood "
                f"(HANDOFF §4, §5.4)."
            )


def test_inference_source_has_no_corpus_b_references() -> None:
    """Assert inference.py source text has no corpus B references."""
    source = _INFERENCE_PATH.read_text()
    for marker in _CORPUS_B_MARKERS:
        assert marker not in source, (
            f"inference.py contains corpus B marker '{marker}'. "
            f"Corpus B is corroboration only — never a posterior input."
        )
