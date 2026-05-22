"""Two-cycle parity holdout (M17): synthetic re-run matches headline shape.

Per PRD §6.6 criterion 13 and HANDOFF §11 v2.3→v2.4: a synthetic re-run
on the same engine version produces matching headline shape. The 30-day
audit window opens after this test passes.

This is a publication gate, not an execution gate.

R5: This test actually runs execute_synthetic_pipeline() twice and compares
structural outputs, rather than being a skeleton with a version assertion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")


def _setup_synthetic_cycle(base: Path) -> Path:
    """Create the directory structure execute_synthetic_pipeline() expects."""
    project_root = base / "projects" / "owasp-llm"
    cycle = project_root / "cycles" / "2026"
    cycle.mkdir(parents=True)
    (project_root / "project.toml").write_text(
        '[project]\n'
        'name = "synthetic"\n'
        'cycle_id = "2026-parity"\n'
        'prng_seed = 42\n'
        'measurability_minimum = 4\n'
        'tier_size = 2\n'
        '\n'
        '[project.hyperparameters]\n'
        'flag_threshold_tau = 0.8\n'
        'prior_scale = 0.5\n'
        'concentration_shape = 5.0\n'
        'concentration_rate = 0.1\n'
        'ess_fraction = 0.4\n'
        'meaningful_kappa_n = 4\n'
        'confidence_threshold = 0.3\n'
    )
    return cycle


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_two_cycle_parity_headline_shape(tmp_path: Path) -> None:
    """Run two identical synthetic cycles and verify headline shape matches."""
    from engine.cli.synthetic import execute_synthetic_pipeline

    cycle1 = _setup_synthetic_cycle(tmp_path / "run1")
    cycle2 = _setup_synthetic_cycle(tmp_path / "run2")

    execute_synthetic_pipeline(cycle1, corpus_mode="synthetic")
    execute_synthetic_pipeline(cycle2, corpus_mode="synthetic")

    s1 = json.loads((cycle1 / "results" / "summary.json").read_text())
    s2 = json.loads((cycle2 / "results" / "summary.json").read_text())

    assert s1["measurable_count"] == s2["measurable_count"]
    assert s1["frame_blind_count"] == s2["frame_blind_count"]
    assert s1["coverage_ratio"] == s2["coverage_ratio"]
    assert s1["non_publishable"] == s2["non_publishable"]
    assert s1["nuts_succeeded"] == s2["nuts_succeeded"]

    if s1["nuts_succeeded"] and s2["nuts_succeeded"]:
        k1 = s1.get("weighted_kappa_median")
        k2 = s2.get("weighted_kappa_median")
        if k1 is not None and k2 is not None:
            assert abs(k1 - k2) < 0.15, (
                f"Kappa diverged beyond MCSE tolerance: {k1:.3f} vs {k2:.3f}"
            )


def test_m17_is_publication_gate_not_execution_gate() -> None:
    """Document that M17 does not block execution or tagging."""
    pass
