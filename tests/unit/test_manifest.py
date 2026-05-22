"""Unit tests for PreregManifest rollup threshold fields (R8)."""

from __future__ import annotations

from typing import Any

from engine.prereg.manifest import PreregManifest


def _make_manifest(**overrides: Any) -> PreregManifest:
    defaults: dict[str, Any] = {
        "engine_version": "0.1.0",
        "engine_version_range_min": "0.1.0",
        "engine_version_range_max": "0.2.0",
        "cycle_id": "test-cycle-001",
        "taxonomy_hash": "aaa",
        "snapshot_hash": "bbb",
        "primary_spec": "negative_binomial_per_stratum",
        "robustness_specs": ("poisson_flat",),
        "flag_threshold_tau": 0.8,
        "statistic": "weighted_cohens_kappa",
        "measurability_minimum": 10,
        "prior_scale": 0.5,
        "concentration_shape": 5.0,
        "concentration_rate": 0.1,
        "ess_fraction": 0.4,
        "meaningful_kappa_n": 4,
        "prng_seed": 42,
        "confidence_threshold": 0.3,
        "rubric_drafting_attestation": None,
        "rubric_reviewer": None,
        "statistical_reviewer": None,
        "classifier_rule_hash": None,
        "rubric_hash": None,
        "post_hoc_register_path": None,
    }
    defaults.update(overrides)
    return PreregManifest(**defaults)


def test_manifest_has_rollup_thresholds() -> None:
    """R8: rollup thresholds are pre-registered in the manifest."""
    m = _make_manifest(
        rollup_threshold=0.01,
        rollup_p_supported=0.8,
        rollup_p_contradicted=0.2,
    )
    assert m.rollup_threshold == 0.01
    assert m.rollup_p_supported == 0.8
    assert m.rollup_p_contradicted == 0.2
    d = m.to_dict()
    assert d["rollup_threshold"] == 0.01
    assert d["rollup_p_supported"] == 0.8
    assert d["rollup_p_contradicted"] == 0.2


def test_manifest_rollup_defaults() -> None:
    """Rollup thresholds have sensible defaults for backward compatibility."""
    m = _make_manifest()
    assert m.rollup_threshold == 0.01
    assert m.rollup_p_supported == 0.8
    assert m.rollup_p_contradicted == 0.2
