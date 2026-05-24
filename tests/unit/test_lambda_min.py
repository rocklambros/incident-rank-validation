"""Test lambda_min field on PreregManifest."""
from __future__ import annotations

from engine.prereg.manifest import PreregManifest


def _make_manifest(**overrides) -> PreregManifest:
    defaults = {
        "engine_version": "1.1.0",
        "engine_version_range_min": "1.0.0",
        "engine_version_range_max": "1.99.0",
        "cycle_id": "test-cycle",
        "taxonomy_hash": "abc",
        "snapshot_hash": "def",
        "primary_spec": "negative_binomial_per_stratum",
        "robustness_specs": (),
        "flag_threshold_tau": 0.5,
        "statistic": "weighted_cohens_kappa",
        "measurability_minimum": 4,
        "prior_scale": 0.5,
        "concentration_shape": 5.0,
        "concentration_rate": 0.1,
        "ess_fraction": 0.4,
        "meaningful_kappa_n": 4,
        "prng_seed": 42,
        "confidence_threshold": 0.3,
        "classifier_rule_hash": None,
        "rubric_hash": None,
        "rubric_drafting_attestation": None,
        "rubric_reviewer": None,
        "statistical_reviewer": None,
        "post_hoc_register_path": None,
    }
    defaults.update(overrides)
    return PreregManifest(**defaults)


def test_lambda_min_default_from_prior_scale() -> None:
    m = _make_manifest(prior_scale=0.5)
    assert m.lambda_min == 0.5 * 0.02  # 0.01


def test_lambda_min_explicit_override() -> None:
    m = _make_manifest(prior_scale=0.5, lambda_min=0.05)
    assert m.lambda_min == 0.05


def test_lambda_min_in_to_dict() -> None:
    m = _make_manifest()
    d = m.to_dict()
    assert "lambda_min" in d
    assert d["lambda_min"] == m.lambda_min
