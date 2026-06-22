from engine.prereg.lock import compute_lock_hash
from engine.prereg.manifest import PreregManifest

_BASE = dict(
    engine_version="1.2.0", engine_version_range_min="1.0.0",
    engine_version_range_max="2.0.0", cycle_id="t", taxonomy_hash="tx",
    snapshot_hash="sn", primary_spec="negative_binomial_per_stratum",
    robustness_specs=(), flag_threshold_tau=0.8, statistic="weighted_cohens_kappa",
    measurability_minimum=4, prior_scale=0.5, concentration_shape=5.0,
    concentration_rate=0.1, ess_fraction=0.4, meaningful_kappa_n=4, prng_seed=42,
    confidence_threshold=0.3, rubric_drafting_attestation=None,
    rubric_reviewer=None, statistical_reviewer=None, classifier_rule_hash=None,
    rubric_hash=None, post_hoc_register_path=None,
)


def test_v1_manifest_excludes_new_fields_from_hash():
    m = PreregManifest(**_BASE)  # schema_version defaults to 1
    d = m.to_dict()
    assert "schema_version" not in d
    assert "goldset_hash" not in d


def test_adding_goldset_hash_under_v1_does_not_change_hash():
    # A v1 manifest with a goldset_hash set but schema_version still 1 must hash
    # identically to one without it -- the field is not part of the v1 canonical form.
    m_plain = PreregManifest(**_BASE)
    m_with = PreregManifest(**_BASE, goldset_hash="abc123")
    assert compute_lock_hash(m_plain) == compute_lock_hash(m_with)


def test_v2_manifest_includes_new_fields_and_changes_hash():
    m1 = PreregManifest(**_BASE)
    m2 = PreregManifest(**_BASE, schema_version=2, goldset_hash="abc123")
    d2 = m2.to_dict()
    assert d2["schema_version"] == 2
    assert d2["goldset_hash"] == "abc123"
    assert compute_lock_hash(m1) != compute_lock_hash(m2)
