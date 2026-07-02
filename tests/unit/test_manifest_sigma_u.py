"""Tests for sigma_u_hyperprior_scale manifest field (schema v2, Plan 8b Task 1)."""

from engine.prereg.lock import compute_lock_hash
from tests.unit.test_prereg import _make_manifest


def test_sigma_u_field_excluded_from_v1_canonical_form() -> None:
    m = _make_manifest()  # schema_version defaults to 1
    assert "sigma_u_hyperprior_scale" not in m.to_dict()


def test_sigma_u_under_v1_does_not_change_hash() -> None:
    base = _make_manifest()
    with_sigma = _make_manifest(sigma_u_hyperprior_scale=2.0)
    assert compute_lock_hash(base) == compute_lock_hash(with_sigma)


def test_sigma_u_included_and_hash_changes_at_v2() -> None:
    base = _make_manifest()
    v2 = _make_manifest(schema_version=2, sigma_u_hyperprior_scale=2.0)
    assert v2.to_dict()["sigma_u_hyperprior_scale"] == 2.0
    assert compute_lock_hash(base) != compute_lock_hash(v2)
