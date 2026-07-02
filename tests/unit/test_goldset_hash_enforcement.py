import pytest

from engine.calibrate.gold_schema import GoldCalibration
from engine.cli.pipeline_executor import _verify_goldset_hash
from tests.unit.test_prereg import _make_manifest


def _gold(h: str) -> GoldCalibration:
    return GoldCalibration(
        recall_labels=[], precision_labels=[], provenance_hash=h,
        rubric_hash="r", adjudicator_id="t", session_count=1,
    )


def test_mismatch_raises() -> None:
    m = _make_manifest(schema_version=2, goldset_hash="expected")
    with pytest.raises(RuntimeError, match="goldset"):
        _verify_goldset_hash(m, _gold("ACTUAL_DIFFERENT"))


def test_match_passes() -> None:
    m = _make_manifest(schema_version=2, goldset_hash="abc123")
    _verify_goldset_hash(m, _gold("abc123"))  # must not raise


def test_unbound_goldset_hash_is_noop() -> None:
    m = _make_manifest()  # v1, goldset_hash None
    _verify_goldset_hash(m, _gold("anything"))  # must not raise


# U2-4: verify the full sentinel acceptance matrix
def test_verify_goldset_hash_accepts_none_empty_and_real() -> None:
    """_verify_goldset_hash must be a no-op for None, '', and 'none'; raise on real mismatch."""
    # None (v1 manifest — goldset_hash field is None)
    _verify_goldset_hash(_make_manifest(), _gold("anything"))

    # Empty string sentinel (new default from U2-4)
    m_empty = _make_manifest(schema_version=2, goldset_hash="")
    _verify_goldset_hash(m_empty, _gold("anything"))

    # Legacy "none" string (pre-U2 bundles)
    m_none = _make_manifest(schema_version=2, goldset_hash="none")
    _verify_goldset_hash(m_none, _gold("anything"))

    # Real hash that matches — must pass
    m_real = _make_manifest(schema_version=2, goldset_hash="abc123")
    _verify_goldset_hash(m_real, _gold("abc123"))

    # Real hash mismatch — must raise
    with pytest.raises(RuntimeError, match="goldset"):
        _verify_goldset_hash(m_real, _gold("DIFFERENT"))
