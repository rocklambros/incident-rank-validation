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
