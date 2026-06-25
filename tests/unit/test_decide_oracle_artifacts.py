"""Tests for build_incidence_ranking_artifact (Plan 8d Task 5)."""
from __future__ import annotations

import numpy as np

from engine.cli.pipeline import build_incidence_ranking_artifact


def test_incidence_artifact_shape_and_order() -> None:
    lambda_samples = np.array([[0.5, 0.1], [0.6, 0.2], [0.4, 0.15]])
    entry_ids = ("A", "B")
    common = ["A", "B"]
    entry_strata: dict[str, tuple[str, ...]] = {"A": ("security",), "B": ("security",)}
    stratum_sizes = {"security": 10}
    art = build_incidence_ranking_artifact(
        lambda_samples, entry_ids, common, entry_strata, stratum_sizes
    )
    assert art["ranking"] == ["A", "B"]
    assert set(art["incidence_median"].keys()) == {"A", "B"}  # type: ignore[attr-defined]
    for lo, hi in art["incidence_ci"].values():  # type: ignore[attr-defined]
        assert lo <= hi


def test_incidence_artifact_is_json_serializable() -> None:
    import json

    lambda_samples = np.array([[0.5, 0.1], [0.6, 0.2]])
    art = build_incidence_ranking_artifact(
        lambda_samples, ("A", "B"), ["A", "B"],
        {"A": ("s",), "B": ("s",)}, {"s": 5},
    )
    restored = json.loads(json.dumps(art))
    assert restored["ranking"] == ["A", "B"]
