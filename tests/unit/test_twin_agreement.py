"""Unit tests for engine.decide.twin_agreement."""

from __future__ import annotations

import numpy as np
import pytest

from engine.decide.twin_agreement import (
    compare_twin_nuts,
)
from engine.model.inference import InferenceResult
from engine.model.twin import TwinResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inference_result(
    entry_ids: tuple[str, ...],
    medians: tuple[float, ...],
    n_samples: int = 100,
) -> InferenceResult:
    """Construct a fake InferenceResult with controlled posterior medians."""
    rng = np.random.default_rng(0)
    n = len(entry_ids)
    # Each column is drawn from Normal(median, 0.01) so the sample median
    # closely matches the desired value.
    lambda_samples = np.column_stack(
        [rng.normal(loc=m, scale=0.01, size=n_samples) for m in medians]
    ).astype(np.float64)
    return InferenceResult(
        lambda_samples=lambda_samples,
        entry_ids=entry_ids,
        r_hat={f"lambda[{i}]": 1.0 for i in range(n)},
        ess={f"lambda[{i}]": float(n_samples) for i in range(n)},
        divergences=0,
        num_warmup=50,
        num_samples=n_samples,
    )


def _make_twin_result(
    entry_ids: tuple[str, ...],
    prevalences: tuple[float, ...],
) -> TwinResult:
    """Construct a TwinResult with given prevalence estimates."""
    prev_dict = dict(zip(entry_ids, prevalences, strict=False))
    rank = tuple(sorted(prev_dict, key=lambda e: prev_dict[e], reverse=True))
    return TwinResult(
        entry_ids=entry_ids,
        prevalence_estimates=prev_dict,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompareTwinNuts:
    def test_perfect_agreement_no_disagreements(self) -> None:
        """When NUTS and twin agree on all pairwise directions, no disagreements."""
        entries = ("A", "B", "C")
        inference = _make_inference_result(entries, medians=(0.3, 0.2, 0.1))
        twin = _make_twin_result(entries, prevalences=(0.3, 0.2, 0.1))

        result = compare_twin_nuts(inference, twin, top_n=3)

        assert result.n_comparisons == 3  # C(3,2) = 3
        assert len(result.disagreements) == 0
        assert result.agreement_rate == pytest.approx(1.0)
        assert not result.has_disagreements

    def test_disagreement_detected_when_rankings_differ(self) -> None:
        """Disagreement is detected when twin and NUTS disagree on a pair direction."""
        entries = ("A", "B", "C")
        # NUTS: A > B > C; twin: B > A > C (A/B pair disagrees)
        inference = _make_inference_result(entries, medians=(0.3, 0.2, 0.1))
        twin = _make_twin_result(entries, prevalences=(0.25, 0.30, 0.1))

        result = compare_twin_nuts(inference, twin, top_n=3)

        assert result.has_disagreements
        # Find the A-B disagreement
        ab_disagreements = [
            d
            for d in result.disagreements
            if {d.entry_a, d.entry_b} == {"A", "B"}
        ]
        assert len(ab_disagreements) == 1

    def test_agreement_rate_is_correct(self) -> None:
        """agreement_rate = 1 - (n_disagreements / n_comparisons)."""
        entries = ("A", "B", "C")
        # NUTS: A > B > C; twin: B > A > C (1 disagreement out of 3 pairs)
        inference = _make_inference_result(entries, medians=(0.3, 0.2, 0.1))
        twin = _make_twin_result(entries, prevalences=(0.25, 0.30, 0.1))

        result = compare_twin_nuts(inference, twin, top_n=3)

        expected_rate = 1.0 - (1 / 3)
        assert result.agreement_rate == pytest.approx(expected_rate)

    def test_has_disagreements_property(self) -> None:
        """has_disagreements returns True iff there is at least one disagreement."""
        entries = ("A", "B")
        inference_agree = _make_inference_result(entries, medians=(0.5, 0.2))
        twin_agree = _make_twin_result(entries, prevalences=(0.5, 0.2))

        inference_disagree = _make_inference_result(entries, medians=(0.5, 0.2))
        twin_disagree = _make_twin_result(entries, prevalences=(0.2, 0.5))

        assert not compare_twin_nuts(inference_agree, twin_agree).has_disagreements
        assert compare_twin_nuts(
            inference_disagree, twin_disagree
        ).has_disagreements

    def test_top_n_limits_entries_compared(self) -> None:
        """Only entries in top_n from either ranking are compared."""
        entries = ("A", "B", "C", "D", "E")
        # All agree; top_n=2 means only pairs from {top-2 NUTS} | {top-2 twin}
        inference = _make_inference_result(
            entries, medians=(0.5, 0.4, 0.3, 0.2, 0.1)
        )
        twin = _make_twin_result(
            entries, prevalences=(0.5, 0.4, 0.3, 0.2, 0.1)
        )

        result_top2 = compare_twin_nuts(inference, twin, top_n=2)
        result_top5 = compare_twin_nuts(inference, twin, top_n=5)

        # top_n=2 considers only {A, B}: 1 pair; top_n=5 considers all 5: 10 pairs
        assert result_top2.n_comparisons == 1
        assert result_top5.n_comparisons == 10

    def test_frozen_dataclasses_immutable(self) -> None:
        """TwinAgreement and TwinDisagreement must be immutable."""
        entries = ("A", "B")
        inference = _make_inference_result(entries, medians=(0.5, 0.2))
        twin = _make_twin_result(entries, prevalences=(0.5, 0.2))
        result = compare_twin_nuts(inference, twin)

        with pytest.raises((AttributeError, TypeError)):
            result.agreement_rate = 0.0  # type: ignore[misc]
