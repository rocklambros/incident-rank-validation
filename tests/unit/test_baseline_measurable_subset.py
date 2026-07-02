"""Tests for compute_measurable_subset_kappa (T3, U3 Cluster A).

Verifies:
- Subset kappa differs from full-set kappa (the key finding)
- STANDING_CAVEAT contradiction is present in the result
- n_measurable is correctly restricted
- Real-data: subset kappa (~0.12) differs from 20-entry kappa (0.2029)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.baselines.measurable_subset import (
    STANDING_CAVEAT_CONTRADICTION,
    compute_measurable_subset_kappa,
)
from engine.baselines.previous_ranking import compute_previous_ranking
from tests.unit.fixtures.baselines import (
    ENTRY_IDS_A,
    ENTRY_STRATA_A,
    KAPPA_SUBSET_A,
    LAMBDA_SAMPLES_A,
    MEASURABLE_IDS_A,
    STRATUM_SIZES_A,
    VOTE_RANK_SAMPLES_A,
)

# ---------------------------------------------------------------------------
# Fixture-based tests
# ---------------------------------------------------------------------------


def test_fixture_a_subset_kappa_differs_from_full() -> None:
    """Measurable subset (E1, E2) kappa != full 4-entry kappa."""
    full_result = compute_previous_ranking(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    sub_result = compute_measurable_subset_kappa(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    assert full_result.kappa_median != sub_result.kappa_median, (
        "subset kappa must differ from full kappa (they share only 2 of 4 entries)"
    )


def test_fixture_a_subset_kappa_hand_computed() -> None:
    """Subset kappa for (E1, E2) == -1.0 (hand-computed)."""
    result = compute_measurable_subset_kappa(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    assert abs(result.kappa_median - KAPPA_SUBSET_A) < 1e-12, (
        f"subset kappa {result.kappa_median} != hand-computed {KAPPA_SUBSET_A}"
    )


def test_fixture_a_n_measurable() -> None:
    """n_measurable == 2 (E1, E2 only)."""
    result = compute_measurable_subset_kappa(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    assert result.n_measurable == len(MEASURABLE_IDS_A), (
        f"n_measurable={result.n_measurable} != {len(MEASURABLE_IDS_A)}"
    )


def test_standing_caveat_contradiction_present() -> None:
    """standing_caveat_contradiction is the module-level constant string."""
    result = compute_measurable_subset_kappa(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    assert result.standing_caveat_contradiction == STANDING_CAVEAT_CONTRADICTION
    assert "concordance.py:48" in result.standing_caveat_contradiction
    assert "20" in result.standing_caveat_contradiction  # references n=20
    assert "17" in result.standing_caveat_contradiction  # references measurable count


def test_full_measurable_equals_previous_ranking() -> None:
    """When measurable_ids == all entries, subset kappa == full kappa."""
    result_full = compute_previous_ranking(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    result_sub = compute_measurable_subset_kappa(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=ENTRY_IDS_A,  # all entries measurable
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    assert abs(result_sub.kappa_median - result_full.kappa_median) < 1e-12


# ---------------------------------------------------------------------------
# Real-data test
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_CYCLE_DIR = Path("projects/owasp-llm/cycles/2026")
_CONCORDANCE_JSON = _CYCLE_DIR / "results/concordance.json"
_LAMBDA_NPY = _CYCLE_DIR / "infer/lambda_samples.npy"
_LABELED_JSON = _CYCLE_DIR / "classify/labeled_incidents.json"
_INF_SUMMARY = _CYCLE_DIR / "infer/inference_summary.json"
_DIAG_JSON = _CYCLE_DIR / "calibration/diagnostic.json"
_RESPONDENT_NPY = _FIXTURES_DIR / "respondent_rankings_2026.npy"
_VOTE_IDS_JSON = _FIXTURES_DIR / "vote_entry_ids_2026.json"

_REAL_DATA_AVAILABLE = (
    _CONCORDANCE_JSON.exists()
    and _LAMBDA_NPY.exists()
    and _LABELED_JSON.exists()
    and _INF_SUMMARY.exists()
    and _DIAG_JSON.exists()
    and _RESPONDENT_NPY.exists()
    and _VOTE_IDS_JSON.exists()
)


def _build_strata(
    labeled: list[dict[str, object]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    from collections import defaultdict

    entry_strata_sets: dict[str, set[str]] = defaultdict(set)
    stratum_counts: dict[str, int] = defaultdict(int)
    for item in labeled:
        eid = str(item.get("entry_id", ""))
        stratum = str(item.get("stratum", "default"))
        entry_strata_sets[eid].add(stratum)
        stratum_counts[stratum] += 1
    entry_strata = {e: tuple(sorted(ss)) for e, ss in entry_strata_sets.items()}
    stratum_sizes = {s: max(c, 1) for s, c in stratum_counts.items()}
    return entry_strata, stratum_sizes


@pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="2026 cycle artifacts or fixture rankings not available",
)
def test_real_data_subset_kappa_differs_from_shipped_20_entry_kappa() -> None:
    """Secondary kappa over 17 measurable entries differs from shipped 20-entry kappa.

    This surfaces the STANDING_CAVEAT contradiction documented in measurable_subset.py.
    """
    import numpy.typing as npt

    from engine.vote.bootstrap import bootstrap_vote_ranks

    concordance = json.loads(_CONCORDANCE_JSON.read_text())
    shipped_kappa: float = float(concordance["weighted_kappa_median"])
    shipped_total: int = int(concordance["total_count"])
    shipped_measurable: int = int(concordance["measurable_count"])

    assert shipped_total == 20
    assert shipped_measurable == 17

    lambda_samples: npt.NDArray[np.float64] = np.load(_LAMBDA_NPY)
    with open(_INF_SUMMARY) as f:
        inf = json.load(f)
    inf_entry_ids: tuple[str, ...] = tuple(inf["entry_ids"])

    with open(_LABELED_JSON) as f:
        labeled: list[dict[str, object]] = json.load(f)
    entry_strata, stratum_sizes = _build_strata(labeled)

    # Identify measurable entries from diagnostic
    with open(_DIAG_JSON) as f:
        diag = json.load(f)
    entry_reports: dict[str, dict[str, object]] = diag["entry_reports"]
    measurable_entry_ids: tuple[str, ...] = tuple(
        sorted(
            eid
            for eid, report in entry_reports.items()
            if not str(report.get("reason", "")).startswith("no-data")
        )
    )
    assert len(measurable_entry_ids) == 17, (
        f"expected 17 measurable but got {len(measurable_entry_ids)}: {measurable_entry_ids}"
    )

    respondent_rankings: npt.NDArray[np.float64] = np.load(_RESPONDENT_NPY)
    with open(_VOTE_IDS_JSON) as f:
        vote_entry_ids: tuple[str, ...] = tuple(json.load(f))
    vote_posterior = bootstrap_vote_ranks(
        respondent_rankings, vote_entry_ids, n_bootstrap=5000, seed=20260520
    )

    result = compute_measurable_subset_kappa(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_posterior.rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        measurable_entry_ids=measurable_entry_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )

    assert result.n_measurable == 17
    # Subset kappa must differ from the 20-entry shipped kappa
    assert result.kappa_median != shipped_kappa, (
        f"measurable-subset kappa {result.kappa_median} should differ from "
        f"shipped 20-entry kappa {shipped_kappa}"
    )
    # Sanity: both are plausible kappa values
    assert -1.0 <= result.kappa_median <= 1.0
    assert -1.0 <= shipped_kappa <= 1.0
