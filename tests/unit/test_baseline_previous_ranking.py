"""Tests for compute_previous_ranking (T1, U3 Cluster A).

Two test groups:
1. Fixture-based: verify hand-computed kappa on synthetic deterministic data.
2. Real-data byte-pin: assert kappa/CI byte-equal to concordance.json (n_common=20,
   tiers=(6,12)), loading respondent_rankings_2026.npy from fixtures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.baselines.previous_ranking import compute_previous_ranking
from tests.unit.fixtures.baselines import (
    ENTRY_IDS_A,
    ENTRY_STRATA_A,
    ENTRY_STRATA_B,
    KAPPA_FULL_A,
    LAMBDA_SAMPLES_A,
    LAMBDA_SAMPLES_B,
    STRATUM_SIZES_A,
    STRATUM_SIZES_B,
    VOTE_RANK_SAMPLES_A,
    VOTE_RANK_SAMPLES_B,
)

# ---------------------------------------------------------------------------
# Fixture-based tests (always run; no cycle files required)
# ---------------------------------------------------------------------------

_ENTRY_IDS_A_TUPLE: tuple[str, ...] = ENTRY_IDS_A


def test_fixture_a_kappa_matches_hand_computed() -> None:
    """Full 4-entry set (delta-0): kappa median == 7/11 (hand-computed)."""
    result = compute_previous_ranking(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    assert result.n_common == 4
    assert result.tier_boundaries == (1, 2), (
        f"expected (1,2) for n=4 but got {result.tier_boundaries}"
    )
    assert abs(result.kappa_median - KAPPA_FULL_A) < 1e-12, (
        f"kappa {result.kappa_median} != hand-computed 7/11={KAPPA_FULL_A}"
    )
    # CI: all draws same -> point mass -> lo == median == hi
    assert abs(result.kappa_ci_lo - KAPPA_FULL_A) < 1e-12
    assert abs(result.kappa_ci_hi - KAPPA_FULL_A) < 1e-12


def test_fixture_a_ranking_order() -> None:
    """Incidence ranking matches lambda order when all strata sizes equal."""
    result = compute_previous_ranking(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    # E1(0.9) > E2(0.7) > E3(0.5) > E4(0.3) in both lambda and incidence
    assert result.ranking == ("E1", "E2", "E3", "E4"), (
        f"unexpected ranking {result.ranking}"
    )


def test_fixture_b_kappa_one_when_vote_matches_incidence() -> None:
    """delta-nonzero fixture: incidence kappa == 1.0 (vote matches incidence)."""
    result = compute_previous_ranking(
        lambda_samples=LAMBDA_SAMPLES_B,
        vote_rank_samples=VOTE_RANK_SAMPLES_B,
        inf_entry_ids=ENTRY_IDS_A,  # same entry IDs
        vote_entry_ids=ENTRY_IDS_A,
        entry_strata=ENTRY_STRATA_B,
        stratum_sizes=STRATUM_SIZES_B,
    )
    assert result.n_common == 4
    assert abs(result.kappa_median - 1.0) < 1e-12, (
        f"expected kappa=1.0 for perfect agreement, got {result.kappa_median}"
    )


def test_fixture_b_ranking_follows_incidence_not_bare_lambda() -> None:
    """delta-nonzero: incidence ranking is E2,E1,E3,E4 (not bare-lambda order E1,E3,E4,E2)."""
    result = compute_previous_ranking(
        lambda_samples=LAMBDA_SAMPLES_B,
        vote_rank_samples=VOTE_RANK_SAMPLES_B,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        entry_strata=ENTRY_STRATA_B,
        stratum_sizes=STRATUM_SIZES_B,
    )
    # incidence: E2=20, E1=13.5, E3=5, E4=1.5 -> E2 > E1 > E3 > E4
    assert result.ranking == ("E2", "E1", "E3", "E4"), (
        f"unexpected ranking {result.ranking}"
    )


def test_n_common_uses_intersection() -> None:
    """Only entries in BOTH inf and vote appear in common (n_common < n_inf)."""
    inf_ids = ("E1", "E2", "E3", "E4", "E5")  # E5 not in vote
    vote_ids = ("E1", "E2", "E3", "E4")
    strata: dict[str, tuple[str, ...]] = {e: ("s",) for e in inf_ids}
    lam = np.tile(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), (5, 1))
    vote = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (5, 1))
    result = compute_previous_ranking(
        lambda_samples=lam,
        vote_rank_samples=vote,
        inf_entry_ids=inf_ids,
        vote_entry_ids=vote_ids,
        entry_strata=strata,
        stratum_sizes={"s": 100},
    )
    assert result.n_common == 4
    assert "E5" not in result.ranking


# ---------------------------------------------------------------------------
# Real-data byte-pin test (reads tracked fixtures, no xlsx required)
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_CYCLE_DIR = Path("projects/owasp-llm/cycles/2026")
_CONCORDANCE_JSON = _CYCLE_DIR / "results/concordance.json"
_LAMBDA_NPY = _CYCLE_DIR / "infer/lambda_samples.npy"
_LABELED_JSON = _CYCLE_DIR / "classify/labeled_incidents.json"
_INF_SUMMARY = _CYCLE_DIR / "infer/inference_summary.json"
_RESPONDENT_NPY = _FIXTURES_DIR / "respondent_rankings_2026.npy"
_VOTE_IDS_JSON = _FIXTURES_DIR / "vote_entry_ids_2026.json"

_REAL_DATA_AVAILABLE = (
    _CONCORDANCE_JSON.exists()
    and _LAMBDA_NPY.exists()
    and _LABELED_JSON.exists()
    and _INF_SUMMARY.exists()
    and _RESPONDENT_NPY.exists()
    and _VOTE_IDS_JSON.exists()
)


def _build_strata_from_labeled(
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
def test_real_data_kappa_byte_equal_concordance_json() -> None:
    """Byte-pin: compute_previous_ranking reproduces concordance.json exactly.

    n_common == 20, tier_boundaries == (6, 12), kappa_median and kappa_ci
    must match cycles/2026/results/concordance.json within atol=1e-9.
    Asserts against the FILE — never a hand-typed constant.
    """
    from engine.vote.bootstrap import bootstrap_vote_ranks

    # Load shipped target
    concordance = json.loads(_CONCORDANCE_JSON.read_text())
    target_kappa: float = float(concordance["weighted_kappa_median"])
    target_ci_lo: float = float(concordance["weighted_kappa_ci"][0])
    target_ci_hi: float = float(concordance["weighted_kappa_ci"][1])
    target_n: int = int(concordance["total_count"])

    # Load cycle artifacts
    lambda_samples: npt.NDArray[np.float64] = np.load(_LAMBDA_NPY)
    with open(_INF_SUMMARY) as f:
        inf = json.load(f)
    inf_entry_ids: tuple[str, ...] = tuple(inf["entry_ids"])

    with open(_LABELED_JSON) as f:
        labeled: list[dict[str, object]] = json.load(f)
    entry_strata, stratum_sizes = _build_strata_from_labeled(labeled)

    # Bootstrap vote ranks from committed respondent matrix (non-circular)
    respondent_rankings: npt.NDArray[np.float64] = np.load(_RESPONDENT_NPY)
    with open(_VOTE_IDS_JSON) as f:
        vote_entry_ids: tuple[str, ...] = tuple(json.load(f))
    vote_posterior = bootstrap_vote_ranks(
        respondent_rankings, vote_entry_ids, n_bootstrap=5000, seed=20260520
    )

    result = compute_previous_ranking(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_posterior.rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )

    assert result.n_common == target_n, (
        f"n_common={result.n_common} != concordance.json total_count={target_n}"
    )
    assert result.tier_boundaries == (6, 12), (
        f"tier_boundaries={result.tier_boundaries} != (6,12) for n=20"
    )

    # Byte-pin: assert against FILE values, not hand-typed constants
    assert abs(result.kappa_median - target_kappa) <= 1e-9, (
        f"kappa_median={result.kappa_median!r} differs from "
        f"concordance.json {target_kappa!r} by "
        f"{abs(result.kappa_median - target_kappa):.2e}"
    )
    assert abs(result.kappa_ci_lo - target_ci_lo) <= 1e-9, (
        f"kappa_ci_lo={result.kappa_ci_lo!r} differs from "
        f"concordance.json {target_ci_lo!r}"
    )
    assert abs(result.kappa_ci_hi - target_ci_hi) <= 1e-9, (
        f"kappa_ci_hi={result.kappa_ci_hi!r} differs from "
        f"concordance.json {target_ci_hi!r}"
    )


# mypy annotation for the npt import inside the test
import numpy.typing as npt  # noqa: E402 — must be after the guard
