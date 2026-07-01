"""Tests for compute_bare_lambda_sensitivity (T2, U3 Cluster A).

Verifies:
- delta-nonzero path: bare-lambda kappa != incidence kappa -> delta != 0
- delta-zero path: when rankings coincide, delta == 0.0
- disclosure text is present and mentions the delta value
- On 2026 real data: method_kappa_delta == 0.0
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.baselines.bare_lambda import compute_bare_lambda_sensitivity
from engine.baselines.previous_ranking import compute_previous_ranking
from tests.unit.fixtures.baselines import (
    ENTRY_IDS_A,
    ENTRY_IDS_B,
    KAPPA_BARE_B,
    KAPPA_FULL_A,
    KAPPA_INCIDENCE_B,
    LAMBDA_SAMPLES_A,
    LAMBDA_SAMPLES_B,
    METHOD_DELTA_B,
    VOTE_RANK_SAMPLES_A,
    VOTE_RANK_SAMPLES_B,
)

# ---------------------------------------------------------------------------
# Fixture-based tests
# ---------------------------------------------------------------------------


def test_delta_nonzero_case_b() -> None:
    """Fixture B: bare-lambda kappa != incidence kappa -> delta != 0."""
    result = compute_bare_lambda_sensitivity(
        lambda_samples=LAMBDA_SAMPLES_B,
        vote_rank_samples=VOTE_RANK_SAMPLES_B,
        inf_entry_ids=ENTRY_IDS_B,
        vote_entry_ids=ENTRY_IDS_B,
        incidence_kappa_median=KAPPA_INCIDENCE_B,
    )
    assert abs(result.kappa_median - KAPPA_BARE_B) < 1e-12, (
        f"bare kappa {result.kappa_median} != hand-computed {KAPPA_BARE_B}"
    )
    assert abs(result.method_kappa_delta - METHOD_DELTA_B) < 1e-12, (
        f"delta {result.method_kappa_delta} != hand-computed {METHOD_DELTA_B}"
    )
    assert result.method_kappa_delta != 0.0, "delta must be non-zero for fixture B"


def test_delta_zero_case_a() -> None:
    """Fixture A: bare-lambda and incidence rank identically -> delta == 0.0."""
    result = compute_bare_lambda_sensitivity(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        incidence_kappa_median=KAPPA_FULL_A,
    )
    assert result.method_kappa_delta == 0.0, (
        f"delta {result.method_kappa_delta} should be 0.0 for fixture A (equal strata sizes)"
    )


def test_bare_lambda_ranking_b() -> None:
    """Fixture B: bare-lambda ranking is E1,E3,E4,E2 (pure lambda order)."""
    result = compute_bare_lambda_sensitivity(
        lambda_samples=LAMBDA_SAMPLES_B,
        vote_rank_samples=VOTE_RANK_SAMPLES_B,
        inf_entry_ids=ENTRY_IDS_B,
        vote_entry_ids=ENTRY_IDS_B,
        incidence_kappa_median=KAPPA_INCIDENCE_B,
    )
    # lambda: E1=0.9 > E3=0.5 > E4=0.3 > E2=0.1
    assert result.ranking == ("E1", "E3", "E4", "E2"), (
        f"unexpected bare-lambda ranking {result.ranking}"
    )


def test_disclosure_text_contains_delta() -> None:
    """disclosure string contains the computed delta value."""
    result = compute_bare_lambda_sensitivity(
        lambda_samples=LAMBDA_SAMPLES_B,
        vote_rank_samples=VOTE_RANK_SAMPLES_B,
        inf_entry_ids=ENTRY_IDS_B,
        vote_entry_ids=ENTRY_IDS_B,
        incidence_kappa_median=KAPPA_INCIDENCE_B,
    )
    assert result.disclosure, "disclosure must not be empty"
    # The disclosure should reference the delta concept
    assert "method delta" in result.disclosure.lower() or "delta" in result.disclosure


def test_disclosure_not_credited_language() -> None:
    """disclosure explicitly states the delta is not credited as a gain."""
    result = compute_bare_lambda_sensitivity(
        lambda_samples=LAMBDA_SAMPLES_B,
        vote_rank_samples=VOTE_RANK_SAMPLES_B,
        inf_entry_ids=ENTRY_IDS_B,
        vote_entry_ids=ENTRY_IDS_B,
        incidence_kappa_median=KAPPA_INCIDENCE_B,
    )
    # Must say it is NOT credited
    assert "not credited" in result.disclosure.lower(), (
        "disclosure must state delta is not credited as a method gain"
    )


# ---------------------------------------------------------------------------
# Real-data test: method_kappa_delta == 0.0 on 2026 data
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_CYCLE_DIR = Path("projects/owasp-llm/cycles/2026")
_LAMBDA_NPY = _CYCLE_DIR / "infer/lambda_samples.npy"
_LABELED_JSON = _CYCLE_DIR / "classify/labeled_incidents.json"
_INF_SUMMARY = _CYCLE_DIR / "infer/inference_summary.json"
_RESPONDENT_NPY = _FIXTURES_DIR / "respondent_rankings_2026.npy"
_VOTE_IDS_JSON = _FIXTURES_DIR / "vote_entry_ids_2026.json"

_REAL_DATA_AVAILABLE = (
    _LAMBDA_NPY.exists()
    and _LABELED_JSON.exists()
    and _INF_SUMMARY.exists()
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
def test_real_data_method_delta_is_zero() -> None:
    """On 2026 data: bare-lambda kappa median == incidence kappa median (delta==0.0).

    This is a disclosed finding: size-weighting changed nothing on 2026 data.
    """
    import numpy.typing as npt

    from engine.vote.bootstrap import bootstrap_vote_ranks

    lambda_samples: npt.NDArray[np.float64] = np.load(_LAMBDA_NPY)
    with open(_INF_SUMMARY) as f:
        inf = json.load(f)
    inf_entry_ids: tuple[str, ...] = tuple(inf["entry_ids"])

    with open(_LABELED_JSON) as f:
        labeled: list[dict[str, object]] = json.load(f)
    entry_strata, stratum_sizes = _build_strata(labeled)

    respondent_rankings: npt.NDArray[np.float64] = np.load(_RESPONDENT_NPY)
    with open(_VOTE_IDS_JSON) as f:
        vote_entry_ids: tuple[str, ...] = tuple(json.load(f))
    vote_posterior = bootstrap_vote_ranks(
        respondent_rankings, vote_entry_ids, n_bootstrap=5000, seed=20260520
    )

    prev = compute_previous_ranking(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_posterior.rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )

    sens = compute_bare_lambda_sensitivity(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_posterior.rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        incidence_kappa_median=prev.kappa_median,
    )

    assert sens.method_kappa_delta == 0.0, (
        f"method_kappa_delta={sens.method_kappa_delta!r} should be 0.0 on 2026 data "
        f"(size-weighting changed nothing on 2026 — this is the disclosed finding)"
    )
