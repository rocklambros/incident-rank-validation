"""Engine-vs-oracle incidence cross-check (T4, U3 Cluster A).

NON-VACUOUS gate: asserts that engine _ranks_from_incidence (using median lambda)
AGREES with oracle_incidence_ranking (two independent implementations — NOT
oracle-vs-itself).  Also proves the gate can FAIL via a deliberately-perturbed
lambda fixture.

The two implementations:
- Engine: engine.decide.concordance._ranks_from_incidence(median_lambda, ...)
  converts to ordinal ranks, then we sort to get best->worst order.
- Oracle: engine.verify.oracle.oracle_incidence_ranking(lambda_samples, ...)
  computes median lambda internally, then sorts by -incidence with tie-breaking.

Independence: different code paths, different internal data structures.
They share the same conceptual definition (lambda*size, descending) but differ
in how they express it: the engine assigns integer ranks via argsort, the oracle
sorts entry names via Python's sorted().
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from engine.decide.concordance import _ranks_from_incidence
from engine.verify.oracle import oracle_incidence_ranking


def _engine_ranking_from_median(
    lambda_samples: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> tuple[str, ...]:
    """Derive best->worst ordering from engine _ranks_from_incidence on median lambda.

    This is the engine path: compute median lambda, call _ranks_from_incidence to
    get ordinal rank per entry, then sort entry names by ascending rank (rank 1 = best).
    """
    median_lam = np.median(lambda_samples, axis=0)
    inf_idx = {e: i for i, e in enumerate(entry_ids)}
    common = list(entry_ids)
    ranks = _ranks_from_incidence(median_lam, inf_idx, common, entry_strata, stratum_sizes)
    # rank 1 = best; sort entries by their rank (ascending)
    ordered = tuple(e for _, e in sorted(zip(ranks.tolist(), common, strict=False)))
    return ordered


# ---------------------------------------------------------------------------
# Fixture-based tests (always run)
# ---------------------------------------------------------------------------

_STRATA_UNIQ: dict[str, tuple[str, ...]] = {
    "E1": ("s",),
    "E2": ("s",),
    "E3": ("s",),
    "E4": ("s",),
}
_SIZES_UNIQ: dict[str, int] = {"s": 100}
_IDS: tuple[str, ...] = ("E1", "E2", "E3", "E4")
# constant lambda: E1=0.9 > E2=0.7 > E3=0.5 > E4=0.3
# incidence = lambda * 100 -> same order
_LAM_CONST = np.tile(np.array([0.9, 0.7, 0.5, 0.3], dtype=np.float64), (20, 1))


def test_engine_agrees_with_oracle_on_fixture() -> None:
    """Engine _ranks_from_incidence ranking == oracle_incidence_ranking (same inputs)."""
    engine_rank = _engine_ranking_from_median(
        _LAM_CONST, _IDS, _STRATA_UNIQ, _SIZES_UNIQ
    )
    oracle_rank = oracle_incidence_ranking(
        _LAM_CONST, _IDS, _STRATA_UNIQ, _SIZES_UNIQ
    )
    assert engine_rank == oracle_rank, (
        f"engine {engine_rank} != oracle {oracle_rank}"
    )


def test_engine_agrees_with_oracle_multistratum_fixture() -> None:
    """Agreement holds for multi-stratum entries."""
    strata: dict[str, tuple[str, ...]] = {
        "E1": ("small", "big"),
        "E2": ("big",),
        "E3": ("small",),
    }
    sizes: dict[str, int] = {"small": 10, "big": 200}
    ids: tuple[str, ...] = ("E1", "E2", "E3")
    # lambda: E1=0.5, E2=0.9, E3=0.3
    # incidence: E1=0.5*(10+200)=105, E2=0.9*200=180, E3=0.3*10=3
    # order: E2(180) > E1(105) > E3(3)
    lam = np.tile(np.array([0.5, 0.9, 0.3], dtype=np.float64), (10, 1))
    engine_rank = _engine_ranking_from_median(lam, ids, strata, sizes)
    oracle_rank = oracle_incidence_ranking(lam, ids, strata, sizes)
    assert engine_rank == oracle_rank == ("E2", "E1", "E3")


def test_gate_can_fail_perturbed_lambda() -> None:
    """Deliberately-perturbed lambda produces DISAGREEMENT (proves the gate can fail).

    Oracle uses the original lambda; engine uses a perturbed lambda where E1 and
    E2 are swapped.  The resulting rankings differ, confirming this is a real gate.
    """
    # Original: E1(0.9) > E2(0.7) > E3(0.5) > E4(0.3)
    lam_original = np.tile(np.array([0.9, 0.7, 0.5, 0.3], dtype=np.float64), (20, 1))
    # Perturbed: E1(0.1) < E2(0.7) < E3(0.5) < E4(0.3) — E1 knocked to last
    lam_perturbed = np.tile(np.array([0.1, 0.7, 0.5, 0.3], dtype=np.float64), (20, 1))

    oracle_rank = oracle_incidence_ranking(
        lam_original, _IDS, _STRATA_UNIQ, _SIZES_UNIQ
    )
    engine_rank = _engine_ranking_from_median(
        lam_perturbed, _IDS, _STRATA_UNIQ, _SIZES_UNIQ
    )
    assert engine_rank != oracle_rank, (
        "perturbed engine lambda must disagree with oracle (gate must be able to fail)"
    )


def test_gate_fails_when_different_inputs_force_disagreement() -> None:
    """Different lambda inputs given to engine vs oracle produce disagreement (gate can fail).

    This test uses DIFFERENT lambda arrays for each implementation to guarantee
    the resulting rankings differ.  It proves the gate is real (not vacuous), but
    does NOT test tie-breaking — the disagreement comes from different inputs, not
    from different tie-breaking strategies with the same input.
    """
    strata: dict[str, tuple[str, ...]] = {"X": ("s",), "Y": ("s",)}
    sizes: dict[str, int] = {"s": 100}
    ids: tuple[str, ...] = ("X", "Y")
    # Different lambda per implementation -> guaranteed disagreement
    lam_engine = np.tile(np.array([0.5, 0.9], dtype=np.float64), (5, 1))  # Y > X
    lam_oracle = np.tile(np.array([0.9, 0.5], dtype=np.float64), (5, 1))  # X > Y

    engine_rank = _engine_ranking_from_median(lam_engine, ids, strata, sizes)
    oracle_rank = oracle_incidence_ranking(lam_oracle, ids, strata, sizes)
    # Y higher in engine, X higher in oracle -> disagree
    assert engine_rank != oracle_rank


def test_same_input_tie_produces_agreement() -> None:
    """Same equal-lambda input to both engine and oracle -> both agree on order.

    This is the true tie-breaking test: when incidence values are identical,
    both implementations must produce the same ordering (alphabetical tiebreak).
    """
    strata: dict[str, tuple[str, ...]] = {"X": ("s",), "Y": ("s",)}
    sizes: dict[str, int] = {"s": 100}
    ids: tuple[str, ...] = ("X", "Y")
    # Equal lambda -> equal incidence (tie); both must agree on tiebreak order
    lam_equal = np.tile(np.array([0.5, 0.5], dtype=np.float64), (5, 1))

    engine_rank = _engine_ranking_from_median(lam_equal, ids, strata, sizes)
    oracle_rank = oracle_incidence_ranking(lam_equal, ids, strata, sizes)
    assert engine_rank == oracle_rank, (
        f"engine {engine_rank} != oracle {oracle_rank} on equal-lambda tie input"
    )


# ---------------------------------------------------------------------------
# Real-data cross-check (reads tracked lambda_samples.npy + labeled_incidents)
# ---------------------------------------------------------------------------

_CYCLE_DIR = Path("projects/owasp-llm/cycles/2026")
_LAMBDA_NPY = _CYCLE_DIR / "infer/lambda_samples.npy"
_LABELED_JSON = _CYCLE_DIR / "classify/labeled_incidents.json"
_INF_SUMMARY = _CYCLE_DIR / "infer/inference_summary.json"

_REAL_DATA_FILES = {
    "lambda_samples.npy": _LAMBDA_NPY,
    "labeled_incidents.json": _LABELED_JSON,
    "inference_summary.json": _INF_SUMMARY,
}


def _build_strata(
    labeled: list[dict[str, object]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    from collections import defaultdict

    es: dict[str, set[str]] = defaultdict(set)
    sc: dict[str, int] = defaultdict(int)
    for item in labeled:
        eid = str(item.get("entry_id", ""))
        stratum = str(item.get("stratum", "default"))
        es[eid].add(stratum)
        sc[stratum] += 1
    entry_strata = {e: tuple(sorted(ss)) for e, ss in es.items()}
    stratum_sizes = {s: max(c, 1) for s, c in sc.items()}
    return entry_strata, stratum_sizes


def test_real_data_engine_oracle_agree() -> None:
    """Engine _ranks_from_incidence ranking AGREES with oracle_incidence_ranking on 2026 data.

    This is the non-vacuous provisional gate: two independent implementations
    must agree before the baseline is considered trustworthy.

    Hard-fail (parity with T11): missing tracked files cause pytest.fail(), not a skip.
    """
    for name, path in _REAL_DATA_FILES.items():
        if not path.exists():
            pytest.fail(
                f"Required tracked file not found: {path}. "
                f"This provisional gate cannot silently vanish — file: {name}",
                pytrace=False,
            )

    lambda_samples: npt.NDArray[np.float64] = np.load(_LAMBDA_NPY)
    with open(_INF_SUMMARY) as f:
        inf = json.load(f)
    inf_entry_ids: tuple[str, ...] = tuple(inf["entry_ids"])

    with open(_LABELED_JSON) as f:
        labeled: list[dict[str, object]] = json.load(f)
    entry_strata, stratum_sizes = _build_strata(labeled)

    engine_rank = _engine_ranking_from_median(
        lambda_samples, inf_entry_ids, entry_strata, stratum_sizes
    )
    oracle_rank = oracle_incidence_ranking(
        lambda_samples, inf_entry_ids, entry_strata, stratum_sizes
    )

    assert engine_rank == oracle_rank, (
        f"PROVISIONAL GATE FAIL: engine and oracle disagree on 2026 incidence ranking.\n"
        f"engine: {engine_rank}\noracle: {oracle_rank}"
    )
    # Confirm 20 entries
    assert len(engine_rank) == 20
