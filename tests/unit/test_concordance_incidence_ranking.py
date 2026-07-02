"""Unit tests for incidence-based ranking in concordance (Task 4, Plan 8a).

Entries may span multiple strata (confirmed in real OWASP-LLM cycle data:
9/20 entries appear in both 'security' and 'ai-harm').  Incidence for entry e
is therefore lambda_e * sum(size_s for s in strata where e is observed).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.decide.concordance import _ranks_from_incidence, compute_concordance
from engine.model.inference import InferenceResult
from engine.vote.bootstrap import VoteRankPosterior


def test_incidence_ranking_uses_lambda_times_size() -> None:
    # Entry A has higher lambda but lives in a tiny stratum; entry B has lower
    # lambda in a huge stratum. By incidence (lambda*size), B outranks A.
    lam = np.array([0.9, 0.4])  # A, B
    idx = {"A": 0, "B": 1}
    common = ["A", "B"]
    entry_strata: dict[str, tuple[str, ...]] = {"A": ("small",), "B": ("big",)}
    sizes = {"small": 10, "big": 1000}
    ranks = _ranks_from_incidence(lam, idx, common, entry_strata, sizes)
    # incidence: A=9, B=400 -> B is rank 1, A is rank 2
    assert ranks[common.index("B")] == 1.0
    assert ranks[common.index("A")] == 2.0


def test_incidence_ranking_multistratum_entry() -> None:
    """Entry spanning two strata uses sum of both stratum sizes."""
    # A: lambda=0.1, strata=["small","tiny"] -> incidence = 0.1*(10+5) = 1.5
    # B: lambda=0.05, strata=["big"]         -> incidence = 0.05*1000  = 50.0
    # C: lambda=0.5,  strata=["small"]       -> incidence = 0.5*10     = 5.0
    # Expected rank order: B(50)=1, C(5)=2, A(1.5)=3
    lam = np.array([0.1, 0.05, 0.5])
    idx = {"A": 0, "B": 1, "C": 2}
    common = ["A", "B", "C"]
    entry_strata: dict[str, tuple[str, ...]] = {
        "A": ("small", "tiny"),
        "B": ("big",),
        "C": ("small",),
    }
    sizes = {"small": 10, "tiny": 5, "big": 1000}
    ranks = _ranks_from_incidence(lam, idx, common, entry_strata, sizes)
    assert ranks[common.index("B")] == 1.0
    assert ranks[common.index("C")] == 2.0
    assert ranks[common.index("A")] == 3.0


def test_incidence_ranking_single_entry() -> None:
    """Single entry always gets rank 1."""
    lam = np.array([0.3])
    idx = {"X": 0}
    common = ["X"]
    entry_strata: dict[str, tuple[str, ...]] = {"X": ("only",)}
    sizes = {"only": 100}
    ranks = _ranks_from_incidence(lam, idx, common, entry_strata, sizes)
    assert ranks[0] == 1.0


def test_incidence_ranking_tie_handling() -> None:
    """Tied incidence values produce valid (though non-deterministic) ranks."""
    lam = np.array([0.5, 0.5])
    idx = {"A": 0, "B": 1}
    common = ["A", "B"]
    entry_strata: dict[str, tuple[str, ...]] = {"A": ("s",), "B": ("s",)}
    sizes = {"s": 100}
    ranks = _ranks_from_incidence(lam, idx, common, entry_strata, sizes)
    assert set(ranks.tolist()) == {1.0, 2.0}


# ---------------------------------------------------------------------------
# Fail-loud precondition tests (Plan 8a Finding 2 / Plan 8e contract)
# ---------------------------------------------------------------------------


def _make_minimal_inference(entry_ids: tuple[str, ...], n_samples: int = 10) -> InferenceResult:
    rng = np.random.default_rng(0)
    n = len(entry_ids)
    return InferenceResult(
        lambda_samples=rng.exponential(scale=1.0, size=(n_samples, n)),
        entry_ids=entry_ids,
        r_hat={f"lambda[{i}]": 1.0 for i in range(n)},
        ess={f"lambda[{i}]": float(n_samples) for i in range(n)},
        divergences=0,
        num_warmup=50,
        num_samples=n_samples,
    )


def _make_minimal_vote_posterior(
    entries: tuple[str, ...], n_bootstrap: int = 10
) -> VoteRankPosterior:
    rng = np.random.default_rng(1)
    n = len(entries)
    rank_samples = np.zeros((n_bootstrap, n), dtype=np.float64)
    for b in range(n_bootstrap):
        order = rng.permutation(n)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        rank_samples[b] = ranks
    medians = {entries[i]: float(np.median(rank_samples[:, i])) for i in range(n)}
    return VoteRankPosterior(
        entries=entries,
        rank_samples=rank_samples,
        median_ranks=medians,
        n_respondents=5,
        n_bootstrap=n_bootstrap,
    )


def test_compute_concordance_raises_named_valueerror_on_missing_entry_strata() -> None:
    """compute_concordance must raise ValueError (not KeyError) when a measurable
    entry is absent from entry_strata — the fail-loud posture required by Plan 8a
    and the contract Plan 8e must satisfy."""
    entries = ("E1", "E2", "E3")
    inference = _make_minimal_inference(entries, n_samples=10)
    vote = _make_minimal_vote_posterior(entries, n_bootstrap=10)

    # entry_strata covers only E1 and E2; E3 is deliberately missing
    entry_strata: dict[str, tuple[str, ...]] = {
        "E1": ("stratum_a",),
        "E2": ("stratum_a",),
        # E3 absent
    }
    stratum_sizes = {"stratum_a": 100}

    with pytest.raises(ValueError, match="E3") as exc_info:
        compute_concordance(
            inference_result=inference,
            vote_posterior=vote,
            tier_boundaries=(1, 2),
            flag_threshold_tau=0.5,
            measurable_count=3,
            total_count=3,
            meaningful_kappa_n=2,
            measurability_minimum=2,
            entry_strata=entry_strata,
            stratum_sizes=stratum_sizes,
        )
    # Must name the missing entry and NOT be a bare KeyError
    assert "entry_strata" in str(exc_info.value)
