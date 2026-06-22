"""Unit tests for incidence-based ranking in concordance (Task 4, Plan 8a).

Entries may span multiple strata (confirmed in real OWASP-LLM cycle data:
9/20 entries appear in both 'security' and 'ai-harm').  Incidence for entry e
is therefore lambda_e * sum(size_s for s in strata where e is observed).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.decide.concordance import _ranks_from_incidence


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
