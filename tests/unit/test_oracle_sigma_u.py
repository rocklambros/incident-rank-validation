"""Tests for the oracle DSL sigma_u surrogate (Plan 8d D3)."""
from __future__ import annotations

import numpy as np

from engine.verify.oracle import oracle_sigma_u_surrogate


def test_sigma_u_zero_when_entries_identical() -> None:
    # All entries share the same log-lambda distribution -> no between-entry SD.
    rng = np.random.default_rng(0)
    base = np.exp(rng.normal(0.0, 0.05, size=(500, 1)))
    lambda_samples = np.repeat(base, 4, axis=1)
    sigma = oracle_sigma_u_surrogate(lambda_samples)
    assert sigma < 0.1


def test_sigma_u_positive_when_entries_spread() -> None:
    # Entries centered at very different log-rates -> positive between-entry SD.
    rng = np.random.default_rng(1)
    cols = []
    for center in (-2.0, -1.0, 0.0, 1.0, 2.0):
        cols.append(np.exp(rng.normal(center, 0.05, size=600)))
    lambda_samples = np.column_stack(cols)
    sigma = oracle_sigma_u_surrogate(lambda_samples)
    assert sigma > 0.5


def test_sigma_u_single_entry_returns_zero() -> None:
    lambda_samples = np.array([[0.1], [0.2], [0.15]])
    assert oracle_sigma_u_surrogate(lambda_samples) == 0.0
