"""Tests for the seeded stratified lockbox split (Plan 8e T2)."""
from __future__ import annotations

from engine.classify.bakeoff import (
    LOCKBOX_FRACTION,
    lockbox_cell_sizes,
    lockbox_split,
)


def _truth(n_a: int, n_b: int) -> dict[str, frozenset[str]]:
    t: dict[str, frozenset[str]] = {}
    for i in range(n_a):
        t[f"a{i}"] = frozenset({"A"})
    for i in range(n_b):
        t[f"b{i}"] = frozenset({"B"})
    return t


def test_default_fraction() -> None:
    assert LOCKBOX_FRACTION == 0.3


def test_split_is_deterministic_for_a_seed() -> None:
    truth = _truth(20, 20)
    d1, l1 = lockbox_split(truth, seed=42)
    d2, l2 = lockbox_split(truth, seed=42)
    assert l1 == l2
    assert d1 == d2


def test_split_is_disjoint_and_covers_all() -> None:
    truth = _truth(20, 20)
    dev, lock = lockbox_split(truth, lockbox_fraction=0.3, seed=7)
    assert dev.isdisjoint(lock)
    assert dev | lock == set(truth)


def test_split_is_stratified() -> None:
    truth = _truth(20, 20)
    _, lock = lockbox_split(truth, lockbox_fraction=0.3, seed=7)
    sizes = lockbox_cell_sizes(lock, truth)
    # ~30% of each class's 20 incidents -> 6 each.
    assert sizes["A"] == 6
    assert sizes["B"] == 6


def test_different_seed_changes_membership() -> None:
    truth = _truth(50, 50)
    _, l1 = lockbox_split(truth, seed=1)
    _, l2 = lockbox_split(truth, seed=2)
    assert l1 != l2
