"""Tests for build_vote_pl_summary (Plan 8c Task 3)."""
from __future__ import annotations

import numpy as np

from engine.cli.pipeline import build_vote_pl_summary


def test_summary_has_required_keys_and_valid_ranking() -> None:
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.5, 1.5, 3.0],
        ]
    )
    entry_ids = ("A", "B", "C")
    summary = build_vote_pl_summary(
        rankings,
        entry_ids,
        mean_rank_ranking=("A", "B", "C"),
        seed=42,
        n_bootstrap=20,
    )
    required = {
        "model",
        "ridge",
        "n_bootstrap",
        "seed",
        "n_respondents",
        "entries",
        "worths",
        "tie_param",
        "ranking",
        "ranking_drop_ties",
        "bootstrap_median_ranks",
        "bootstrap_top5_frequency",
        "mean_kendall_tau_vs_point",
        "kendall_tau_vs_meanrank",
        "kendall_tau_withties_vs_dropties",
    }
    assert required.issubset(summary.keys())
    assert summary["model"] == "davidson_tie_aware"
    assert summary["n_respondents"] == 3
    assert summary["seed"] == 42
    # ranking is a permutation of the entries
    assert set(summary["ranking"]) == set(entry_ids)  # type: ignore[call-overload]
    # clean strict data: Davidson ranking matches the mean-rank ranking
    assert summary["kendall_tau_vs_meanrank"] == 1.0


def test_summary_is_json_serializable() -> None:
    import json

    rankings = np.tile(np.array([1.0, 2.0]), (5, 1))
    summary = build_vote_pl_summary(
        rankings, ("A", "B"), mean_rank_ranking=("A", "B"), seed=1, n_bootstrap=10
    )
    # Must round-trip through JSON without error.
    restored = json.loads(json.dumps(summary))
    assert restored["entries"] == ["A", "B"]


def test_summary_surfaces_convergence() -> None:
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (6, 1))
    summary = build_vote_pl_summary(
        rankings,
        ("A", "B", "C"),
        mean_rank_ranking=("A", "B", "C"),
        seed=42,
        n_bootstrap=10,
    )
    assert summary["converged"] is True
    assert "n_nonconverged_bootstrap" in summary
    assert isinstance(summary["n_nonconverged_bootstrap"], int)
