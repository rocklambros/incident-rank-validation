"""Tests for Plan 8c render extensions (vote Plackett-Luce / Davidson)."""
from __future__ import annotations

from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult
from engine.report.render import _render_robustness_lines, _render_vote_pl_lines


def _spec_with_pl() -> SpecResult:
    return SpecResult(
        spec_name="kappa_concordance",
        weighted_kappa_median=0.20,
        weighted_kappa_ci=(-0.16, 0.57),
        flags=(),
        extra_rankings={"plackett_luce": ("A", "B", "C", "D", "E", "F")},
    )


def test_robustness_lines_render_extra_rankings() -> None:
    spread = RobustnessSpread(primary=_spec_with_pl(), robustness=())
    text = "".join(_render_robustness_lines(spread))
    assert "plackett_luce" in text
    # top entries are shown
    assert "A" in text and "B" in text


def test_render_vote_pl_lines_shows_diagnostics() -> None:
    vote_pl: dict[str, object] = {
        "model": "davidson_tie_aware",
        "ranking": ["A", "B", "C"],
        "tie_param": 1.37,
        "kendall_tau_vs_meanrank": 0.91,
        "kendall_tau_withties_vs_dropties": 0.86,
        "mean_kendall_tau_vs_point": 0.79,
        "n_respondents": 49,
        "n_bootstrap": 2000,
    }
    text = "".join(_render_vote_pl_lines(vote_pl))
    assert "Vote Aggregation" in text
    assert "Plackett-Luce" in text or "Davidson" in text
    assert "1.37" in text  # tie parameter
    assert "0.91" in text  # kendall tau vs mean-rank
    assert "0.79" in text  # bootstrap stability


def test_render_vote_pl_lines_none_is_empty() -> None:
    assert _render_vote_pl_lines(None) == []
