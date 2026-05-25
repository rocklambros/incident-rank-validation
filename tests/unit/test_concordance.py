"""Unit tests for engine.decide.concordance — transparency-first concordance."""

from __future__ import annotations

import json

import numpy as np

from engine.decide.concordance import ConcordanceResult, STANDING_CAVEAT, compute_concordance
from engine.decide.robustness_multiplicity import FlagDirection, FlagFinding
from engine.model.inference import InferenceResult
from engine.vote.bootstrap import VoteRankPosterior

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inference(
    entry_ids: tuple[str, ...],
    n_samples: int = 100,
    seed: int = 42,
) -> InferenceResult:
    """Synthetic InferenceResult with random lambda samples."""
    rng = np.random.default_rng(seed)
    n = len(entry_ids)
    # Lambda values: higher = more prevalent
    lam = rng.exponential(scale=1.0, size=(n_samples, n))
    return InferenceResult(
        lambda_samples=lam,
        entry_ids=entry_ids,
        r_hat={f"lambda[{i}]": 1.0 for i in range(n)},
        ess={f"lambda[{i}]": float(n_samples) for i in range(n)},
        divergences=0,
        num_warmup=100,
        num_samples=n_samples,
    )


def _make_vote_posterior(
    entries: tuple[str, ...],
    n_bootstrap: int = 100,
    seed: int = 99,
) -> VoteRankPosterior:
    """Synthetic VoteRankPosterior with random rank samples."""
    rng = np.random.default_rng(seed)
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
        n_respondents=50,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeConcordance:
    def test_below_meaningful_kappa_n_returns_none(self) -> None:
        """When measurable_count < meaningful_kappa_n, kappa should be None."""
        entries = ("A", "B")
        inf = _make_inference(entries)
        vote = _make_vote_posterior(entries)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(3,),
            flag_threshold_tau=0.5,
            measurable_count=2,
            total_count=10,
            meaningful_kappa_n=5,  # 2 < 5
            measurability_minimum=3,
        )

        assert result.weighted_kappa_median is None
        assert result.weighted_kappa_ci is None
        assert result.standing_caveat == STANDING_CAVEAT

    def test_normal_case_returns_kappa(self) -> None:
        """With enough entries, kappa should be a finite number."""
        entries = tuple(f"E{i}" for i in range(10))
        inf = _make_inference(entries, n_samples=200)
        vote = _make_vote_posterior(entries, n_bootstrap=200)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(3, 7),
            flag_threshold_tau=0.5,
            measurable_count=10,
            total_count=15,
            meaningful_kappa_n=5,
            measurability_minimum=5,
        )

        assert result.weighted_kappa_median is not None
        assert -1.0 <= result.weighted_kappa_median <= 1.0
        assert result.weighted_kappa_ci is not None
        assert result.weighted_kappa_ci[0] <= result.weighted_kappa_ci[1]

    def test_coverage_ratio(self) -> None:
        """Coverage ratio should be measurable / total."""
        entries = ("A", "B", "C", "D", "E")
        inf = _make_inference(entries, n_samples=50)
        vote = _make_vote_posterior(entries, n_bootstrap=50)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(2,),
            flag_threshold_tau=0.5,
            measurable_count=5,
            total_count=20,
            meaningful_kappa_n=3,
            measurability_minimum=3,
        )

        assert abs(result.coverage_ratio - 0.25) < 1e-9

    def test_below_prereg_minimum(self) -> None:
        """below_prereg_minimum should be True when measurable < minimum."""
        entries = ("A", "B", "C", "D", "E")
        inf = _make_inference(entries, n_samples=50)
        vote = _make_vote_posterior(entries, n_bootstrap=50)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(2,),
            flag_threshold_tau=0.5,
            measurable_count=5,
            total_count=20,
            meaningful_kappa_n=3,
            measurability_minimum=10,  # 5 < 10
        )

        assert result.below_prereg_minimum is True

    def test_no_common_entries_returns_none(self) -> None:
        """When inference and vote have no common entries, kappa is None."""
        inf = _make_inference(("A", "B", "C", "D", "E"), n_samples=50)
        vote = _make_vote_posterior(("X", "Y", "Z", "W", "V"), n_bootstrap=50)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(2,),
            flag_threshold_tau=0.5,
            measurable_count=5,
            total_count=10,
            meaningful_kappa_n=3,
            measurability_minimum=3,
        )

        assert result.weighted_kappa_median is None

    def test_standing_caveat_always_present(self) -> None:
        """Every ConcordanceResult must carry the standing caveat."""
        entries = tuple(f"E{i}" for i in range(6))
        inf = _make_inference(entries, n_samples=50)
        vote = _make_vote_posterior(entries, n_bootstrap=50)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(2, 4),
            flag_threshold_tau=0.5,
            measurable_count=6,
            total_count=6,
            meaningful_kappa_n=3,
            measurability_minimum=3,
        )

        assert "Internal triangulation" in result.standing_caveat


class TestCiMethodField:
    def test_ci_method_default_value(self) -> None:
        """ConcordanceResult should have ci_method with correct default."""
        result = ConcordanceResult(
            weighted_kappa_median=0.20,
            weighted_kappa_ci=(-0.16, 0.57),
            measurable_count=17,
            total_count=20,
            coverage_ratio=0.85,
            below_prereg_minimum=False,
            meaningful_kappa_n=5,
            flags=(),
            standing_caveat=STANDING_CAVEAT,
        )
        assert result.ci_method == "paired_draw_percentile"

    def test_ci_method_in_na_result(self) -> None:
        """N/A results should also carry the ci_method default."""
        entries = ("A", "B")
        inf = _make_inference(entries)
        vote = _make_vote_posterior(entries)
        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(3,),
            flag_threshold_tau=0.5,
            measurable_count=2,
            total_count=10,
            meaningful_kappa_n=5,
            measurability_minimum=3,
        )
        assert result.ci_method == "paired_draw_percentile"

    def test_ci_method_in_normal_result(self) -> None:
        """Normal results should carry the ci_method default."""
        entries = tuple(f"E{i}" for i in range(10))
        inf = _make_inference(entries, n_samples=200)
        vote = _make_vote_posterior(entries, n_bootstrap=200)
        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(3, 7),
            flag_threshold_tau=0.5,
            measurable_count=10,
            total_count=15,
            meaningful_kappa_n=5,
            measurability_minimum=5,
        )
        assert result.ci_method == "paired_draw_percentile"


class TestCiMethodSerialization:
    def test_ci_method_in_serialized_dict(self) -> None:
        """ci_method should appear in the concordance JSON output."""
        result = ConcordanceResult(
            weighted_kappa_median=0.20,
            weighted_kappa_ci=(-0.16, 0.57),
            measurable_count=17,
            total_count=20,
            coverage_ratio=0.85,
            below_prereg_minimum=False,
            meaningful_kappa_n=5,
            flags=(),
            standing_caveat=STANDING_CAVEAT,
        )
        conc_dict = {
            "weighted_kappa_median": result.weighted_kappa_median,
            "weighted_kappa_ci": list(result.weighted_kappa_ci) if result.weighted_kappa_ci else None,
            "measurable_count": result.measurable_count,
            "total_count": result.total_count,
            "coverage_ratio": result.coverage_ratio,
            "below_prereg_minimum": result.below_prereg_minimum,
            "ci_method": result.ci_method,
            "flags": [],
        }
        serialized = json.dumps(conc_dict)
        data = json.loads(serialized)
        assert data["ci_method"] == "paired_draw_percentile"

    def test_round_trip_preserves_ci_method(self) -> None:
        """Serialize then deserialize: ci_method must survive the round trip."""
        original = ConcordanceResult(
            weighted_kappa_median=0.20,
            weighted_kappa_ci=(-0.16, 0.57),
            measurable_count=17,
            total_count=20,
            coverage_ratio=0.85,
            below_prereg_minimum=False,
            meaningful_kappa_n=5,
            flags=(
                FlagFinding(entry_id="LLM01", probability=0.88, direction=FlagDirection.VOTE_OVER_RANKS),
            ),
            standing_caveat="test caveat",
        )

        conc_dict = {
            "weighted_kappa_median": original.weighted_kappa_median,
            "weighted_kappa_ci": list(original.weighted_kappa_ci) if original.weighted_kappa_ci else None,
            "measurable_count": original.measurable_count,
            "total_count": original.total_count,
            "coverage_ratio": original.coverage_ratio,
            "below_prereg_minimum": original.below_prereg_minimum,
            "ci_method": original.ci_method,
            "flags": [
                {"entry_id": f.entry_id, "probability": f.probability, "direction": f.direction.value}
                for f in original.flags
            ],
        }
        json_str = json.dumps(conc_dict, indent=2)
        data = json.loads(json_str)

        flags_raw = data.get("flags", [])
        flags = tuple(
            FlagFinding(
                entry_id=f["entry_id"],
                probability=f["probability"],
                direction=FlagDirection(f["direction"]),
            )
            for f in flags_raw
        )
        reconstructed = ConcordanceResult(
            weighted_kappa_median=data.get("weighted_kappa_median"),
            weighted_kappa_ci=tuple(data["weighted_kappa_ci"]) if data.get("weighted_kappa_ci") else None,
            measurable_count=data["measurable_count"],
            total_count=data["total_count"],
            coverage_ratio=data["coverage_ratio"],
            below_prereg_minimum=data.get("below_prereg_minimum", False),
            meaningful_kappa_n=5,
            flags=flags,
            standing_caveat=STANDING_CAVEAT,
            ci_method=data.get("ci_method", "paired_draw_percentile"),
        )

        assert reconstructed.ci_method == original.ci_method
        assert reconstructed.weighted_kappa_median == original.weighted_kappa_median
        assert reconstructed.flags[0].entry_id == "LLM01"
