"""Transparency-first concordance (HANDOFF §5.5).

Concordance ties together kappa, per-entry flags, and the measurability gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from engine.decide.kappa import quadratic_weighted_kappa
from engine.decide.robustness_multiplicity import FlagDirection, FlagFinding
from engine.model.inference import InferenceResult
from engine.vote.bootstrap import VoteRankPosterior

__all__ = [
    "ConcordanceResult",
    "STANDING_CAVEAT",
    "compute_concordance",
    "format_rank_comparison_report",
    "_tier_agreement_label",
    "_ranks_from_incidence",
]


@dataclass(frozen=True, slots=True)
class ConcordanceResult:
    """Full concordance output including kappa, CI, flags, and caveat."""

    weighted_kappa_median: float | None
    weighted_kappa_ci: tuple[float, float] | None
    measurable_count: int
    total_count: int
    coverage_ratio: float
    below_prereg_minimum: bool
    meaningful_kappa_n: int
    flags: tuple[FlagFinding, ...]
    standing_caveat: str
    ci_method: str = "paired_draw_percentile"
    entry_comparisons: tuple[dict[str, object], ...] | None = None


STANDING_CAVEAT = (
    "Internal triangulation against a contaminated index, not validation against "
    "reality. This concordance is computed over the measurable subset only; entries "
    "the corpus frame cannot observe or the classifier cannot recover are listed "
    "separately in the measurability map."
)


def _ranks_from_lambda(
    lam_draw: npt.NDArray[np.float64],
    idx_map: dict[str, int],
    common: list[str],
) -> npt.NDArray[np.float64]:
    """Convert a lambda draw to ordinal ranks for the common entries."""
    vals = np.array([lam_draw[idx_map[e]] for e in common])
    order = np.argsort(-vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(common) + 1, dtype=np.float64)
    return ranks


def _ranks_from_incidence(
    lam_draw: npt.NDArray[np.float64],
    idx_map: dict[str, int],
    common: list[str],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> npt.NDArray[np.float64]:
    """Rank entries by incidence (lambda_e * sum of stratum sizes), descending.

    Entries may span multiple strata (confirmed in real OWASP-LLM data: 9/20
    entries appear in both 'security' and 'ai-harm').  Incidence is therefore
    lambda_e * Σ_s size_s for all strata s where entry e was observed.  This
    sum correctly weights multi-stratum entries by their total exposure rather
    than arbitrarily picking one stratum.
    """
    vals = np.array(
        [
            lam_draw[idx_map[e]] * float(sum(stratum_sizes[s] for s in entry_strata[e]))
            for e in common
        ]
    )
    order = np.argsort(-vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(common) + 1, dtype=np.float64)
    return ranks


def _tier_agreement_label(tier_a: int, tier_b: int) -> str:
    """Return a human-readable tier-agreement label."""
    diff = abs(tier_a - tier_b)
    if diff == 0:
        return "same"
    elif diff == 1:
        return "±1"
    else:
        return "±2+"


def _na_result(
    measurable_count: int,
    total_count: int,
    coverage: float,
    below_min: bool,
    meaningful_kappa_n: int,
) -> ConcordanceResult:
    return ConcordanceResult(
        weighted_kappa_median=None,
        weighted_kappa_ci=None,
        measurable_count=measurable_count,
        total_count=total_count,
        coverage_ratio=coverage,
        below_prereg_minimum=below_min,
        meaningful_kappa_n=meaningful_kappa_n,
        flags=(),
        standing_caveat=STANDING_CAVEAT,
    )


def compute_concordance(
    inference_result: InferenceResult,
    vote_posterior: VoteRankPosterior,
    tier_boundaries: tuple[int, ...],
    flag_threshold_tau: float,
    measurable_count: int,
    total_count: int,
    meaningful_kappa_n: int,
    measurability_minimum: int,
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> ConcordanceResult:
    """Transparency-first concordance (HANDOFF §5.5).

    Ranks by incidence (lambda_e * Σ stratum_sizes) rather than bare lambda,
    which is the decision-relevant quantity (SD8, Plan 8a Task 4).  Entries
    spanning multiple strata (9/20 in the 2026 OWASP-LLM cycle) use the sum
    of all strata sizes they were observed in.
    """
    below_min = measurable_count < measurability_minimum
    coverage = measurable_count / total_count if total_count > 0 else 0.0

    # Precondition: every measurable entry must appear in entry_strata (Plan 8e
    # must build entry_strata to cover ALL measurable entries).
    vote_set_pre = set(vote_posterior.entries)
    common_pre = [e for e in inference_result.entry_ids if e in vote_set_pre]
    missing_entries = sorted(e for e in common_pre if e not in entry_strata)
    if missing_entries:
        raise ValueError(
            f"compute_concordance: measurable entries missing from entry_strata: "
            f"{missing_entries}; entry_strata must cover all measurable entries "
            f"(see Plan 8e)"
        )
    # Precondition: every stratum referenced by common entries must be in stratum_sizes.
    missing_strata = sorted(
        s
        for e in common_pre
        for s in entry_strata[e]
        if s not in stratum_sizes
    )
    if missing_strata:
        raise ValueError(
            f"compute_concordance: strata referenced by entries but absent from "
            f"stratum_sizes: {missing_strata}; stratum_sizes must cover all strata "
            f"referenced by entry_strata (see Plan 8e)"
        )

    # If below meaningful kappa N, report N/A
    if measurable_count < meaningful_kappa_n:
        return _na_result(measurable_count, total_count, coverage, below_min, meaningful_kappa_n)

    # Map entries: only those in both inference and vote
    vote_set = set(vote_posterior.entries)
    common = [e for e in inference_result.entry_ids if e in vote_set]
    if len(common) < meaningful_kappa_n:
        return _na_result(measurable_count, total_count, coverage, below_min, meaningful_kappa_n)

    inf_idx = {e: i for i, e in enumerate(inference_result.entry_ids)}
    vote_idx = {e: i for i, e in enumerate(vote_posterior.entries)}

    # Recompute tier boundaries from common set size, not caller's full count
    n_common = len(common)
    if n_common <= 3:
        tier_boundaries = tuple(range(1, n_common))
    else:
        third = n_common // 3
        tier_boundaries = (third, 2 * third)

    # Compute kappa over bootstrap x posterior draws
    n_draws = min(len(inference_result.lambda_samples), len(vote_posterior.rank_samples))
    kappas: list[float] = []

    for s in range(n_draws):
        inc_ranks = _ranks_from_incidence(
            inference_result.lambda_samples[s], inf_idx, common, entry_strata, stratum_sizes
        )
        vote_draw = vote_posterior.rank_samples[s]
        vote_ranks = np.array([vote_draw[vote_idx[e]] for e in common])

        k = quadratic_weighted_kappa(inc_ranks, vote_ranks, tier_boundaries)
        if not np.isnan(k):
            kappas.append(k)

    if not kappas:
        return _na_result(measurable_count, total_count, coverage, below_min, meaningful_kappa_n)

    kappa_arr = np.array(kappas)
    median_k = float(np.median(kappa_arr))
    ci = (float(np.percentile(kappa_arr, 2.5)), float(np.percentile(kappa_arr, 97.5)))

    # Per-entry flags
    flags: list[FlagFinding] = []
    for e in common:
        mismatch_count = 0
        for s in range(n_draws):
            inc_ranks = _ranks_from_incidence(
                inference_result.lambda_samples[s], inf_idx, common, entry_strata, stratum_sizes
            )
            vote_draw = vote_posterior.rank_samples[s]
            vote_ranks = np.array([vote_draw[vote_idx[c]] for c in common])

            e_pos = common.index(e)
            inc_tier = sum(1 for b in tier_boundaries if inc_ranks[e_pos] > b)
            vote_tier = sum(1 for b in tier_boundaries if vote_ranks[e_pos] > b)
            if inc_tier != vote_tier:
                mismatch_count += 1

        prob = mismatch_count / n_draws
        if prob > flag_threshold_tau:
            # Determine direction from median ranks
            median_inc_samples = []
            for s in range(min(n_draws, 100)):
                r = _ranks_from_incidence(
                    inference_result.lambda_samples[s], inf_idx, common, entry_strata, stratum_sizes
                )
                median_inc_samples.append(r[common.index(e)])
            median_inc_rank = float(np.median(median_inc_samples))
            median_vote_rank = float(np.median(vote_posterior.rank_samples[:, vote_idx[e]]))

            if median_vote_rank < median_inc_rank - 0.5:
                direction = FlagDirection.VOTE_OVER_RANKS
            elif median_vote_rank > median_inc_rank + 0.5:
                direction = FlagDirection.VOTE_UNDER_RANKS
            else:
                direction = FlagDirection.INDETERMINATE

            flags.append(FlagFinding(entry_id=e, direction=direction, probability=prob))

    # Per-entry rank comparison
    comparisons: list[dict[str, object]] = []
    for e in common:
        e_pos = common.index(e)

        inc_rank_samples = []
        vote_rank_samples = []
        for s in range(n_draws):
            inc_r = _ranks_from_incidence(
                inference_result.lambda_samples[s], inf_idx, common, entry_strata, stratum_sizes
            )
            inc_rank_samples.append(inc_r[e_pos])
            vote_rank_samples.append(vote_posterior.rank_samples[s][vote_idx[e]])

        inc_arr = np.array(inc_rank_samples)
        vote_arr = np.array(vote_rank_samples)

        lambda_med = float(np.median(inc_arr))
        lambda_ci = (float(np.percentile(inc_arr, 5)), float(np.percentile(inc_arr, 95)))
        vote_med = float(np.median(vote_arr))
        vote_ci = (float(np.percentile(vote_arr, 5)), float(np.percentile(vote_arr, 95)))

        lambda_tier = sum(1 for b in tier_boundaries if lambda_med > b)
        vote_tier = sum(1 for b in tier_boundaries if vote_med > b)

        tier_agree = _tier_agreement_label(lambda_tier, vote_tier)

        if abs(lambda_med - vote_med) < 0.5:
            comp_direction = "concordant"
        elif vote_med < lambda_med:
            comp_direction = "votes-over-lambda"
        else:
            comp_direction = "lambda-over-votes"

        if tier_agree == "same":
            action = "confirmed"
        elif tier_agree == "±1":
            action = "note"
        else:
            action = "review"

        comparisons.append({
            "entry_id": e,
            "lambda_rank_median": lambda_med,
            "lambda_rank_ci": lambda_ci,
            "vote_rank_median": vote_med,
            "vote_rank_ci": vote_ci,
            "lambda_tier": lambda_tier,
            "vote_tier": vote_tier,
            "tier_agreement": tier_agree,
            "direction": comp_direction,
            "action": action,
        })

    return ConcordanceResult(
        weighted_kappa_median=median_k,
        weighted_kappa_ci=ci,
        measurable_count=measurable_count,
        total_count=total_count,
        coverage_ratio=coverage,
        below_prereg_minimum=below_min,
        meaningful_kappa_n=meaningful_kappa_n,
        flags=tuple(flags),
        standing_caveat=STANDING_CAVEAT,
        entry_comparisons=tuple(comparisons),
    )


def format_rank_comparison_report(result: ConcordanceResult) -> str:
    """Format a markdown rank comparison report from a ConcordanceResult."""
    if (
        result.entry_comparisons is None
        or result.weighted_kappa_median is None
        or result.weighted_kappa_ci is None
    ):
        return "No rank comparison data available (insufficient entries for kappa).\n"

    lines = ["# Rank Comparison Report\n"]
    lines.append(
        f"Kappa: {result.weighted_kappa_median:.3f} "
        f"[{result.weighted_kappa_ci[0]:.3f}, {result.weighted_kappa_ci[1]:.3f}]\n"
    )
    lines.append(
        "| Entry | Lambda Rank (90% CI) | Vote Rank (90% CI) | "
        "Tier Agreement | Direction | Action |"
    )
    lines.append(
        "|-------|---------------------|-------------------|----------------|-----------|--------|"
    )

    for comp in result.entry_comparisons:
        lci = cast("tuple[float, float]", comp["lambda_rank_ci"])
        vci = cast("tuple[float, float]", comp["vote_rank_ci"])
        lines.append(
            f"| {comp['entry_id']} "
            f"| {comp['lambda_rank_median']:.1f} ({lci[0]:.1f}–{lci[1]:.1f}) "
            f"| {comp['vote_rank_median']:.1f} ({vci[0]:.1f}–{vci[1]:.1f}) "
            f"| {comp['tier_agreement']} "
            f"| {comp['direction']} "
            f"| {comp['action']} |"
        )

    actions = [c["action"] for c in result.entry_comparisons]
    confirmed = actions.count("confirmed")
    noted = actions.count("note")
    review = actions.count("review")
    lines.append(f"\nSummary: {confirmed} confirmed, {noted} note, {review} review")

    return "\n".join(lines) + "\n"
