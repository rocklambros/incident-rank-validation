"""Twin-vs-NUTS agreement reporter.

Compares the point-estimate twin ranking against the NUTS posterior ranking
for the top-tier entries and reports pairwise direction disagreements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.model.inference import InferenceResult
from engine.model.twin import TwinResult

__all__ = ["TwinDisagreement", "TwinAgreement", "compare_twin_nuts"]


@dataclass(frozen=True, slots=True)
class TwinDisagreement:
    entry_a: str
    entry_b: str
    nuts_direction: str  # "a > b" or "b > a"
    twin_direction: str  # "a > b" or "b > a"


@dataclass(frozen=True, slots=True)
class TwinAgreement:
    """Compare twin and NUTS top-tier assignments."""

    disagreements: tuple[TwinDisagreement, ...]
    n_comparisons: int
    agreement_rate: float

    @property
    def has_disagreements(self) -> bool:
        return len(self.disagreements) > 0


def compare_twin_nuts(
    inference_result: InferenceResult,
    twin_result: TwinResult,
    top_n: int = 5,
) -> TwinAgreement:
    """Compare direction of top-tier pairwise comparisons between twin and NUTS.

    For each pair of entries in the top_n by either method, check if they
    agree on which entry has higher prevalence.
    """
    entries = inference_result.entry_ids
    n = min(top_n, len(entries))

    # NUTS ranking: by posterior median
    lambda_medians: dict[str, float] = {
        entries[i]: float(np.median(inference_result.lambda_samples[:, i]))
        for i in range(len(entries))
    }
    nuts_rank = sorted(lambda_medians, key=lambda e: lambda_medians[e], reverse=True)

    # Top entries from either ranking
    top_entries = set(nuts_rank[:n]) | set(twin_result.rank[:n])
    top_list = sorted(top_entries)

    disagreements: list[TwinDisagreement] = []
    n_comparisons = 0

    for i, a in enumerate(top_list):
        for b in top_list[i + 1 :]:
            if (
                a not in twin_result.prevalence_estimates
                or b not in twin_result.prevalence_estimates
            ):
                continue
            if a not in lambda_medians or b not in lambda_medians:
                continue
            n_comparisons += 1
            nuts_dir = "a > b" if lambda_medians[a] > lambda_medians[b] else "b > a"
            twin_dir = (
                "a > b"
                if twin_result.prevalence_estimates[a]
                > twin_result.prevalence_estimates[b]
                else "b > a"
            )
            if nuts_dir != twin_dir:
                disagreements.append(
                    TwinDisagreement(
                        entry_a=a,
                        entry_b=b,
                        nuts_direction=nuts_dir,
                        twin_direction=twin_dir,
                    )
                )

    rate = 1.0 - (len(disagreements) / max(n_comparisons, 1))
    return TwinAgreement(
        disagreements=tuple(disagreements),
        n_comparisons=n_comparisons,
        agreement_rate=rate,
    )
