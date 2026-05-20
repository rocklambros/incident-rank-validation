"""Multiplicity disclosure via empirical vote-permutation null.

HANDOFF v2.5 §5.5 / §6.11(g) requires multiplicity disclosure.

Null model choice (M15): we use vote-label permutation across entries to break
the vote-incident relationship. This preserves the marginal rank distribution
of the vote but destroys per-entry vote-rank identity. It is ONE construction
of H0 -- a per-entry-independent-mismatch null would also be defensible. We
chose label-permutation because it is the most pessimistic for entries with
extreme ranks (which dominate flags), giving an upper-bound noise floor.
The reader should treat the disclosed false-flag rate as an upper bound, not
a point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from engine.decide.kappa import quadratic_weighted_kappa


@dataclass(frozen=True, slots=True)
class MultiplicityDisclosure:
    """Result of permutation null test for kappa multiplicity."""

    null_kappa_median: float
    null_kappa_ci: tuple[float, float]
    observed_kappa_median: float
    p_value: float  # fraction of null kappas >= observed
    n_permutations: int


def permutation_null(
    incident_ranks: npt.NDArray[np.float64],
    vote_ranks: npt.NDArray[np.float64],
    tier_boundaries: tuple[int, ...],
    observed_kappa: float,
    n_permutations: int = 1000,
    seed: int = 42,
) -> MultiplicityDisclosure:
    """Test whether observed concordance beats chance via vote-label permutation."""
    rng = np.random.default_rng(seed)
    null_kappas: list[float] = []

    for _ in range(n_permutations):
        perm_vote = rng.permutation(vote_ranks)
        k = quadratic_weighted_kappa(incident_ranks, perm_vote, tier_boundaries)
        if not np.isnan(k):
            null_kappas.append(k)

    if not null_kappas:
        return MultiplicityDisclosure(
            null_kappa_median=float("nan"),
            null_kappa_ci=(float("nan"), float("nan")),
            observed_kappa_median=observed_kappa,
            p_value=1.0,
            n_permutations=n_permutations,
        )

    arr = np.array(null_kappas)
    p = float(np.mean(arr >= observed_kappa))
    return MultiplicityDisclosure(
        null_kappa_median=float(np.median(arr)),
        null_kappa_ci=(float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
        observed_kappa_median=observed_kappa,
        p_value=p,
        n_permutations=n_permutations,
    )
