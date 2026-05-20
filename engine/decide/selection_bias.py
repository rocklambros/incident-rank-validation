"""Selection-bias quantification via Kruskal-Wallis (M14 / HANDOFF v2.5 §6.11(h))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kruskal


@dataclass(frozen=True, slots=True)
class SelectionBiasDisclosure:
    """Result of selection-bias test across measurability verdict groups."""

    statistic_name: str  # "kruskal_wallis_h"
    statistic_value: float
    p_value: float
    n_entries_per_group: dict[str, int]
    severity: str  # "low" | "moderate" | "high"

    def is_concerning(self) -> bool:
        return self.severity in {"moderate", "high"}


def compute_selection_bias(
    measurability_verdicts: dict[str, str],  # entry_id -> verdict string value
    median_vote_ranks: dict[str, float],  # entry_id -> median vote rank
) -> SelectionBiasDisclosure:
    """Kruskal-Wallis H test: do vote-rank distributions differ across verdict groups?"""
    groups: dict[str, list[float]] = {
        "frame_blind_unmeasurable": [],
        "classifier_blind_bounded": [],
        "measurable": [],
    }
    for entry, verdict in measurability_verdicts.items():
        if entry in median_vote_ranks:
            groups[verdict].append(median_vote_ranks[entry])

    non_empty = [np.array(g) for g in groups.values() if len(g) >= 2]
    n_per_group = {k: len(v) for k, v in groups.items()}

    if len(non_empty) < 2:
        return SelectionBiasDisclosure(
            statistic_name="kruskal_wallis_h",
            statistic_value=float("nan"),
            p_value=float("nan"),
            n_entries_per_group=n_per_group,
            severity="low",
        )

    h, p = kruskal(*non_empty)
    severity = "high" if p < 0.01 else "moderate" if p < 0.05 else "low"
    return SelectionBiasDisclosure(
        statistic_name="kruskal_wallis_h",
        statistic_value=float(h),
        p_value=float(p),
        n_entries_per_group=n_per_group,
        severity=severity,
    )
