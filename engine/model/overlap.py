"""OverlapWeights: directed FP leakage matrix between taxonomy entries.

See HANDOFF §5.4 for the overlap-weights contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OverlapWeights:
    """Directed FP leakage weights.

    weights[target][source] = fraction of source's FPs landing in target.

    Column-stochastic constraint: for each source, sum over targets <= 1.0.
    Remainder goes to out-of-scope.
    """

    weights: dict[str, dict[str, float]]

    def __post_init__(self) -> None:
        # Self-loop check (M2): entry leaking into itself is nonsense.
        for target, sources in self.weights.items():
            if target in sources:
                raise ValueError(
                    f"overlap weights cannot contain self-loop: W[{target}][{target}]"
                )
        # Column-stochasticity check.
        all_sources: set[str] = set()
        for tgt_map in self.weights.values():
            all_sources.update(tgt_map.keys())
        for src in all_sources:
            col_sum = sum(
                self.weights.get(tgt, {}).get(src, 0.0) for tgt in self.weights
            )
            if col_sum > 1.0 + 1e-6:
                raise ValueError(
                    f"overlap weights for source {src!r} sum to {col_sum:.4f} > 1"
                )
