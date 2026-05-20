"""Beta posteriors for per-entry, per-stratum precision and recall.

See HANDOFF §5.3: "per-entry, per-stratum precision and recall as Beta
posteriors from the labeled counts."
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BetaPosterior", "Calibration"]


@dataclass(frozen=True, slots=True)
class BetaPosterior:
    """Beta(alpha, beta) posterior for a rate parameter (precision or recall)."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"Beta parameters must be positive: alpha={self.alpha}, beta={self.beta}"
            )

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @classmethod
    def from_counts(
        cls,
        successes: int,
        failures: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> BetaPosterior:
        """Conjugate update: Beta(prior_a + successes, prior_b + failures)."""
        return cls(alpha=prior_alpha + successes, beta=prior_beta + failures)


@dataclass(frozen=True, slots=True)
class Calibration:
    """Per-entry, per-stratum precision and recall Beta posteriors."""

    recall: dict[tuple[str, str], BetaPosterior]    # {(entry_id, stratum): Beta}
    precision: dict[tuple[str, str], BetaPosterior]  # {(entry_id, stratum): Beta}
