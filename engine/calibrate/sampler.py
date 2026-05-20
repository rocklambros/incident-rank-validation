"""Sampler protocol and StratifiedSampler stub.

The Sampler protocol defines the interface for gold-set sampling.
The actual stratified sampling implementation is a Plan 4 deliverable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from engine.schema import IncidentRecord

__all__ = ["Sampler", "StratifiedSampler"]


@runtime_checkable
class Sampler(Protocol):
    def draw(
        self,
        incidents: list[IncidentRecord],
        stratum_counts: dict[str, int],
        seed: int,
    ) -> list[IncidentRecord]:
        ...


class StratifiedSampler:
    """Stub: actual stratified sampling implemented in Plan 4."""

    def draw(
        self,
        incidents: list[IncidentRecord],
        stratum_counts: dict[str, int],
        seed: int,
    ) -> list[IncidentRecord]:
        raise NotImplementedError(
            "StratifiedSampler.draw is a Plan 4 deliverable. "
            "Plan 1 proves the protocol shape only."
        )
