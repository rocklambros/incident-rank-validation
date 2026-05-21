"""Sampler protocol, StratifiedSampler stub, and sampling result types.

The Sampler protocol defines the interface for gold-set sampling.
The actual stratified sampling implementation is a Plan 4 deliverable.
SampleFrame, SampleRequest, and SampleResult are the data types used to
communicate sampling parameters and results between the sampler and batch stages.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from engine.schema import IncidentRecord

__all__ = [
    "Sampler",
    "StratifiedSampler",
    "SampleFrame",
    "SampleRequest",
    "SampleResult",
]


# ---------------------------------------------------------------------------
# SampleFrame
# ---------------------------------------------------------------------------


class SampleFrame(str, enum.Enum):
    """Which sampling frame is being drawn.

    PRECISION
        Incidents drawn for a specific entry_id to assess precision.
    RECALL
        Incidents drawn across all strata to assess recall (entry-agnostic draw).
    """

    PRECISION = "precision"
    RECALL = "recall"


# ---------------------------------------------------------------------------
# SampleRequest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampleRequest:
    """Parameters that fully describe a sampling request.

    Fields
    ------
    frame
        The sampling frame (PRECISION or RECALL).
    entry_id
        The entry identifier for PRECISION frames, or None for RECALL frames.
    stratum
        The corpus stratum from which to sample.
    n
        Requested sample size.
    """

    frame: SampleFrame
    entry_id: str | None
    stratum: str
    n: int


# ---------------------------------------------------------------------------
# SampleResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampleResult:
    """The result of a sampling draw.

    Fields
    ------
    incidents
        The sampled incidents (immutable tuple).
    request
        The original SampleRequest that produced this result.
    actual_n
        The number of incidents actually sampled (may differ from request.n
        when the stratum is smaller than the requested size).
    sample_hash
        A deterministic hash of the sample for provenance tracking.
    """

    incidents: tuple[IncidentRecord, ...]
    request: SampleRequest
    actual_n: int
    sample_hash: str


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
