"""Sampler protocol for two-frame gold-set calibration.

The Sampler protocol defines the interface for drawing precision-frame
and recall-frame samples from the incident corpus.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from engine.schema import IncidentRecord

__all__ = [
    "SampleFrame",
    "SampleRequest",
    "SampleResult",
    "Sampler",
]


class SampleFrame(Enum):
    PRECISION = "precision"
    RECALL = "recall"


@dataclass(frozen=True, slots=True)
class SampleRequest:
    frame: SampleFrame
    entry_id: str | None
    stratum: str | None
    n: int


@dataclass(frozen=True, slots=True)
class SampleResult:
    incidents: tuple[IncidentRecord, ...]
    request: SampleRequest
    actual_n: int
    sample_hash: str

    @staticmethod
    def compute_sample_hash(incidents: tuple[IncidentRecord, ...]) -> str:
        sorted_ids = sorted(inc.id for inc in incidents)
        return hashlib.sha256(
            json.dumps(sorted_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@runtime_checkable
class Sampler(Protocol):
    def draw(
        self,
        request: SampleRequest,
        incidents: list[IncidentRecord],
        seed: int,
    ) -> SampleResult:
        ...
