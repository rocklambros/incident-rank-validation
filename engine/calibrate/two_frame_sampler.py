"""Two-frame gold-set sampler: precision frame + recall frame.

Precision frame: sample from classifier-positive incidents for a specific entry.
Recall frame: random stratified sample from all incidents.
"""
from __future__ import annotations

import random

from engine.calibrate.sampler import SampleFrame, SampleRequest, SampleResult
from engine.classify.stub import ClassificationResult
from engine.schema import IncidentRecord


class TwoFrameSampler:
    def __init__(self, classification_result: ClassificationResult) -> None:
        self._classification = classification_result

    def draw(
        self,
        request: SampleRequest,
        incidents: list[IncidentRecord],
        seed: int,
    ) -> SampleResult:
        if request.frame == SampleFrame.PRECISION:
            return self._draw_precision(request, incidents, seed)
        return self._draw_recall(request, incidents, seed)

    def _draw_precision(
        self,
        request: SampleRequest,
        incidents: list[IncidentRecord],
        seed: int,
    ) -> SampleResult:
        if request.entry_id is None:
            raise ValueError("entry_id required for PRECISION frame")

        classified_ids = {
            c.incident_id
            for c in self._classification.classifications
            if c.entry_id == request.entry_id
        }
        pool = [
            inc for inc in incidents
            if inc.id in classified_ids
            and (request.stratum is None or inc.corpus_stratum == request.stratum)
        ]

        if len(pool) < 20:
            sampled = tuple(pool)
        elif len(pool) <= request.n:
            sampled = tuple(pool)
        else:
            rng = random.Random(seed)
            sampled = tuple(rng.sample(pool, request.n))

        return SampleResult(
            incidents=sampled,
            request=request,
            actual_n=len(sampled),
            sample_hash=SampleResult.compute_sample_hash(sampled),
        )

    def _draw_recall(
        self,
        request: SampleRequest,
        incidents: list[IncidentRecord],
        seed: int,
    ) -> SampleResult:
        pool = [
            inc for inc in incidents
            if request.stratum is None or inc.corpus_stratum == request.stratum
        ]

        n = min(request.n, len(pool))
        rng = random.Random(seed)
        sampled = tuple(rng.sample(pool, n))

        return SampleResult(
            incidents=sampled,
            request=request,
            actual_n=len(sampled),
            sample_hash=SampleResult.compute_sample_hash(sampled),
        )
