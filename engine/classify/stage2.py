"""Stage-2 LLM-assisted classifier — concrete implementation."""
from __future__ import annotations

import json
import logging

from engine.classify.cost_tracker import CostTracker
from engine.classify.runpod_client import RunPodClient, RunPodError
from engine.classify.stage2_prompt import build_prompt, compute_prompt_hash
from engine.classify.stage2_protocol import Stage2Classification
from engine.schema import IncidentRecord

logger = logging.getLogger(__name__)


class Stage2Classifier:
    def __init__(
        self,
        client: RunPodClient,
        cost_tracker: CostTracker,
        rubric_json: str,
        model_identity: str,
        weight_provenance_hash: str,
        prng_seed: int,
        cost_per_job_usd: float = 0.01,
    ) -> None:
        self._client = client
        self._tracker = cost_tracker
        self._rubric_json = rubric_json
        self._model_identity = model_identity
        self._weight_hash = weight_provenance_hash
        self._prompt_hash = compute_prompt_hash(rubric_json)
        self._seed = prng_seed
        self._cost_per_job = cost_per_job_usd

    def classify(
        self,
        incident: IncidentRecord,
        rubric_hash: str,
    ) -> Stage2Classification:
        self._tracker.check_or_abort()
        prompt = build_prompt(incident, self._rubric_json)
        try:
            resp = self._client.run_sync(prompt, seed=self._seed)
            self._tracker.record(
                job_id=resp.job_id,
                cost_usd=self._cost_per_job,
                execution_time_ms=resp.execution_time_ms,
            )
            return self._parse_response(incident.id, resp.output_text)
        except RunPodError:
            logger.warning("Stage-2 RunPod error for %s, classifying as out-of-scope", incident.id)
            return self._fallback(incident.id)

    def classify_batch(
        self,
        incidents: tuple[IncidentRecord, ...],
        rubric_hash: str,
    ) -> tuple[Stage2Classification, ...]:
        return tuple(self.classify(inc, rubric_hash) for inc in incidents)

    def _parse_response(self, incident_id: str, output: str) -> Stage2Classification:
        try:
            data = json.loads(output)
            return Stage2Classification(
                incident_id=incident_id,
                entry_id=str(data.get("entry_id", "out-of-scope")),
                confidence=float(data.get("confidence", 0.0)),
                rationale=str(data.get("rationale", "")),
                model_identity=self._model_identity,
                weight_provenance_hash=self._weight_hash,
                prompt_hash=self._prompt_hash,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning("Malformed Stage-2 response for %s", incident_id)
            return self._fallback(incident_id)

    def _fallback(self, incident_id: str) -> Stage2Classification:
        return Stage2Classification(
            incident_id=incident_id,
            entry_id="out-of-scope",
            confidence=0.0,
            rationale="Stage-2 classification failed or returned malformed response",
            model_identity=self._model_identity,
            weight_provenance_hash=self._weight_hash,
            prompt_hash=self._prompt_hash,
        )
