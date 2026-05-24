"""Multi-model pre-labeling pipeline (spec A3)."""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from engine.classify.runpod_client import RunPodClient, RunPodError
from engine.classify.stage2_prompt import build_messages
from engine.schema import IncidentRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelVote:
    model_id: str
    entry_id: str
    confidence: float
    rationale: str


@dataclass(frozen=True, slots=True)
class PreLabelResult:
    incident_id: str
    model_votes: list[ModelVote]
    consensus: str | None
    agreement: str
    triage_tier: str


class MultiModelPreLabeler:
    def __init__(
        self,
        models: list[tuple[RunPodClient, str]],
        rubric_json: str,
        prng_seed: int,
    ) -> None:
        self._models = models
        self._rubric_json = rubric_json
        self._seed = prng_seed

    def pre_label(self, incident: IncidentRecord) -> PreLabelResult:
        messages = build_messages(incident, self._rubric_json)
        votes: list[ModelVote] = []

        for client, model_id in self._models:
            try:
                resp = client.run_sync(messages, seed=self._seed)
                data = json.loads(resp.output_text)
                votes.append(ModelVote(
                    model_id=model_id,
                    entry_id=str(data.get("entry_id", "out-of-scope")),
                    confidence=float(data.get("confidence", 0.0)),
                    rationale=str(data.get("rationale", "")),
                ))
            except (RunPodError, json.JSONDecodeError, ValueError) as e:
                logger.warning("Model %s failed for %s: %s", model_id, incident.id, e)
                votes.append(ModelVote(
                    model_id=model_id,
                    entry_id="out-of-scope",
                    confidence=0.0,
                    rationale=f"Model error: {e}",
                ))

        entry_counts = Counter(v.entry_id for v in votes)
        most_common = entry_counts.most_common()
        top_count = most_common[0][1] if most_common else 0
        n_models = len(votes)

        if top_count == n_models:
            triage_tier = "agree"
            consensus = most_common[0][0]
            agreement = f"{n_models}-of-{n_models}"
        elif top_count > 1:
            triage_tier = "split"
            consensus = most_common[0][0]
            agreement = f"{top_count}-of-{n_models}"
        else:
            triage_tier = "disagree"
            consensus = None
            agreement = f"1-of-{n_models}"

        return PreLabelResult(
            incident_id=incident.id,
            model_votes=votes,
            consensus=consensus,
            agreement=agreement,
            triage_tier=triage_tier,
        )

    def pre_label_batch(
        self,
        incidents: list[IncidentRecord],
        checkpoint_path: Path,
    ) -> None:
        done_ids: set[str] = set()
        if checkpoint_path.exists():
            for line in checkpoint_path.read_text().strip().splitlines():
                if line.strip():
                    record = json.loads(line)
                    done_ids.add(record["incident_id"])

        with checkpoint_path.open("a", encoding="utf-8") as f:
            for incident in incidents:
                if incident.id in done_ids:
                    continue
                result = self.pre_label(incident)
                record = {
                    "incident_id": result.incident_id,
                    "text": incident.text,
                    "model_votes": [
                        {
                            "model_id": v.model_id,
                            "entry_id": v.entry_id,
                            "confidence": v.confidence,
                            "rationale": v.rationale,
                        }
                        for v in result.model_votes
                    ],
                    "consensus": result.consensus,
                    "agreement": result.agreement,
                    "triage_tier": result.triage_tier,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
