"""Real-data pipeline orchestrator (R1).

Analogous to execute_synthetic_pipeline() but for real corpus data.
Wires Stage-1 classification, Stage-1→Stage-2 routing, calibration
loading, NUTS inference, and decision-layer assembly.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.classify.stage2_protocol import Stage2Classification
from engine.classify.stub import Classification, ClassificationResult


def route_to_stage2(
    classifications: tuple[Classification, ...],
    confidence_threshold: float,
) -> set[str]:
    return {
        c.incident_id
        for c in classifications
        if c.confidence < confidence_threshold
    }


def merge_classifications(
    stage1: tuple[Classification, ...],
    stage2: tuple[Stage2Classification, ...],
    confidence_threshold: float,
) -> tuple[Classification, ...]:
    stage2_by_id = {s.incident_id: s for s in stage2}
    merged: list[Classification] = []
    for c in stage1:
        if c.confidence < confidence_threshold and c.incident_id in stage2_by_id:
            s2 = stage2_by_id[c.incident_id]
            merged.append(Classification(
                incident_id=s2.incident_id,
                entry_id=s2.entry_id,
                confidence=s2.confidence,
                stage=2,
                rationale=s2.rationale,
            ))
        else:
            merged.append(c)
    return tuple(merged)


def write_classify_artifacts(
    result: ClassificationResult,
    out_dir: Path,
    stage2_results: tuple[Stage2Classification, ...] = (),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled = [
        {
            "incident_id": c.incident_id,
            "entry_id": c.entry_id,
            "confidence": c.confidence,
            "stage": c.stage,
            "rationale": c.rationale,
        }
        for c in result.classifications
    ]
    (out_dir / "labeled_incidents.json").write_text(
        json.dumps(labeled, indent=2) + "\n"
    )
    (out_dir / "stage1_results.json").write_text(
        json.dumps({
            "classifier_version": result.classifier_version,
            "classifier_rule_hash": result.classifier_rule_hash,
            "n_classifications": len(result.classifications),
        }, indent=2) + "\n"
    )
    if stage2_results:
        s2_data = [
            {
                "incident_id": s.incident_id,
                "entry_id": s.entry_id,
                "confidence": s.confidence,
                "rationale": s.rationale,
                "model_identity": s.model_identity,
                "prompt_hash": s.prompt_hash,
            }
            for s in stage2_results
        ]
        (out_dir / "stage2_results.json").write_text(
            json.dumps(s2_data, indent=2) + "\n"
        )
