"""CLI commands for the 6-stage calibration pipeline.

classify → sample → generate-batches → tally → calibrate → cv-stability
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click

from engine.version import __version__


@click.command("cal-classify")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option("--rubric", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--manifest", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--snapshot-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--snapshot-date", required=True, type=str)
@click.option("--confidence-threshold", type=float, default=0.3)
def cal_classify(
    cycle: Path,
    rubric: Path,
    manifest: Path,
    snapshot_dir: Path,
    snapshot_date: str,
    confidence_threshold: float,
) -> None:
    """Stage 1: Run the deterministic keyword/indicator classifier."""
    from engine.calibrate.provenance import StageProvenance, hash_file, hash_json, write_provenance
    from engine.classify.classifier import build_rules_from_rubric, classify_real
    from engine.prereg.gates import require_classifier_rule_hash_match
    from engine.prereg.manifest import PreregManifest
    from engine.prereg.rubric_io import read_rubric

    cal_dir = cycle / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_lock_hash = hash_file(manifest)

    rb = read_rubric(rubric)
    rules = build_rules_from_rubric(rb, confidence_threshold)

    mreg = PreregManifest(**manifest_data)
    require_classifier_rule_hash_match(mreg, rules.rule_hash)

    from engine.adapters.genai_agentic import GenAIAgenticAdapter
    adapter = GenAIAgenticAdapter(snapshot_dir, snapshot_date)
    incidents = tuple(adapter.iter_incidents())

    result = classify_real(incidents, rules)

    out_path = cal_dir / "classifications.json"
    out_data = {
        "classifier_version": result.classifier_version,
        "classifier_rule_hash": result.classifier_rule_hash,
        "classification_count": len(result.classifications),
        "classifications": [
            {
                "incident_id": c.incident_id,
                "entry_id": c.entry_id,
                "confidence": c.confidence,
                "stage": c.stage,
                "rationale": c.rationale,
            }
            for c in result.classifications
        ],
    }
    out_path.write_text(json.dumps(out_data, indent=2) + "\n")

    prov = StageProvenance(
        stage_name="classify",
        manifest_lock_hash=manifest_lock_hash,
        input_hashes={"rubric": hash_file(rubric)},
        output_hash=hash_json(out_data),
        timestamp=datetime.now(UTC).isoformat(),
        engine_version=__version__,
    )
    write_provenance(prov, cal_dir / "classify_provenance.json")

    click.echo(f"Classified {len(incidents)} incidents → {len(result.classifications)} labels.")
    click.echo(f"Rule hash: {result.classifier_rule_hash}")


@click.command("cal-sample")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_sample(cycle: Path) -> None:
    """Stage 2: Draw precision-frame and recall-frame samples."""
    click.echo("cal-sample: Not yet wired (Task 11 placeholder for CLI registration).")


@click.command("cal-generate-batches")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_generate_batches(cycle: Path) -> None:
    """Stage 3: Generate batch files for manual coding."""
    click.echo("cal-generate-batches: Not yet wired.")


@click.command("cal-tally")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_tally(cycle: Path) -> None:
    """Stage 4: Aggregate coded labels into per-entry per-stratum counts."""
    click.echo("cal-tally: Not yet wired.")


@click.command("cal-calibrate")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_calibrate(cycle: Path) -> None:
    """Stage 5: Compute Beta posteriors from tally counts."""
    click.echo("cal-calibrate: Not yet wired.")


@click.command("cal-cv-stability")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_cv_stability(cycle: Path) -> None:
    """Stage 6: k=5 cross-validation for calibration stability."""
    click.echo("cal-cv-stability: Not yet wired.")
