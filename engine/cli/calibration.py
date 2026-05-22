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
@click.option("--manifest", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--snapshot-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--snapshot-date", required=True, type=str)
@click.option("--precision-n", type=int, default=40)
@click.option("--recall-n", type=int, default=100)
@click.option("--seed", type=int, default=42)
def cal_sample(
    cycle: Path,
    manifest: Path,
    snapshot_dir: Path,
    snapshot_date: str,
    precision_n: int,
    recall_n: int,
    seed: int,
) -> None:
    """Stage 2: Draw precision-frame and recall-frame samples."""
    from engine.adapters.genai_agentic import GenAIAgenticAdapter
    from engine.calibrate.provenance import (
        StageProvenance,
        hash_json,
        read_provenance,
        write_provenance,
    )
    from engine.calibrate.sampler import SampleFrame, SampleRequest
    from engine.calibrate.two_frame_sampler import TwoFrameSampler
    from engine.classify.stub import ClassificationResult

    cal_dir = cycle / "calibration"
    classify_prov = read_provenance(cal_dir / "classify_provenance.json")

    classifications_path = cal_dir / "classifications.json"
    cls_data = json.loads(classifications_path.read_text())
    from engine.classify.stub import Classification

    classifications = tuple(
        Classification(
            incident_id=c["incident_id"],
            entry_id=c["entry_id"],
            confidence=c["confidence"],
            stage=c.get("stage", 1),
            rationale=c.get("rationale", ""),
        )
        for c in cls_data["classifications"]
    )
    cls_result = ClassificationResult(
        classifications=classifications,
        classifier_version=cls_data["classifier_version"],
        classifier_rule_hash=cls_data["classifier_rule_hash"],
    )

    adapter = GenAIAgenticAdapter(snapshot_dir, snapshot_date)
    incidents = list(adapter.iter_incidents())
    entry_defs = adapter.entry_definitions()
    non_fb_ids = [e.entry_id for e in entry_defs if not e.frame_blind]
    strata = sorted({inc.corpus_stratum for inc in incidents})

    sampler = TwoFrameSampler(classification_result=cls_result)
    sample_results: list[dict[str, object]] = []

    for eid in non_fb_ids:
        for s in strata:
            req = SampleRequest(
                frame=SampleFrame.PRECISION, entry_id=eid, stratum=s, n=precision_n,
            )
            sr = sampler.draw(req, incidents, seed=seed)
            if sr.actual_n > 0:
                sample_results.append({
                    "frame": "precision",
                    "entry_id": eid,
                    "stratum": s,
                    "actual_n": sr.actual_n,
                    "sample_hash": sr.sample_hash,
                    "incident_ids": sorted(inc.id for inc in sr.incidents),
                })

    for s in strata:
        req = SampleRequest(
            frame=SampleFrame.RECALL, entry_id=None, stratum=s, n=recall_n,
        )
        sr = sampler.draw(req, incidents, seed=seed)
        sample_results.append({
            "frame": "recall",
            "entry_id": None,
            "stratum": s,
            "actual_n": sr.actual_n,
            "sample_hash": sr.sample_hash,
            "incident_ids": sorted(inc.id for inc in sr.incidents),
        })

    fb_ids = sorted(e.entry_id for e in entry_defs if e.frame_blind)
    samples_data = {
        "seed": seed,
        "precision_n": precision_n,
        "recall_n": recall_n,
        "frame_blind_ids": fb_ids,
        "samples": sample_results,
    }
    out_path = cal_dir / "samples.json"
    out_path.write_text(json.dumps(samples_data, indent=2) + "\n")

    prov = StageProvenance(
        stage_name="sample",
        manifest_lock_hash=classify_prov.manifest_lock_hash,
        input_hashes={"classify": classify_prov.output_hash},
        output_hash=hash_json(samples_data),
        timestamp=datetime.now(UTC).isoformat(),
        engine_version=__version__,
    )
    write_provenance(prov, cal_dir / "sample_provenance.json")

    n_prec = sum(1 for s in sample_results if s["frame"] == "precision")
    n_rec = sum(1 for s in sample_results if s["frame"] == "recall")
    click.echo(
        f"Sampled {n_prec} precision batches + {n_rec} recall batches."
    )


@click.command("cal-generate-batches")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option("--manifest", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--rubric", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--snapshot-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--snapshot-date", required=True, type=str)
@click.option("--coder-id", required=True, type=str)
@click.option("--seed", type=int, default=42)
def cal_generate_batches(
    cycle: Path,
    manifest: Path,
    rubric: Path,
    snapshot_dir: Path,
    snapshot_date: str,
    coder_id: str,
    seed: int,
) -> None:
    """Stage 3: Generate batch files for manual coding."""
    from engine.adapters.genai_agentic import GenAIAgenticAdapter
    from engine.calibrate.batch import generate_batch
    from engine.calibrate.provenance import (
        StageProvenance,
        hash_file,
        hash_json,
        read_provenance,
        write_provenance,
    )
    from engine.calibrate.sampler import SampleFrame, SampleRequest
    from engine.calibrate.two_frame_sampler import TwoFrameSampler
    from engine.classify.stub import Classification, ClassificationResult
    from engine.prereg.rubric_io import read_rubric

    cal_dir = cycle / "calibration"
    sample_prov = read_provenance(cal_dir / "sample_provenance.json")

    samples_data = json.loads((cal_dir / "samples.json").read_text())
    cls_data = json.loads((cal_dir / "classifications.json").read_text())

    classifications = tuple(
        Classification(
            incident_id=c["incident_id"],
            entry_id=c["entry_id"],
            confidence=c["confidence"],
            stage=c.get("stage", 1),
            rationale=c.get("rationale", ""),
        )
        for c in cls_data["classifications"]
    )
    cls_result = ClassificationResult(
        classifications=classifications,
        classifier_version=cls_data["classifier_version"],
        classifier_rule_hash=cls_data["classifier_rule_hash"],
    )

    adapter = GenAIAgenticAdapter(snapshot_dir, snapshot_date)
    incidents = list(adapter.iter_incidents())

    rb = read_rubric(rubric)
    checklist = {entry.entry_id: entry.canonical_name for entry in rb.entries}

    rubric_hash = hash_file(rubric)
    manifest_lock_hash = hash_file(manifest)
    cycle_id = cycle.name

    sampler = TwoFrameSampler(classification_result=cls_result)
    batch_dir = cal_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_count = 0
    for sample_info in samples_data["samples"]:
        frame = SampleFrame(sample_info["frame"])
        req = SampleRequest(
            frame=frame,
            entry_id=sample_info["entry_id"],
            stratum=sample_info["stratum"],
            n=sample_info["actual_n"],
        )
        sr = sampler.draw(req, incidents, seed=samples_data["seed"])

        coding_checklist = checklist if frame == SampleFrame.RECALL else None
        batch = generate_batch(
            sample_result=sr,
            rubric_hash=rubric_hash,
            manifest_lock_hash=manifest_lock_hash,
            coder_id=coder_id,
            cycle_id=cycle_id,
            coding_checklist=coding_checklist,
        )
        batch.write(batch_dir / f"{batch.header.batch_id}.json")
        batch_count += 1

    batches_manifest = {
        "batch_count": batch_count,
        "coder_id": coder_id,
        "rubric_hash": rubric_hash,
        "manifest_lock_hash": manifest_lock_hash,
    }
    (cal_dir / "batches_manifest.json").write_text(
        json.dumps(batches_manifest, indent=2) + "\n"
    )

    prov = StageProvenance(
        stage_name="generate-batches",
        manifest_lock_hash=manifest_lock_hash,
        input_hashes={"sample": sample_prov.output_hash},
        output_hash=hash_json(batches_manifest),
        timestamp=datetime.now(UTC).isoformat(),
        engine_version=__version__,
    )
    write_provenance(prov, cal_dir / "generate_batches_provenance.json")

    click.echo(f"Generated {batch_count} batch files in {batch_dir}")
    click.echo("Code the batches, then run cal-tally.")


@click.command("cal-tally")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option("--manifest", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--rubric", required=True, type=click.Path(exists=True, path_type=Path))
def cal_tally(cycle: Path, manifest: Path, rubric: Path) -> None:
    """Stage 4: Aggregate coded labels into per-entry per-stratum counts."""
    from engine.calibrate.provenance import (
        StageProvenance,
        hash_file,
        hash_json,
        read_provenance,
        write_provenance,
    )
    from engine.calibrate.tally import validate_and_tally
    from engine.prereg.rubric_io import read_rubric

    cal_dir = cycle / "calibration"
    read_provenance(cal_dir / "generate_batches_provenance.json")

    rb = read_rubric(rubric)
    all_entry_ids = {entry.entry_id for entry in rb.entries}
    rollup_ids = {entry.entry_id for entry in rb.entries if entry.is_rollup_candidate}

    sample_hashes: dict[str, str] = {}
    batch_dir = cal_dir / "batches"
    batch_paths = sorted(batch_dir.glob("*.json"))

    from engine.calibrate.batch import CodingBatch

    for p in batch_paths:
        b = CodingBatch.read(p)
        sample_hashes[b.header.batch_id] = b.header.sample_hash

    corpus_ids = None

    rubric_hash = hash_file(rubric)
    lock_hash = hash_file(manifest)

    tally = validate_and_tally(
        batch_paths=list(batch_paths),
        valid_entry_ids=all_entry_ids,
        rollup_entry_ids=rollup_ids,
        expected_sample_hashes=sample_hashes,
        expected_rubric_hash=rubric_hash,
        expected_lock_hash=lock_hash,
        all_entry_ids=all_entry_ids,
        expected_incident_ids=corpus_ids,
    )

    tally_data = {
        "total_coded": tally.total_coded,
        "amendments_applied": tally.amendments_applied,
        "precision_counts": {
            f"{k[0]}::{k[1]}": {
                "true_positives": v.true_positives,
                "false_positives": v.false_positives,
                "total": v.total,
            }
            for k, v in tally.precision_counts.items()
        },
        "recall_counts": {
            f"{k[0]}::{k[1]}": {
                "true_positives": v.true_positives,
                "false_negatives": v.false_negatives,
                "total_in_sample": v.total_in_sample,
            }
            for k, v in tally.recall_counts.items()
        },
        "rollup_counts": {
            f"{k[0]}::{k[1]}": {
                "true_positives": v.true_positives,
                "false_positives": v.false_positives,
                "total": v.total,
            }
            for k, v in tally.rollup_counts.items()
        },
    }
    out_path = cal_dir / "tally.json"
    out_path.write_text(json.dumps(tally_data, indent=2) + "\n")

    prov = StageProvenance(
        stage_name="tally",
        manifest_lock_hash=lock_hash,
        input_hashes={"batches_rubric_hash": rubric_hash},
        output_hash=hash_json(tally_data),
        timestamp=datetime.now(UTC).isoformat(),
        engine_version=__version__,
    )
    write_provenance(prov, cal_dir / "tally_provenance.json")

    click.echo(f"Tallied {tally.total_coded} coded labels, {tally.amendments_applied} amendments.")


@click.command("cal-calibrate")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option("--rubric", required=True, type=click.Path(exists=True, path_type=Path))
def cal_calibrate(cycle: Path, rubric: Path) -> None:
    """Stage 5: Compute Beta posteriors from tally counts."""
    from engine.calibrate.calibrate import compute_calibration
    from engine.calibrate.provenance import (
        StageProvenance,
        hash_json,
        read_provenance,
        write_provenance,
    )
    from engine.calibrate.tally import PrecisionTally, RecallTally, TallyResult
    from engine.prereg.rubric_io import read_rubric

    cal_dir = cycle / "calibration"
    tally_prov = read_provenance(cal_dir / "tally_provenance.json")

    tally_data = json.loads((cal_dir / "tally.json").read_text())

    prec_counts: dict[tuple[str, str], PrecisionTally] = {}
    for key_str, v in tally_data["precision_counts"].items():
        parts = key_str.split("::")
        prec_counts[(parts[0], parts[1])] = PrecisionTally(
            true_positives=v["true_positives"],
            false_positives=v["false_positives"],
            total=v["total"],
        )

    rec_counts: dict[tuple[str, str], RecallTally] = {}
    for key_str, v in tally_data["recall_counts"].items():
        parts = key_str.split("::")
        rec_counts[(parts[0], parts[1])] = RecallTally(
            true_positives=v["true_positives"],
            false_negatives=v["false_negatives"],
            total_in_sample=v["total_in_sample"],
        )

    rollup_counts: dict[tuple[str, str], PrecisionTally] = {}
    for key_str, v in tally_data.get("rollup_counts", {}).items():
        parts = key_str.split("::")
        rollup_counts[(parts[0], parts[1])] = PrecisionTally(
            true_positives=v["true_positives"],
            false_positives=v["false_positives"],
            total=v["total"],
        )

    tally = TallyResult(
        precision_counts=prec_counts,
        recall_counts=rec_counts,
        rollup_counts=rollup_counts,
        total_coded=tally_data["total_coded"],
        amendments_applied=tally_data["amendments_applied"],
    )

    rb = read_rubric(rubric)
    all_entry_ids = [entry.entry_id for entry in rb.entries]

    samples_data = json.loads((cal_dir / "samples.json").read_text())
    frame_blind_ids: set[str] = set(samples_data.get("frame_blind_ids", []))

    cal, diag = compute_calibration(
        tally, all_entry_ids=all_entry_ids, frame_blind_ids=frame_blind_ids,
    )

    posteriors_data: dict[str, dict[str, dict[str, float]]] = {
        "recall": {},
        "precision": {},
    }
    for (eid, stratum), bp in cal.recall.items():
        posteriors_data["recall"][f"{eid}::{stratum}"] = {
            "alpha": bp.alpha, "beta": bp.beta,
        }
    for (eid, stratum), bp in cal.precision.items():
        posteriors_data["precision"][f"{eid}::{stratum}"] = {
            "alpha": bp.alpha, "beta": bp.beta,
        }

    posteriors_path = cal_dir / "posteriors.json"
    posteriors_path.write_text(json.dumps(posteriors_data, indent=2) + "\n")

    diag_data = {
        "entries_with_both_frames": diag.entries_with_both_frames,
        "entries_recall_only": diag.entries_recall_only,
        "entries_no_data": diag.entries_no_data,
        "entry_reports": {
            eid: {
                "has_precision_data": r.has_precision_data,
                "has_recall_data": r.has_recall_data,
                "precision_ci_width": r.precision_ci_width,
                "recall_ci_width": r.recall_ci_width,
                "recall_sample_size": r.recall_sample_size,
                "precision_sample_size": r.precision_sample_size,
                "min_fold_count": r.min_fold_count,
                "flag": r.flag,
                "reason": r.reason,
            }
            for eid, r in diag.entry_reports.items()
        },
    }
    (cal_dir / "diagnostic.json").write_text(json.dumps(diag_data, indent=2) + "\n")

    prov = StageProvenance(
        stage_name="calibrate",
        manifest_lock_hash=tally_prov.manifest_lock_hash,
        input_hashes={"tally": tally_prov.output_hash},
        output_hash=hash_json(posteriors_data),
        timestamp=datetime.now(UTC).isoformat(),
        engine_version=__version__,
    )
    write_provenance(prov, cal_dir / "calibrate_provenance.json")

    adequate = sum(1 for r in diag.entry_reports.values() if r.flag == "adequate")
    wide = sum(1 for r in diag.entry_reports.values() if r.flag == "wide")
    no_data = sum(1 for r in diag.entry_reports.values() if r.flag == "no-data")
    click.echo(f"Calibration complete: {adequate} adequate, {wide} wide, {no_data} no-data.")
    click.echo(f"Posteriors written to {posteriors_path}")


@click.command("cal-cv-stability")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option("--n-folds", type=int, default=5)
def cal_cv_stability(cycle: Path, n_folds: int) -> None:
    """Stage 6: k=5 cross-validation for calibration stability."""
    from engine.calibrate.cv import cross_validate_calibration
    from engine.calibrate.provenance import (
        StageProvenance,
        hash_json,
        read_provenance,
        write_provenance,
    )

    cal_dir = cycle / "calibration"
    cal_prov = read_provenance(cal_dir / "calibrate_provenance.json")

    batch_dir = cal_dir / "batches"
    batch_paths = sorted(batch_dir.glob("*.json"))

    from engine.calibrate.batch import CodingBatch

    precision_labels: dict[tuple[str, str], list[bool]] = {}
    recall_labels: dict[tuple[str, str], list[bool]] = {}

    for p in batch_paths:
        batch = CodingBatch.read(p)
        if batch.header.frame == "precision" and batch.header.entry_id:
            key = (batch.header.entry_id, batch.header.stratum)
            precision_labels.setdefault(key, [])
            for inc in batch.incidents:
                if inc.labels is not None:
                    precision_labels[key].append(
                        batch.header.entry_id in inc.labels
                    )
        elif batch.header.frame == "recall":
            tally_data = json.loads((cal_dir / "tally.json").read_text())
            all_eids = set()
            for key_str in tally_data.get("recall_counts", {}):
                all_eids.add(key_str.split("::")[0])
            for inc in batch.incidents:
                if inc.labels is not None:
                    for eid in all_eids:
                        key = (eid, batch.header.stratum)
                        recall_labels.setdefault(key, [])
                        recall_labels[key].append(eid in inc.labels)

    cv = cross_validate_calibration(precision_labels, recall_labels, n_folds=n_folds)

    cv_data = {
        "n_folds": cv.n_folds,
        "fold_variances": {
            f"{k[0]}::{k[1]}": v for k, v in cv.fold_variances.items()
        },
        "interpretation": {
            f"{k[0]}::{k[1]}": v for k, v in cv.interpretation.items()
        },
        "min_per_fold": {
            f"{k[0]}::{k[1]}": v for k, v in cv.min_per_fold.items()
        },
    }
    out_path = cal_dir / "cv_result.json"
    out_path.write_text(json.dumps(cv_data, indent=2) + "\n")

    prov = StageProvenance(
        stage_name="cv-stability",
        manifest_lock_hash=cal_prov.manifest_lock_hash,
        input_hashes={"calibrate": cal_prov.output_hash},
        output_hash=hash_json(cv_data),
        timestamp=datetime.now(UTC).isoformat(),
        engine_version=__version__,
    )
    write_provenance(prov, cal_dir / "cv_stability_provenance.json")

    stable = sum(1 for v in cv.interpretation.values() if v == "stable")
    total = len(cv.interpretation)
    click.echo(f"CV stability: {stable}/{total} entries stable ({n_folds} folds).")
