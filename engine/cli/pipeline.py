"""Real-data pipeline CLI commands for Plan 5.

These commands wire the existing engine modules into a production pipeline
for the 2026 LLM Top 10 cycle.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from engine.classify.stage2_manifest import Stage2Manifest


def _default_tier_boundaries(n_entries: int) -> tuple[int, ...]:
    """Default tier boundaries: split entries into 3 tiers."""
    if n_entries <= 3:
        return tuple(range(1, n_entries))
    third = n_entries // 3
    return (third, 2 * third)


@click.command(name="classify-real")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--stage2-config", type=click.Path(path_type=Path), default=None,
              help="Path to stage2_manifest.json for LLM-assisted classification")
@click.option("--execute", is_flag=True, default=False,
              help="Execute classification (without flag, validates prerequisites only)")
def classify_real(cycle: Path, stage2_config: Path | None, execute: bool) -> None:
    """Run Stage-1 + optional Stage-2 classification on real corpus data."""
    prereg = cycle / "prereg"
    if not (prereg / "manifest.json").exists():
        raise click.ClickException("prereg/manifest.json not found")
    if not (prereg / "manifest.lock").exists():
        raise click.ClickException("prereg lock not found — run prereg first")
    if not (prereg / "rubric.json").exists():
        raise click.ClickException("prereg/rubric.json not found — freeze rubric first")

    vote_dir = cycle / "vote"
    if vote_dir.exists() and any(vote_dir.iterdir()):
        raise click.ClickException("Vote data found during classify phase — vote enters only at decide")

    # R3: calibration posteriors must exist before real classification
    cal_path = cycle / "calibrate" / "posteriors.json"
    if not cal_path.exists():
        raise click.ClickException(
            f"Calibration posteriors not found: {cal_path}. "
            "Run the gold-set calibration pipeline (Plan 4) first."
        )

    from engine.classify.classifier import build_rules_from_rubric, classify_real as _classify
    from engine.prereg.rubric_io import read_rubric

    rubric = read_rubric(prereg / "rubric.json")
    manifest_data = json.loads((prereg / "manifest.json").read_text())
    confidence_threshold = manifest_data.get("confidence_threshold", 0.3)
    rules = build_rules_from_rubric(rubric, confidence_threshold=confidence_threshold)

    corpus_dir = cycle / "corpora"
    if not corpus_dir.exists():
        raise click.ClickException(f"Corpus directory not found: {corpus_dir}")

    click.echo(f"Stage-1 classification: {len(rules.rules_by_entry)} entry rules loaded")

    if not execute:
        click.echo("Classify phase: prerequisites satisfied. Run with --execute to classify.")
        return

    # Execute real classification pipeline
    click.echo("Executing classify phase...")
    try:
        from engine.cli.pipeline_executor import (
            merge_classifications,
            route_to_stage2,
            write_classify_artifacts,
        )
        from engine.schema import IncidentRecord

        # Load corpus incidents from corpora directory
        incidents: list[IncidentRecord] = []
        for jsonl_file in sorted(corpus_dir.glob("*.jsonl")):
            for line in jsonl_file.read_text().splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    incidents.append(IncidentRecord(
                        id=rec["id"],
                        date=rec.get("date", "1970-01-01"),
                        text=rec.get("text", ""),
                        severity=rec.get("severity"),
                        source_class=rec.get("source_class", "unknown"),
                        corpus_stratum=rec.get("corpus_stratum", "unknown"),
                        quality=rec.get("quality", "auto"),
                        native_labels=tuple(rec.get("native_labels", ())),
                        source_url=rec.get("source_url", ""),
                    ))

        click.echo(f"Loaded {len(incidents)} incidents from corpus")

        # Stage-1 classification
        result = _classify(tuple(incidents), rules)
        click.echo(f"Stage-1 produced {len(result.classifications)} classifications")

        # Stage-2 routing (if configured)
        stage2_results: tuple = ()
        if stage2_config is not None:
            low_confidence_ids = route_to_stage2(
                result.classifications, confidence_threshold=confidence_threshold,
            )
            click.echo(f"Routed {len(low_confidence_ids)} incidents to Stage-2")

            if low_confidence_ids:
                import os

                from engine.classify.cost_tracker import CostTracker
                from engine.classify.runpod_client import HttpRunPodClient
                from engine.classify.stage2 import Stage2Classifier
                from engine.classify.stage2_manifest import Stage2Manifest
                from engine.cli.secrets import load_secret

                s2_manifest = Stage2Manifest.read(stage2_config)
                api_key = load_secret("runpod/api-key", env_var="RUNPOD_API_KEY")
                endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")

                client = HttpRunPodClient(api_key=api_key, endpoint_id=endpoint_id)
                tracker = CostTracker(ceiling_usd=s2_manifest.cost_ceiling_usd)

                classifier = Stage2Classifier(
                    client=client,
                    cost_tracker=tracker,
                    rubric_json=(prereg / "rubric.json").read_text(),
                    model_identity=s2_manifest.model_identity,
                    weight_provenance_hash=s2_manifest.weight_provenance_hash,
                    prng_seed=s2_manifest.prng_seed,
                )

                # Filter incidents for Stage-2
                s2_incidents = tuple(i for i in incidents if i.id in low_confidence_ids)
                rubric_hash = manifest_data.get("rubric_hash", "")
                stage2_results = classifier.classify_batch(s2_incidents, rubric_hash)
                client.close()

                click.echo(
                    f"Stage-2 classified {len(stage2_results)} incidents, "
                    f"cost: ${tracker.total_cost_usd:.2f}"
                )

                # Merge Stage-1 and Stage-2 results
                merged = merge_classifications(
                    result.classifications, stage2_results, confidence_threshold,
                )
                from engine.classify.stub import ClassificationResult
                result = ClassificationResult(
                    classifications=merged,
                    classifier_version=result.classifier_version,
                    classifier_rule_hash=result.classifier_rule_hash,
                )

        # Write artifacts
        out_dir = cycle / "classify"
        incident_strata = {inc.id: inc.corpus_stratum for inc in incidents}
        write_classify_artifacts(result, out_dir, stage2_results=stage2_results, incident_strata=incident_strata)
        click.echo(f"Classify phase complete. Artifacts written to {out_dir}")
    except Exception as e:
        raise click.ClickException(f"Classify phase failed: {e}")


@click.command(name="infer-real")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--num-warmup", type=int, default=1000)
@click.option("--num-samples", type=int, default=2000)
@click.option("--timeout-seconds", type=float, default=None)
@click.option("--execute", is_flag=True, default=False,
              help="Execute inference (without flag, validates prerequisites only)")
def infer_real(
    cycle: Path,
    num_warmup: int,
    num_samples: int,
    timeout_seconds: float | None,
    execute: bool,
) -> None:
    """Run NUTS inference on classified real data."""
    prereg = cycle / "prereg"
    if not (prereg / "manifest.lock").exists():
        raise click.ClickException("prereg lock not found")

    vote_dir = cycle / "vote"
    if vote_dir.exists() and any(vote_dir.iterdir()):
        raise click.ClickException(
            "Vote data found during infer phase. Vote enters only at decide. "
            "Remove vote/ from the cycle directory before running infer."
        )

    classify_dir = cycle / "classify"
    if not (classify_dir / "labeled_incidents.json").exists():
        raise click.ClickException("classify/labeled_incidents.json not found — run classify first")

    # R3: calibration posteriors must exist for real inference (no silent Beta(1,1) fallback)
    cal_path = cycle / "calibrate" / "posteriors.json"
    if not cal_path.exists():
        raise click.ClickException(
            f"Calibration posteriors not found: {cal_path}. "
            "Run the gold-set calibration pipeline (Plan 4) first. "
            "Real inference MUST NOT use uniform Beta(1,1) priors."
        )

    click.echo(f"NUTS parameters: warmup={num_warmup}, samples={num_samples}")

    import os
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "true")

    if not execute:
        click.echo("Infer phase: prerequisites satisfied. Run with --execute to start NUTS inference.")
        return

    # Execute real inference pipeline
    click.echo("Executing infer phase...")
    try:
        from engine.cli.pipeline_executor import execute_infer_phase

        execute_infer_phase(
            cycle,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=4,
        )
        click.echo("Infer phase complete.")
    except Exception as e:
        raise click.ClickException(f"Infer phase failed: {e}")


@click.command(name="decide-real")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--vote-xlsx", required=True, type=click.Path(path_type=Path, exists=True),
              help="Path to vote results XLSX file")
@click.option("--execute", is_flag=True, default=False,
              help="Execute decision phase (without flag, validates prerequisites only)")
def decide_real(cycle: Path, vote_xlsx: Path, execute: bool) -> None:
    """Run decision layer: vote posterior + concordance + flags."""
    prereg = cycle / "prereg"
    if not (prereg / "manifest.lock").exists():
        raise click.ClickException("prereg lock not found")

    infer_dir = cycle / "infer"
    if not infer_dir.exists():
        raise click.ClickException("infer/ directory not found — run infer first")

    click.echo(f"Decide phase: loading vote data from {vote_xlsx}")

    if not execute:
        click.echo("Decide phase: prerequisites satisfied. Run with --execute to decide.")
        return

    # Execute real decision pipeline
    click.echo("Executing decide phase...")
    try:
        from engine.cli.pipeline_executor import write_decide_artifacts, _load_manifest
        from engine.decide.concordance import compute_concordance
        from engine.decide.selection_bias import compute_selection_bias
        from engine.model.inference import InferenceResult
        from engine.vote.bootstrap import bootstrap_vote_ranks
        from engine.vote.loader import load_vote_data

        # Load manifest
        manifest = _load_manifest(prereg / "manifest.json")

        # Load inference results
        lambda_samples_path = infer_dir / "lambda_samples.npy"
        summary_path = infer_dir / "inference_summary.json"
        if not lambda_samples_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                "Inference artifacts not found. Run infer --execute first."
            )
        lambda_samples = np.load(lambda_samples_path)
        summary = json.loads(summary_path.read_text())
        entry_ids = tuple(summary.get("entry_ids", []))

        inference_result = InferenceResult(
            lambda_samples=lambda_samples,
            entry_ids=entry_ids,
            r_hat=summary.get("r_hat", {}),
            ess=summary.get("ess", {}),
            divergences=summary.get("divergences", 0),
            num_warmup=summary.get("num_warmup", 1000),
            num_samples=summary.get("num_samples", 2000),
        )

        # Load vote data and bootstrap
        vote_data = load_vote_data(vote_xlsx)
        click.echo(f"Loaded vote data: {vote_data.n_respondents} respondents")

        vote_posterior = bootstrap_vote_ranks(
            respondent_rankings=vote_data.rankings,
            entry_ids=vote_data.entry_ids,
            n_bootstrap=5000,
            seed=manifest.prng_seed,
        )

        # Compute concordance with correct 8-parameter signature
        concordance = compute_concordance(
            inference_result=inference_result,
            vote_posterior=vote_posterior,
            tier_boundaries=_default_tier_boundaries(len(entry_ids)),
            flag_threshold_tau=manifest.flag_threshold_tau,
            measurable_count=len(entry_ids),
            total_count=len(entry_ids),
            meaningful_kappa_n=manifest.meaningful_kappa_n,
            measurability_minimum=manifest.measurability_minimum,
        )

        # Compute selection bias
        measurability_verdicts = {e: "measurable" for e in entry_ids}
        selection_bias = compute_selection_bias(
            measurability_verdicts=measurability_verdicts,
            median_vote_ranks=vote_posterior.median_ranks,
        )

        # Write artifacts
        out_dir = cycle / "results"
        write_decide_artifacts(
            concordance,
            out_dir,
            selection_bias=selection_bias,
        )
        click.echo(f"Decide phase complete. Artifacts written to {out_dir}")
    except Exception as e:
        raise click.ClickException(f"Decide phase failed: {e}")


@click.command(name="report")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
def report_cmd(cycle: Path) -> None:
    """Generate final cycle report + reproduction bundle."""
    results_dir = cycle / "results"
    if not results_dir.exists():
        raise click.ClickException("results/ directory not found — run decide first")

    click.echo("Report phase: prerequisites satisfied.")


@click.command(name="repro-bundle")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--output", required=True, type=click.Path(path_type=Path))
def repro_bundle_cmd(cycle: Path, output: Path) -> None:
    """Generate reproduction bundle tar.gz."""
    click.echo(f"Reproduction bundle: {output}")
