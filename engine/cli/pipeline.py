"""Real-data pipeline CLI commands for Plan 5.

These commands wire the existing engine modules into a production pipeline
for the 2026 LLM Top 10 cycle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np

if TYPE_CHECKING:
    from engine.classify.stage2_protocol import Stage2Classification
    from engine.schema import IncidentRecord


def _load_measurability_verdicts(
    calibration_dir: Path,
    entry_ids: tuple[str, ...],
) -> dict[str, str]:
    """Read measurability verdicts from calibration/diagnostic.json.

    Maps diagnostic flags to selection-bias verdict groups:
    - "no-data" → "frame_blind_unmeasurable"
    - anything else → "measurable"
    """
    diag_path = calibration_dir / "diagnostic.json"
    if not diag_path.exists():
        return {e: "measurable" for e in entry_ids}

    diag = json.loads(diag_path.read_text())
    entry_reports = diag.get("entry_reports", {})

    verdicts: dict[str, str] = {}
    for eid in entry_ids:
        report = entry_reports.get(eid, {})
        flag = report.get("flag", "")
        if flag == "no-data":
            verdicts[eid] = "frame_blind_unmeasurable"
        else:
            verdicts[eid] = "measurable"
    return verdicts


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
        raise click.ClickException(
            "Vote data found during classify phase — vote enters only at decide"
        )

    # R3: calibration posteriors must exist before real classification
    cal_path = cycle / "calibration" / "posteriors.json"
    if not cal_path.exists():
        raise click.ClickException(
            f"Calibration posteriors not found: {cal_path}. "
            "Run the gold-set calibration pipeline (Plan 4) first."
        )

    from engine.classify.classifier import build_rules_from_rubric
    from engine.classify.classifier import classify_real as _classify
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
        from engine.adapters.genai_agentic import GenAIAgenticAdapter
        from engine.cli.pipeline_executor import (
            merge_classifications,
            route_to_stage2,
            write_classify_artifacts,
        )

        snapshot_dirs = sorted(corpus_dir.glob("*/*/provenance.json"))
        if not snapshot_dirs:
            snapshot_dirs = sorted(corpus_dir.glob("*/provenance.json"))
        if not snapshot_dirs:
            raise click.ClickException(
                f"No provenance.json found under {corpus_dir}. "
                "Expected corpora/<adapter>/<hash>/provenance.json"
            )
        prov_path = snapshot_dirs[0]
        prov_data = json.loads(prov_path.read_text())
        snapshot_dir = prov_path.parent
        snapshot_date = prov_data["pull_date"]

        adapter = GenAIAgenticAdapter(snapshot_dir, snapshot_date)
        incidents_list = list(adapter.iter_incidents())

        click.echo(f"Loaded {len(incidents_list)} incidents from corpus")

        # Stage-1 classification
        result = _classify(tuple(incidents_list), rules)
        click.echo(f"Stage-1 produced {len(result.classifications)} classifications")

        # Stage-2 routing (if configured)
        stage2_results: tuple[Stage2Classification, ...] = ()
        if stage2_config is not None:
            all_ids = {inc.id for inc in incidents_list}
            low_confidence_ids = route_to_stage2(
                result.classifications, all_ids,
                confidence_threshold=confidence_threshold,
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

                client = HttpRunPodClient(
                    api_key=api_key,
                    endpoint_id=endpoint_id,
                    model_name=s2_manifest.model_identity,
                )
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
                s2_incidents = tuple(i for i in incidents_list if i.id in low_confidence_ids)
                rubric_hash = manifest_data.get("rubric_hash", "")
                total_s2 = len(s2_incidents)
                click.echo(f"Stage-2: classifying {total_s2} incidents via RunPod (concurrent)...")

                import concurrent.futures
                import threading

                s2_results_map: dict[int, Stage2Classification] = {}
                completed_count = 0
                lock = threading.Lock()

                def _classify_one(
                    idx_inc: tuple[int, IncidentRecord],
                ) -> tuple[int, Stage2Classification]:
                    idx, inc = idx_inc
                    return idx, classifier.classify(inc, rubric_hash)

                max_concurrent = 18  # 3 workers × 6 batch slots each
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
                    future_to_idx = {
                        pool.submit(_classify_one, (i, inc)): i
                        for i, inc in enumerate(s2_incidents)
                    }
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx, result_s2 = future.result()
                        s2_results_map[idx] = result_s2
                        with lock:
                            completed_count += 1
                            if completed_count % 100 == 0 or completed_count == total_s2:
                                click.echo(
                                    f"  Stage-2 progress: {completed_count}/{total_s2} "
                                    f"(${tracker.total_cost_usd:.2f})"
                                )

                stage2_results = tuple(s2_results_map[i] for i in range(total_s2))
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
        incident_strata = {inc.id: inc.corpus_stratum for inc in incidents_list}
        write_classify_artifacts(
            result, out_dir,
            stage2_results=stage2_results,
            incident_strata=incident_strata,
        )
        click.echo(f"Classify phase complete. Artifacts written to {out_dir}")
    except Exception as e:
        raise click.ClickException(f"Classify phase failed: {e}") from e


@click.command(name="infer-real")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--num-warmup", type=int, default=1000)
@click.option("--num-samples", type=int, default=2000)
@click.option("--timeout-seconds", type=float, default=None)
@click.option("--execute", is_flag=True, default=False,
              help="Execute inference (without flag, validates prerequisites only)")
@click.option("--wandb/--no-wandb", default=False, help="Enable WandB monitoring")
def infer_real(
    cycle: Path,
    num_warmup: int,
    num_samples: int,
    timeout_seconds: float | None,
    execute: bool,
    wandb: bool,
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
    cal_path = cycle / "calibration" / "posteriors.json"
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
        click.echo(
            "Infer phase: prerequisites satisfied."
            " Run with --execute to start NUTS inference."
        )
        return

    # Execute real inference pipeline
    click.echo("Executing infer phase...")
    try:
        from engine.cli.pipeline_executor import execute_infer_phase
        from engine.monitoring.wandb_logger import WandBLogger

        wandb_logger = WandBLogger.create(enabled=False)
        if wandb:
            try:
                from engine.cli.secrets import load_secret

                wandb_key = load_secret("wandb/api-key", env_var="WANDB_API_KEY")
                import os
                os.environ.setdefault("WANDB_API_KEY", wandb_key)
                wandb_logger = WandBLogger.create(
                    enabled=True,
                    cycle_id=str(cycle),
                    tags=["infer"],
                )
            except RuntimeError:
                click.echo("WandB credentials not found; continuing without monitoring")

        execute_infer_phase(
            cycle,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=4,
            wandb_logger=wandb_logger,
        )
        wandb_logger.finish()
        click.echo("Infer phase complete.")
    except Exception as e:
        raise click.ClickException(f"Infer phase failed: {e}") from e


@click.command(name="decide-real")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--vote-xlsx", required=True, type=click.Path(path_type=Path, exists=True),
              help="Path to vote results XLSX file")
@click.option("--execute", is_flag=True, default=False,
              help="Execute decision phase (without flag, validates prerequisites only)")
@click.option("--wandb/--no-wandb", default=False, help="Enable WandB monitoring")
def decide_real(cycle: Path, vote_xlsx: Path, execute: bool, wandb: bool) -> None:
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
        from engine.cli.pipeline_executor import _load_manifest, write_decide_artifacts
        from engine.decide.concordance import compute_concordance
        from engine.decide.selection_bias import compute_selection_bias
        from engine.model.inference import InferenceResult
        from engine.monitoring.wandb_logger import WandBLogger
        from engine.vote.bootstrap import bootstrap_vote_ranks
        from engine.vote.loader import load_vote_data

        wandb_logger = WandBLogger.create(enabled=False)
        if wandb:
            try:
                from engine.cli.secrets import load_secret

                wandb_key = load_secret("wandb/api-key", env_var="WANDB_API_KEY")
                import os
                os.environ.setdefault("WANDB_API_KEY", wandb_key)
                wandb_logger = WandBLogger.create(
                    enabled=True,
                    cycle_id=str(cycle),
                    tags=["decide"],
                )
            except RuntimeError:
                click.echo("WandB credentials not found; continuing without monitoring")

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

        # Load measurability verdicts from calibration diagnostic
        measurability_verdicts = _load_measurability_verdicts(
            cycle / "calibration", entry_ids,
        )
        measurable_ids = [
            e for e, v in measurability_verdicts.items()
            if v != "frame_blind_unmeasurable"
        ]
        measurable_count = len(measurable_ids)

        # Compute concordance with correct 8-parameter signature
        concordance = compute_concordance(
            inference_result=inference_result,
            vote_posterior=vote_posterior,
            tier_boundaries=_default_tier_boundaries(len(entry_ids)),
            flag_threshold_tau=manifest.flag_threshold_tau,
            measurable_count=measurable_count,
            total_count=len(entry_ids),
            meaningful_kappa_n=manifest.meaningful_kappa_n,
            measurability_minimum=manifest.measurability_minimum,
        )

        wandb_logger.log_concordance(
            kappa_median=concordance.weighted_kappa_median,
            kappa_ci=concordance.weighted_kappa_ci,
            measurable_count=concordance.measurable_count,
            total_count=concordance.total_count,
        )

        # Compute selection bias
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

        # Write rank comparison report
        from engine.decide.concordance import format_rank_comparison_report
        report_text = format_rank_comparison_report(concordance)
        report_path = out_dir / "rank_comparison_report.md"
        report_path.write_text(report_text)

        # Summary counts
        if concordance.entry_comparisons:
            actions = [c["action"] for c in concordance.entry_comparisons]
            click.echo(
                f"Rank comparison: {actions.count('confirmed')} confirmed, "
                f"{actions.count('note')} note, {actions.count('review')} review"
            )

        wandb_logger.finish()
        click.echo(f"Decide phase complete. Artifacts written to {out_dir}")
    except Exception as e:
        raise click.ClickException(f"Decide phase failed: {e}") from e


@click.command(name="report")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
def report_cmd(cycle: Path) -> None:
    """Generate final cycle report + reproduction bundle."""
    results_dir = cycle / "results"
    if not results_dir.exists():
        raise click.ClickException("results/ directory not found — run decide first")

    prereg = cycle / "prereg"
    infer_dir = cycle / "infer"

    try:
        from engine.cli.pipeline_executor import _load_manifest
        from engine.decide.concordance import ConcordanceResult
        from engine.decide.measurability import MeasurabilityMap
        from engine.decide.selection_bias import SelectionBiasDisclosure
        from engine.report.diff import compute_prereg_diff
        from engine.report.render import ReportInputs, render_report
        from engine.version import __version__

        manifest = _load_manifest(prereg / "manifest.json")

        concordance_path = results_dir / "concordance.json"
        if not concordance_path.exists():
            raise click.ClickException("concordance.json not found — run decide first")
        conc_data = json.loads(concordance_path.read_text())

        flags_raw = conc_data.get("flags", [])
        from engine.decide.robustness_multiplicity import FlagDirection, FlagFinding
        flags = tuple(
            FlagFinding(
                entry_id=f["entry_id"],
                probability=f["probability"],
                direction=FlagDirection(f["direction"]),
            )
            for f in flags_raw
        )

        concordance = ConcordanceResult(
            weighted_kappa_median=conc_data.get("weighted_kappa_median"),
            weighted_kappa_ci=(
                tuple(conc_data["weighted_kappa_ci"])
                if conc_data.get("weighted_kappa_ci") else None
            ),
            measurable_count=conc_data["measurable_count"],
            total_count=conc_data["total_count"],
            coverage_ratio=conc_data["coverage_ratio"],
            below_prereg_minimum=conc_data.get("below_prereg_minimum", False),
            meaningful_kappa_n=manifest.meaningful_kappa_n,
            flags=flags,
            standing_caveat="",
            ci_method=conc_data.get("ci_method", "paired_draw_percentile"),
        )

        sel_bias_path = results_dir / "selection_bias.json"
        sb_data = json.loads(sel_bias_path.read_text()) if sel_bias_path.exists() else {}
        selection_bias = SelectionBiasDisclosure(
            statistic_name=sb_data.get("statistic_name", "kruskal_wallis_h"),
            statistic_value=float(sb_data.get("statistic_value", float("nan"))),
            p_value=float(sb_data.get("p_value", float("nan"))),
            n_entries_per_group=sb_data.get("n_entries_per_group", {}),
            severity=sb_data.get("severity", "low"),
        )

        summary_path = infer_dir / "inference_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        entry_ids = tuple(summary.get("entry_ids", []))

        from engine.model.censoring import MeasurabilityVerdict
        verdicts = _load_measurability_verdicts(
            cycle / "calibration", entry_ids,
        )
        measurable_eids = tuple(
            e for e, v in verdicts.items() if v != "frame_blind_unmeasurable"
        )
        frame_blind_eids = tuple(
            e for e, v in verdicts.items() if v == "frame_blind_unmeasurable"
        )
        verdict_enum = {
            eid: (
                MeasurabilityVerdict.FRAME_BLIND_UNMEASURABLE
                if v == "frame_blind_unmeasurable"
                else MeasurabilityVerdict.MEASURABLE
            )
            for eid, v in verdicts.items()
        }
        meas_map = MeasurabilityMap(
            verdict=verdict_enum,
            recall_p_above_threshold={
                eid: (0.0 if eid in frame_blind_eids else 1.0)
                for eid in entry_ids
            },
            measurable=measurable_eids,
            classifier_blind=(),
            frame_blind=frame_blind_eids,
            coverage_ratio=concordance.coverage_ratio,
            below_prereg_minimum=concordance.below_prereg_minimum,
        )

        prereg_diff = compute_prereg_diff(
            prereg_primary_spec=manifest.primary_spec,
            actual_primary_spec=manifest.primary_spec,
            prereg_flag_tau=manifest.flag_threshold_tau,
            actual_flag_tau=manifest.flag_threshold_tau,
            prereg_measurability_min=manifest.measurability_minimum,
            actual_measurability_min=manifest.measurability_minimum,
        )

        s2_manifest_path = prereg / "stage2_manifest.json"
        runpod_cost = None
        cost_ceiling = None
        if s2_manifest_path.exists():
            s2_data = json.loads(s2_manifest_path.read_text())
            runpod_cost = s2_data.get("actual_cost_usd")
            cost_ceiling = s2_data.get("cost_ceiling_usd")

        corpus_b_corr = None
        cb_path = results_dir / "corpus_b_corroboration.json"
        if cb_path.exists():
            corpus_b_corr = json.loads(cb_path.read_text())

        inputs = ReportInputs(
            cycle_id=manifest.cycle_id,
            engine_version=__version__,
            measurability_map=meas_map,
            concordance=concordance,
            selection_bias=selection_bias,
            robustness=None,
            twin_agreement=None,
            non_publishable=True,
            prereg_diff=prereg_diff,
            runpod_cost_usd=runpod_cost,
            cost_ceiling_usd=cost_ceiling,
            corpus_b_corroboration=corpus_b_corr,
        )
        report_text = render_report(inputs)
        report_path = results_dir / "report.md"
        report_path.write_text(report_text)
        click.echo(f"Report written to {report_path}")
    except Exception as e:
        raise click.ClickException(f"Report generation failed: {e}") from e


@click.command(name="repro-bundle")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--output", required=True, type=click.Path(path_type=Path))
def repro_bundle_cmd(cycle: Path, output: Path) -> None:
    """Generate reproduction bundle tar.gz."""
    import hashlib
    import tarfile

    from engine.repro.bundle import ReproductionBundle
    from engine.snapshot.hashing import snapshot_hash
    from engine.version import __version__

    prereg = cycle / "prereg"
    if not (prereg / "manifest.json").exists():
        raise click.ClickException("prereg/manifest.json not found")

    manifest_hash = snapshot_hash(prereg / "manifest.json")
    lock_path = prereg / "manifest.lock"
    lockfile_hash = snapshot_hash(lock_path) if lock_path.exists() else "none"
    snap_path = prereg / "snapshot.json"
    snap_hash = snapshot_hash(snap_path) if snap_path.exists() else "none"

    provenance: dict[str, str] = {}
    s2_path = prereg / "stage2_manifest.json"
    if s2_path.exists():
        provenance["stage2_manifest_hash"] = snapshot_hash(s2_path)
    cal_path = cycle / "calibration" / "posteriors.json"
    if cal_path.exists():
        provenance["calibration_hash"] = snapshot_hash(cal_path)
    vote_path = cycle / "polling" / "vote_results.xlsx"
    if vote_path.exists():
        h = hashlib.sha256(vote_path.read_bytes()).hexdigest()
        provenance["vote_data_hash"] = h

    manifest_data = json.loads((prereg / "manifest.json").read_text())
    cycle_id = manifest_data.get("cycle_id", cycle.name)

    bundle = ReproductionBundle(
        cycle_id=cycle_id,
        engine_version=__version__,
        snapshot_hash=snap_hash,
        manifest_hash=manifest_hash,
        lockfile_hash=lockfile_hash,
        provenance=provenance,
    )

    bundle_json_path = cycle / "results" / "reproduction_bundle.json"
    bundle_json_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.write(bundle_json_path)

    with tarfile.open(output, "w:gz") as tar:
        for subdir in ("prereg", "classify", "infer", "results", "calibration", "taxonomy"):
            dir_path = cycle / subdir
            if dir_path.exists():
                tar.add(str(dir_path), arcname=subdir)
        tar.add(str(bundle_json_path), arcname="reproduction_bundle.json")

    click.echo(f"Reproduction bundle: {output} ({output.stat().st_size / 1024:.0f} KB)")


@click.command(name="corroborate")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--corpus-b-dir", required=True, type=click.Path(path_type=Path, exists=True),
              help="Path to vendored corpus B snapshot directory")
@click.option("--execute", is_flag=True, default=False,
              help="Execute corroboration (without flag, validates prerequisites only)")
def corroborate(cycle: Path, corpus_b_dir: Path, execute: bool) -> None:
    """Run corpus B corroboration cross-check (Plan 6).

    Classifies corpus B through Stage-1 (+ Stage-2 if available),
    detects incident overlap with corpus A, computes agreement,
    and writes the corroboration artifact.

    Corpus B is qualitative corroboration only — NEVER a posterior input.
    """
    prereg = cycle / "prereg"
    if not (prereg / "rubric.json").exists():
        raise click.ClickException("prereg/rubric.json not found — rubric must be frozen")

    classify_dir = cycle / "classify"
    corpus_a_labels_path = classify_dir / "labeled_incidents.json"
    if not corpus_a_labels_path.exists():
        raise click.ClickException(
            "classify/labeled_incidents.json not found — run classify first"
        )

    results_dir = cycle / "results"
    conc_path = results_dir / "concordance.json"
    if not conc_path.exists():
        raise click.ClickException(
            "results/concordance.json not found — run decide first"
        )

    click.echo(f"Corpus B corroboration: loading from {corpus_b_dir}")

    if not execute:
        click.echo(
            "Corroborate: prerequisites satisfied. "
            "Run with --execute to compute corroboration."
        )
        return

    click.echo("Executing corpus B corroboration...")
    try:
        from engine.adapters.owasp_asi import OWASPASIAdapter
        from engine.classify.classifier import build_rules_from_rubric, classify_real
        from engine.cli.pipeline_executor import _load_manifest
        from engine.decide.corpus_b_corroboration import (
            compute_agreement,
            detect_overlaps,
        )
        from engine.prereg.rubric_io import read_rubric

        rubric = read_rubric(prereg / "rubric.json")
        manifest = _load_manifest(prereg / "manifest.json")
        confidence_threshold = manifest.confidence_threshold

        adapter = OWASPASIAdapter(corpus_b_dir)
        corpus_b_incidents = list(adapter.iter_incidents())
        click.echo(f"Loaded {len(corpus_b_incidents)} corpus B incidents")

        rules = build_rules_from_rubric(rubric, confidence_threshold=confidence_threshold)
        b_result = classify_real(tuple(corpus_b_incidents), rules)
        click.echo(f"Stage-1 classified corpus B: {len(b_result.classifications)} classifications")

        classification_stages = "stage1"

        stage2_config = prereg / "stage2_manifest.json"
        if stage2_config.exists():
            from engine.cli.pipeline_executor import merge_classifications, route_to_stage2

            all_b_ids = {inc.id for inc in corpus_b_incidents}
            low_conf_ids = route_to_stage2(
                b_result.classifications, all_b_ids,
                confidence_threshold=confidence_threshold,
            )
            click.echo(f"Stage-2 candidates: {len(low_conf_ids)} corpus B incidents")

            if low_conf_ids:
                try:
                    import os

                    from engine.classify.cost_tracker import CostTracker
                    from engine.classify.runpod_client import HttpRunPodClient
                    from engine.classify.stage2 import Stage2Classifier
                    from engine.classify.stage2_manifest import Stage2Manifest
                    from engine.cli.secrets import load_secret

                    s2_manifest = Stage2Manifest.read(stage2_config)
                    api_key = load_secret("runpod/api-key", env_var="RUNPOD_API_KEY")
                    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")

                    client = HttpRunPodClient(
                        api_key=api_key,
                        endpoint_id=endpoint_id,
                        model_name=s2_manifest.model_identity,
                    )
                    tracker = CostTracker(ceiling_usd=s2_manifest.cost_ceiling_usd)
                    classifier = Stage2Classifier(
                        client=client,
                        cost_tracker=tracker,
                        rubric_json=(prereg / "rubric.json").read_text(),
                        model_identity=s2_manifest.model_identity,
                        weight_provenance_hash=s2_manifest.weight_provenance_hash,
                        prng_seed=s2_manifest.prng_seed,
                    )

                    s2_incidents = tuple(i for i in corpus_b_incidents if i.id in low_conf_ids)
                    rubric_hash = manifest.rubric_hash or ""
                    click.echo(f"Stage-2: classifying {len(s2_incidents)} corpus B incidents...")

                    s2_results = tuple(
                        classifier.classify(inc, rubric_hash) for inc in s2_incidents
                    )
                    client.close()

                    merged = merge_classifications(
                        b_result.classifications, s2_results, confidence_threshold,
                    )
                    from engine.classify.stub import ClassificationResult
                    b_result = ClassificationResult(
                        classifications=merged,
                        classifier_version=b_result.classifier_version,
                        classifier_rule_hash=b_result.classifier_rule_hash,
                    )
                    classification_stages = "stage1+stage2"
                    click.echo(f"Stage-2 complete for corpus B ({len(s2_results)} results)")
                except (RuntimeError, OSError) as exc:
                    click.echo(
                        f"Stage-2 unavailable for corpus B ({exc}); "
                        f"proceeding with Stage-1 only"
                    )

        b_labeled = [
            {
                "incident_id": c.incident_id,
                "entry_id": c.entry_id,
                "confidence": c.confidence,
                "stage": c.stage,
                "rationale": c.rationale,
                "stratum": "corroboration",
            }
            for c in b_result.classifications
        ]
        b_labeled_path = classify_dir / "corpus_b_labeled.json"
        b_labeled_path.write_text(json.dumps(b_labeled, indent=2) + "\n")
        click.echo(f"Corpus B classifications written to {b_labeled_path}")

        a_labels_raw = json.loads(corpus_a_labels_path.read_text())
        a_label_map: dict[str, str] = {}
        a_label_conf: dict[str, float] = {}
        for rec in a_labels_raw:
            iid = rec["incident_id"]
            conf = rec["confidence"]
            if iid not in a_label_map or conf > a_label_conf.get(iid, -1.0):
                a_label_map[iid] = rec["entry_id"]
                a_label_conf[iid] = conf

        b_label_map: dict[str, str] = {}
        b_label_conf: dict[str, float] = {}
        for c in b_result.classifications:
            prev_conf = b_label_conf.get(c.incident_id, -1.0)
            if c.incident_id not in b_label_map or c.confidence > prev_conf:
                b_label_map[c.incident_id] = c.entry_id
                b_label_conf[c.incident_id] = c.confidence

        snapshot_dirs = list((cycle / "corpora" / "genai_agentic").iterdir())
        if not snapshot_dirs:
            raise click.ClickException("No corpus A snapshot found")
        from engine.adapters.genai_agentic import GenAIAgenticAdapter
        corpus_a_adapter = GenAIAgenticAdapter(snapshot_dirs[0], "2099-12-31")
        corpus_a_incidents = list(corpus_a_adapter.iter_incidents())

        overlaps = detect_overlaps(corpus_a_incidents, corpus_b_incidents)
        click.echo(f"Detected {len(overlaps)} incident overlaps between corpora")

        conc_data = json.loads(conc_path.read_text())
        baseline_kappa = conc_data.get("weighted_kappa_median", 0.0) or 0.0

        b_records_map = {inc.id: inc for inc in corpus_b_incidents}
        corroboration = compute_agreement(
            overlaps=overlaps,
            corpus_a_labels=a_label_map,
            corpus_b_labels=b_label_map,
            corpus_b_records=b_records_map,
            baseline_kappa=baseline_kappa,
            corpus_a_count=len(corpus_a_incidents),
            corpus_b_count=len(corpus_b_incidents),
            classification_stages=classification_stages,
        )

        results_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "corpus_b_incident_count": corroboration.corpus_b_incident_count,
            "corpus_a_incident_count": corroboration.corpus_a_incident_count,
            "overlap_count": corroboration.overlap_count,
            "classification_stages_used": corroboration.classification_stages_used,
            "agreement_count": corroboration.agreement_count,
            "disagreement_count": corroboration.disagreement_count,
            "agreement_rate": corroboration.agreement_rate,
            "baseline_kappa": corroboration.baseline_kappa,
            "overlap_method_limitations": list(corroboration.overlap_method_limitations),
            "per_incident": [
                {
                    "corpus_a_id": a.corpus_a_id,
                    "corpus_b_id": a.corpus_b_id,
                    "corpus_b_title": a.corpus_b_title,
                    "match_method": a.match_method,
                    "corpus_a_label": a.corpus_a_label,
                    "corpus_b_label": a.corpus_b_label,
                    "corpus_b_native_labels": list(a.corpus_b_native_labels),
                    "agrees": a.agrees,
                }
                for a in corroboration.per_incident
            ],
            "systematic_divergences": [
                {
                    "pattern": d.pattern,
                    "count": d.count,
                    "incidents": list(d.incidents),
                }
                for d in corroboration.systematic_divergences
            ],
        }
        artifact_path = results_dir / "corpus_b_corroboration.json"
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
        click.echo(f"Corroboration artifact written to {artifact_path}")
        click.echo(
            f"Result: {corroboration.overlap_count} shared incidents, "
            f"{corroboration.agreement_count} agree, "
            f"{corroboration.disagreement_count} disagree "
            f"(rate={corroboration.agreement_rate:.2f})"
        )
        if corroboration.systematic_divergences:
            click.echo("Systematic divergences detected:")
            for d in corroboration.systematic_divergences:
                click.echo(f"  {d.pattern} ({d.count} incidents)")

    except Exception as e:
        raise click.ClickException(f"Corroboration failed: {e}") from e


@click.command(name="report-narrative")
@click.option(
    "--cycle-dir",
    type=click.Path(exists=True, path_type=Path, resolve_path=True),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
)
def report_narrative_cmd(cycle_dir: Path, output_dir: Path | None) -> None:
    """Generate standalone narrative report with figures."""
    from engine.report.narrative import generate_narrative_report

    if output_dir is None:
        output_dir = Path("notebooks") / "narrative"
    result_path = generate_narrative_report(cycle_dir, output_dir)
    click.echo(f"Narrative report written to {result_path}")
