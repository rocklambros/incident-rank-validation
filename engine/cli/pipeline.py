"""Real-data pipeline CLI commands for Plan 5.

These commands wire the existing engine modules into a production pipeline
for the 2026 LLM Top 10 cycle.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau

from engine.vote.plackett_luce import (
    DEFAULT_RIDGE,
    N_BOOTSTRAP_DEFAULT,
    bootstrap_davidson,
    fit_davidson,
)

if TYPE_CHECKING:
    from engine.classify.stage2_protocol import Stage2Classification
    from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult
    from engine.schema import IncidentRecord


def assert_robustness_complete(manifest: object, spread: RobustnessSpread) -> None:
    """Refuse a report whose declared robustness specs were not all run (Plan 8a, SD4).

    The pipeline declares the robustness specs it intends to run in the manifest.
    A report that silently dropped a declared spec (NUTS crash, persistence gap)
    would understate the cherry-pick risk, so we fail hard if any are missing.

    Also refuses a name-complete-but-null-kappa decoy spread (U2-8): every
    declared spec present in the spread must carry a finite weighted_kappa_median.
    """
    declared = set(getattr(manifest, "robustness_specs", ()))
    present = {s.spec_name for s in spread.robustness}
    missing = declared - present
    if missing:
        raise ValueError(
            f"declared robustness specs not run: {sorted(missing)}"
        )
    # Content-validate: each declared spec must have a finite kappa (U2-8).
    # A null or non-finite value means the decide phase did not produce a
    # valid result for this spec — refuse the report rather than silently
    # propagating a decoy spread.
    spec_by_name = {s.spec_name: s for s in spread.robustness}
    for spec_name in sorted(declared):
        spec = spec_by_name.get(spec_name)
        if spec is None:
            continue  # already caught by the missing check above
        kappa = spec.weighted_kappa_median
        if kappa is None or not math.isfinite(kappa):
            raise ValueError(
                f"robustness spec {spec_name!r} has non-finite "
                f"weighted_kappa_median ({kappa!r}); re-run decide to regenerate spread"
            )


def build_robustness_spread(
    primary_spec_result: SpecResult,
    robustness_results: tuple[SpecResult, ...],
) -> RobustnessSpread:
    """Assemble the primary + robustness SpecResults into a RobustnessSpread.

    Kept as a thin, importable seam so Plans 8b/8c can populate sigma_u /
    extra_rankings on the SpecResults they construct without re-implementing
    assembly. The completeness gate is applied separately by the caller via
    assert_robustness_complete().
    """
    from engine.decide.robustness_multiplicity import RobustnessSpread

    return RobustnessSpread(
        primary=primary_spec_result,
        robustness=robustness_results,
    )


def build_vote_pl_summary(
    rankings: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    mean_rank_ranking: tuple[str, ...],
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    ridge: float = DEFAULT_RIDGE,
) -> dict[str, object]:
    """Fit the Davidson tie-aware vote model and assemble JSON diagnostics.

    Computes the worth ranking (with ties), the drop-ties Bradley-Terry
    sensitivity, and the respondent bootstrap, plus Kendall-tau concordance
    against the vote's median-rank point summary (the mean-rank vote ranking) and
    against the drop-ties ranking.  The returned dict is the auditable
    ``vote_plackett_luce.json`` payload; its ``"ranking"`` is also attached to
    ``SpecResult.extra_rankings["plackett_luce"]`` by the caller.
    """
    from engine.vote.plackett_luce import _ranking_to_rank_vector

    fit = fit_davidson(rankings, entry_ids, ridge=ridge, include_ties=True)
    fit_drop = fit_davidson(rankings, entry_ids, ridge=ridge, include_ties=False)
    post = bootstrap_davidson(
        rankings, entry_ids, n_bootstrap=n_bootstrap, seed=seed, ridge=ridge
    )

    pl_vec = _ranking_to_rank_vector(fit.ranking, entry_ids)
    mean_vec = _ranking_to_rank_vector(mean_rank_ranking, entry_ids)
    drop_vec = _ranking_to_rank_vector(fit_drop.ranking, entry_ids)
    def _finite(value: float) -> float | None:
        return value if math.isfinite(value) else None

    tau_meanrank = _finite(float(kendalltau(pl_vec, mean_vec)[0]))
    tau_dropties = _finite(float(kendalltau(pl_vec, drop_vec)[0]))
    boot_stability = _finite(post.mean_kendall_tau_vs_point)

    return {
        "model": "davidson_tie_aware",
        "ridge": ridge,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_respondents": int(rankings.shape[0]),
        "entries": list(entry_ids),
        "worths": fit.worths,
        "tie_param": fit.tie_param,
        "ranking": list(fit.ranking),
        "ranking_drop_ties": list(fit_drop.ranking),
        "bootstrap_median_ranks": post.median_ranks,
        "bootstrap_top5_frequency": post.top5_frequency,
        "mean_kendall_tau_vs_point": boot_stability,
        "kendall_tau_vs_meanrank": tau_meanrank,
        "kendall_tau_withties_vs_dropties": tau_dropties,
        "converged": fit.converged,
        "converged_drop_ties": fit_drop.converged,
        "n_nonconverged_bootstrap": post.n_nonconverged,
    }


def build_incidence_ranking_artifact(
    lambda_samples: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    common: list[str],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> dict[str, object]:
    """Persist the engine's incidence deliverable for the oracle to check (8d).

    Uses the engine's OWN _ranks_from_incidence on the median lambda vector so
    the persisted ranking is exactly the engine's method; the oracle re-derives
    independently and compares.
    """
    from engine.decide.concordance import _ranks_from_incidence

    inf_idx = {e: i for i, e in enumerate(entry_ids)}
    median_lambda = np.median(lambda_samples, axis=0)
    point_ranks = _ranks_from_incidence(
        median_lambda, inf_idx, common, entry_strata, stratum_sizes
    )
    ranking = [e for _, e in sorted(zip(point_ranks, common, strict=True))]

    incidence_median: dict[str, float] = {}
    incidence_ci: dict[str, list[float]] = {}
    for e in common:
        total_size = float(sum(stratum_sizes[s] for s in entry_strata[e]))
        draws = lambda_samples[:, inf_idx[e]] * total_size
        incidence_median[e] = float(np.median(draws))
        incidence_ci[e] = [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ]
    return {
        "ranking": ranking,
        "incidence_median": incidence_median,
        "incidence_ci": incidence_ci,
    }


def _load_robustness_spread(path: Path) -> RobustnessSpread | None:
    """Reload a persisted RobustnessSpread (Plan 8a Task 6).

    Returns None if no spread was persisted (e.g. a decide run from before this
    artifact existed), so older cycles still render. Cycles that ran decide on
    this engine always persist a spread, even when robustness_specs=().
    """
    if not path.exists():
        return None
    from engine.decide.robustness_multiplicity import (
        FlagDirection,
        FlagFinding,
        RobustnessSpread,
        SpecResult,
    )

    data: dict[str, Any] = json.loads(path.read_text())

    def _to_spec(d: dict[str, Any]) -> SpecResult:
        ci = d.get("weighted_kappa_ci")
        rankings = d.get("extra_rankings")
        return SpecResult(
            spec_name=str(d["spec_name"]),
            weighted_kappa_median=d.get("weighted_kappa_median"),
            weighted_kappa_ci=tuple(ci) if ci else None,
            flags=tuple(
                FlagFinding(
                    entry_id=f["entry_id"],
                    probability=f["probability"],
                    direction=FlagDirection(f["direction"]),
                )
                for f in d.get("flags", [])
            ),
            sigma_u=d.get("sigma_u"),
            extra_rankings=(
                {k: tuple(v) for k, v in rankings.items()}
                if rankings else None
            ),
        )

    return RobustnessSpread(
        primary=_to_spec(data["primary"]),
        robustness=tuple(_to_spec(s) for s in data.get("robustness", [])),
    )


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
        from engine.calibrate.coverage import read_snapshot_universe_ids, write_classify_coverage
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
        write_classify_coverage(
            out_dir,
            snapshot_hash=manifest_data.get("snapshot_hash", ""),
            corpus_incident_ids=read_snapshot_universe_ids(snapshot_dir / "incidents.json"),
            in_scope_incident_ids={c.incident_id for c in result.classifications},
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
        lambda_samples = np.load(lambda_samples_path, allow_pickle=False)
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

        # Build entry_strata and stratum_sizes from labeled incidents
        from engine.cli.pipeline_executor import _build_counts_from_labeled
        classify_dir = cycle / "classify"
        labeled_incidents_path = classify_dir / "labeled_incidents.json"
        labeled_incidents = json.loads(labeled_incidents_path.read_text())
        _obs_counts, stratum_sizes, _, _ = _build_counts_from_labeled(labeled_incidents)
        # Build entry_strata: each entry maps to the tuple of strata it appears in
        from collections import defaultdict as _defaultdict
        _entry_strata_sets: dict[str, set[str]] = _defaultdict(set)
        for (eid, s) in _obs_counts:
            _entry_strata_sets[eid].add(s)
        entry_strata: dict[str, tuple[str, ...]] = {
            e: tuple(sorted(ss)) for e, ss in _entry_strata_sets.items()
        }

        # F8 guard: assert strata populations are disjoint before incidence ranking.
        from engine.verify.strata_guard import check_strata_disjoint as _check_strata
        _check_strata(labeled_incidents, entry_strata)

        # Compute concordance
        concordance = compute_concordance(
            inference_result=inference_result,
            vote_posterior=vote_posterior,
            tier_boundaries=_default_tier_boundaries(len(entry_ids)),
            flag_threshold_tau=manifest.flag_threshold_tau,
            measurable_count=measurable_count,
            total_count=len(entry_ids),
            meaningful_kappa_n=manifest.meaningful_kappa_n,
            measurability_minimum=manifest.measurability_minimum,
            entry_strata=entry_strata,
            stratum_sizes=stratum_sizes,
        )

        wandb_logger.log_concordance(
            kappa_median=concordance.weighted_kappa_median,
            kappa_ci=concordance.weighted_kappa_ci,
            measurable_count=concordance.measurable_count,
            total_count=concordance.total_count,
        )

        # Plan 8a Task 6: assemble the robustness spread from the per-spec
        # NUTS outputs the infer phase persisted, compute per-spec concordance
        # (reusing the SAME entry_strata/stratum_sizes as the primary), and
        # refuse the cycle if a declared spec was not run. Synthetic/parity
        # cycles declare robustness_specs=() so this is inert there.
        from engine.decide.robustness_multiplicity import SpecResult

        # Plan 8c: Davidson tie-aware vote model as a vote-side robustness lens.
        # The bootstrap mean-rank ordering (ascending median rank, 1 = best) is
        # the primary vote ranking we compare the worth ranking against.
        mean_rank_ranking = tuple(
            sorted(
                vote_posterior.entries,
                key=lambda e: (vote_posterior.median_ranks[e], e),
            )
        )
        vote_pl_summary = build_vote_pl_summary(
            vote_data.rankings,
            vote_data.entry_ids,
            mean_rank_ranking=mean_rank_ranking,
            seed=manifest.prng_seed,
        )
        pl_ranking = tuple(str(e) for e in vote_pl_summary["ranking"])  # type: ignore[attr-defined]

        primary_spec_result = SpecResult(
            spec_name=manifest.primary_spec,
            weighted_kappa_median=concordance.weighted_kappa_median,
            weighted_kappa_ci=concordance.weighted_kappa_ci,
            flags=concordance.flags,
            extra_rankings={"plackett_luce": pl_ranking},
        )
        robustness_results: list[SpecResult] = []
        for spec_name in manifest.robustness_specs:
            r_lambda_path = infer_dir / f"robustness_{spec_name}_lambda.npy"
            r_summary_path = infer_dir / f"robustness_{spec_name}_summary.json"
            if not r_lambda_path.exists() or not r_summary_path.exists():
                continue  # gate below raises with the missing-spec list
            r_summary = json.loads(r_summary_path.read_text())
            r_inference = InferenceResult(
                lambda_samples=np.load(r_lambda_path, allow_pickle=False),
                entry_ids=tuple(r_summary.get("entry_ids", [])),
                r_hat=r_summary.get("r_hat", {}),
                ess=r_summary.get("ess", {}),
                divergences=r_summary.get("divergences", 0),
                num_warmup=r_summary.get("num_warmup", 1000),
                num_samples=r_summary.get("num_samples", 2000),
                sigma_u=r_summary.get("sigma_u"),
            )
            r_concordance = compute_concordance(
                inference_result=r_inference,
                vote_posterior=vote_posterior,
                tier_boundaries=_default_tier_boundaries(len(entry_ids)),
                flag_threshold_tau=manifest.flag_threshold_tau,
                measurable_count=measurable_count,
                total_count=len(entry_ids),
                meaningful_kappa_n=manifest.meaningful_kappa_n,
                measurability_minimum=manifest.measurability_minimum,
                entry_strata=entry_strata,
                stratum_sizes=stratum_sizes,
            )
            robustness_results.append(SpecResult(
                spec_name=spec_name,
                weighted_kappa_median=r_concordance.weighted_kappa_median,
                weighted_kappa_ci=r_concordance.weighted_kappa_ci,
                flags=r_concordance.flags,
                sigma_u=r_inference.sigma_u,
            ))
        spread = build_robustness_spread(
            primary_spec_result, tuple(robustness_results),
        )
        assert_robustness_complete(manifest, spread)

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
            robustness=spread,
        )
        (out_dir / "vote_plackett_luce.json").write_text(
            json.dumps(vote_pl_summary, indent=2, sort_keys=True)
        )

        # Plan 8d: persist the engine deliverables the oracle checks against,
        # and the ballot matrix so re-verification is self-contained.
        _common = [e for e in inference_result.entry_ids if e in set(vote_posterior.entries)]
        incidence_artifact = build_incidence_ranking_artifact(
            inference_result.lambda_samples,
            inference_result.entry_ids,
            _common,
            entry_strata,
            stratum_sizes,
        )
        (out_dir / "incidence_ranking.json").write_text(
            json.dumps(incidence_artifact, indent=2, sort_keys=True)
        )
        np.save(out_dir / "vote_rankings.npy", vote_data.rankings)
        (out_dir / "vote_entry_ids.json").write_text(
            json.dumps(list(vote_data.entry_ids), indent=2)
        )

        # Plan 8d: run the independent oracle and persist its verdict. A
        # verification crash must NOT invalidate a completed decide (artifacts
        # are already written); report it and let decide succeed.
        from engine.verify.check import run_oracle
        try:
            oracle_verdict_obj = run_oracle(cycle)
        except Exception as oracle_err:  # noqa: BLE001
            click.echo(
                f"Oracle consistency check could not run ({oracle_err}); decide "
                f"artifacts are intact. Run `verify-oracle --cycle {cycle}` to retry."
            )
        else:
            n_skip = sum(
                1 for d in oracle_verdict_obj.deliverables if d.status == "SKIP"
            )
            n_checked = len(oracle_verdict_obj.deliverables) - n_skip
            if oracle_verdict_obj.provisional:
                click.echo(
                    f"Oracle consistency check: PROVISIONAL ({n_checked} checked, "
                    f"{n_skip} skipped; one or more deliverables disagree)"
                )
            else:
                click.echo(
                    f"Oracle consistency check: PASS ({n_checked} checked, "
                    f"{n_skip} skipped)"
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

        # Plan 8a Task 6: compare the DECLARED primary spec against the spec the
        # infer phase actually executed (recorded in inference_summary.json), so
        # the drift-diff is honest rather than comparing the declared literal to
        # itself. The primary is unchanged this cycle, so they coincide — but a
        # silent fallback to a different model would now surface as a deviation.
        actual_primary_spec = summary.get("primary_spec", manifest.primary_spec)
        prereg_diff = compute_prereg_diff(
            prereg_primary_spec=manifest.primary_spec,
            actual_primary_spec=actual_primary_spec,
            prereg_flag_tau=manifest.flag_threshold_tau,
            actual_flag_tau=manifest.flag_threshold_tau,
            prereg_measurability_min=manifest.measurability_minimum,
            actual_measurability_min=manifest.measurability_minimum,
        )

        # U2-8: load the robustness spread the decide phase persisted.
        # Grandfather clause: the "declared but missing" error fires ONLY for
        # schema_version >= 3.  Locked v1/v2 cycles (e.g. the committed 2026
        # cycle, schema_version=1) declare robustness_specs but never wrote a
        # spread and must still regenerate their report.
        # Content-validation (assert_robustness_complete) runs whenever the
        # spread IS present, regardless of schema_version: a null/non-finite
        # weighted_kappa_median is always refused.
        robustness_spread = _load_robustness_spread(
            results_dir / "robustness_spread.json",
        )
        _declared_specs = set(getattr(manifest, "robustness_specs", ()))
        _schema_version = getattr(manifest, "schema_version", 1)
        if robustness_spread is None and _declared_specs and _schema_version >= 3:
            raise ValueError(
                "robustness_specs declared in manifest but robustness_spread.json "
                "not found — re-run decide phase to generate it"
            )
        if robustness_spread is not None:
            assert_robustness_complete(manifest, robustness_spread)

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

        vote_pl_path = results_dir / "vote_plackett_luce.json"
        vote_pl_summary: dict[str, object] | None = None
        if vote_pl_path.exists():
            vote_pl_summary = json.loads(vote_pl_path.read_text())

        oracle_path = results_dir / "oracle_report.json"
        oracle_verdict: dict[str, object] | None = None
        if oracle_path.exists():
            oracle_verdict = json.loads(oracle_path.read_text())

        inputs = ReportInputs(
            cycle_id=manifest.cycle_id,
            engine_version=__version__,
            measurability_map=meas_map,
            concordance=concordance,
            selection_bias=selection_bias,
            robustness=robustness_spread,
            twin_agreement=None,
            non_publishable=True,
            prereg_diff=prereg_diff,
            runpod_cost_usd=runpod_cost,
            cost_ceiling_usd=cost_ceiling,
            corpus_b_corroboration=corpus_b_corr,
            vote_plackett_luce=vote_pl_summary,
            oracle_verdict=oracle_verdict,
        )
        report_text = render_report(inputs)
        report_path = results_dir / "report.md"
        report_path.write_text(report_text)
        click.echo(f"Report written to {report_path}")
    except Exception as e:
        raise click.ClickException(f"Report generation failed: {e}") from e


@click.command("verify-oracle")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
def verify_oracle_cmd(cycle: Path) -> None:
    """Run the independent consistency-check oracle over a completed cycle."""
    from engine.verify.check import run_oracle

    verdict = run_oracle(cycle)
    for d in verdict.deliverables:
        click.echo(f"[{d.status}] {d.name}: {d.metric} ; {d.detail}")
    click.echo(f"PROVISIONAL: {verdict.provisional}")


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

    # Resolve goldset_hash: infer/goldset_hash.txt → manifest.goldset_hash → "".
    # When BOTH are non-empty the manifest is authoritative; a mismatch means the
    # scored goldset differs from the pre-registered one — a provenance break.
    _infer_hash_path = cycle / "infer" / "goldset_hash.txt"
    _file_goldset_hash = (
        _infer_hash_path.read_text().strip()
        if _infer_hash_path.exists()
        else ""
    )
    _manifest_goldset_hash = manifest_data.get("goldset_hash") or ""
    if (
        _file_goldset_hash
        and _manifest_goldset_hash
        and _file_goldset_hash != _manifest_goldset_hash
    ):
        raise click.ClickException(
            f"goldset_hash provenance break: infer/goldset_hash.txt has "
            f"{_file_goldset_hash!r} but manifest.goldset_hash is "
            f"{_manifest_goldset_hash!r}; the scored goldset differs from "
            f"the pre-registered one."
        )
    goldset_hash = _file_goldset_hash or _manifest_goldset_hash or ""

    bundle = ReproductionBundle(
        cycle_id=cycle_id,
        engine_version=__version__,
        snapshot_hash=snap_hash,
        manifest_hash=manifest_hash,
        lockfile_hash=lockfile_hash,
        goldset_hash=goldset_hash,
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
