"""Synthetic end-to-end pipeline: full cycle from adapter through report.

Orchestrates all pipeline stages for a synthetic validation cycle,
including the M3 stratum-size sanity check.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tomli

from engine.adapters.base import CorpusAdapter
from engine.adapters.synthetic import SyntheticAdapter
from engine.adapters.synthetic_stress import SyntheticStressAdapter
from engine.calibrate.beta import BetaPosterior, Calibration
from engine.classify.stub import classify_stub
from engine.decide.concordance import compute_concordance
from engine.decide.measurability import build_measurability_map
from engine.decide.selection_bias import compute_selection_bias
from engine.decide.twin_agreement import compare_twin_nuts
from engine.model.censoring import partition_entries
from engine.model.inference import DiagnosticsFailure, InferenceResult, run_inference
from engine.model.twin import compute_twin
from engine.prereg.lock import write_lock
from engine.prereg.manifest import PreregManifest
from engine.report.render import ReportInputs, render_report
from engine.schema import IncidentRecord
from engine.snapshot.hashing import snapshot_hash
from engine.version import __version__
from engine.vote.bootstrap import bootstrap_vote_ranks


def _load_project_config(cycle: Path) -> dict[str, Any]:
    project_toml = cycle.parents[1] / "project.toml"
    with open(project_toml, "rb") as f:
        return tomli.load(f)


def _build_uniform_calibration(
    entry_ids: tuple[str, ...],
    strata: tuple[str, ...],
) -> Calibration:
    """Beta(1,1) uniform priors: no gold-set information."""
    uniform = BetaPosterior(alpha=1.0, beta=1.0)
    recall: dict[tuple[str, str], BetaPosterior] = {}
    precision: dict[tuple[str, str], BetaPosterior] = {}
    for eid in entry_ids:
        for s in strata:
            recall[(eid, s)] = uniform
            precision[(eid, s)] = uniform
    return Calibration(recall=recall, precision=precision)


def _stratum_size_sanity_check(
    adapter: CorpusAdapter,
    incidents: tuple[IncidentRecord, ...],
) -> None:
    """M3: stratum_size must be >= observed count per stratum."""
    stratum_sizes = adapter.stratum_sizes()
    actual_counts: dict[str, int] = {}
    for inc in incidents:
        actual_counts[inc.corpus_stratum] = actual_counts.get(inc.corpus_stratum, 0) + 1
    for s, declared in stratum_sizes.items():
        actual = actual_counts.get(s, 0)
        if int(declared) < actual:
            raise ValueError(
                f"stratum_size[{s!r}] = {declared} but adapter emitted {actual} incidents; "
                "stratum_size is the EXPOSURE term, must be >= observed count"
            )


def _generate_synthetic_votes(
    entry_ids: tuple[str, ...],
    ground_truth_ranks: list[str],
    prng_seed: int,
    n_respondents: int = 50,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Generate synthetic respondent rankings reflecting ground-truth ordering.

    Each respondent's ranking is the ground-truth ordering plus noise,
    so the average ranking follows the true prevalence order.
    """
    rng = np.random.default_rng(prng_seed)
    n_entries = len(entry_ids)
    gt_order = {eid: i for i, eid in enumerate(ground_truth_ranks)}

    rankings = np.zeros((n_respondents, n_entries), dtype=np.float64)
    for r in range(n_respondents):
        # Base rank from ground truth + Gaussian noise
        scores = np.array([float(gt_order[eid]) for eid in entry_ids])
        scores += rng.normal(0, 0.5, size=n_entries)
        order = np.argsort(scores)
        rank_vec = np.empty(n_entries, dtype=np.float64)
        rank_vec[order] = np.arange(1, n_entries + 1, dtype=np.float64)
        rankings[r] = rank_vec

    return rankings


def execute_synthetic_pipeline(
    cycle: Path,
    corpus_mode: str,
    with_signoff: bool = False,
    adapter_factory: Callable[..., CorpusAdapter] = SyntheticAdapter,
    manifest_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run the full synthetic validation cycle end-to-end."""
    # 1. Load project config
    config = _load_project_config(cycle)
    proj = config["project"]
    hyper = proj["hyperparameters"]
    prng_seed: int = proj["prng_seed"]
    cycle_id: str = proj["cycle_id"]
    measurability_minimum: int = proj["measurability_minimum"]
    flag_threshold_tau: float = hyper.get("flag_threshold_tau", 0.1)
    meaningful_kappa_n: int = hyper["meaningful_kappa_n"]

    # 2. Build adapter — select based on project name if caller didn't override
    _ADAPTER_REGISTRY: dict[str, Callable[..., CorpusAdapter]] = {
        "synthetic": SyntheticAdapter,
        "synthetic-stress": SyntheticStressAdapter,
    }
    if adapter_factory is SyntheticAdapter:
        project_name: str = proj["name"]
        resolved_factory = _ADAPTER_REGISTRY.get(project_name, adapter_factory)
    else:
        resolved_factory = adapter_factory
    adapter = resolved_factory(seed=prng_seed)

    # 3. Gather incidents
    incidents = tuple(adapter.iter_incidents())
    entries = adapter.entry_definitions()
    entry_ids = tuple(e.entry_id for e in entries)
    strata = tuple(sorted({inc.corpus_stratum for inc in incidents}))
    stratum_sizes_raw = adapter.stratum_sizes()
    stratum_sizes_int = {s: int(v) for s, v in stratum_sizes_raw.items()}
    overlap = adapter.overlap_weights()

    # 4. M3 stratum-size sanity check
    _stratum_size_sanity_check(adapter, incidents)

    # 5. Classify
    classification = classify_stub(incidents, entry_ids)
    incidents_by_id = {inc.id: inc for inc in incidents}
    observed_counts = classification.counts_by_entry_stratum(incidents_by_id)

    # 6-7. Build calibration (uniform Beta(1,1) for synthetic)
    non_frame_blind_ids = tuple(e.entry_id for e in entries if not e.frame_blind)
    calibration = _build_uniform_calibration(non_frame_blind_ids, strata)

    # 8. Partition entries
    censoring = partition_entries(entries, calibration=None)

    # 9. Build measurability map
    meas_map = build_measurability_map(
        censoring, calibration=None, measurability_minimum=measurability_minimum
    )

    # 10. Write coverage.json
    results_dir = cycle / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    meas_map.write_coverage(results_dir / "coverage.json")

    # 11. Build PreregManifest
    taxonomy_path = cycle / "taxonomy" / "taxonomy.json"
    taxonomy_hash_val = snapshot_hash(taxonomy_path) if taxonomy_path.exists() else "none"

    manifest = PreregManifest(
        engine_version=__version__,
        engine_version_range_min=__version__,
        engine_version_range_max=__version__,
        cycle_id=cycle_id,
        taxonomy_hash=taxonomy_hash_val,
        snapshot_hash="synthetic-no-snapshot",
        primary_spec="negative_binomial_per_stratum",
        robustness_specs=(),
        flag_threshold_tau=flag_threshold_tau,
        statistic="weighted_cohens_kappa",
        measurability_minimum=measurability_minimum,
        prior_scale=hyper["prior_scale"],
        concentration_shape=hyper["concentration_shape"],
        concentration_rate=hyper["concentration_rate"],
        ess_fraction=hyper["ess_fraction"],
        meaningful_kappa_n=meaningful_kappa_n,
        prng_seed=prng_seed,
        confidence_threshold=hyper.get("confidence_threshold", 0.3),
        rubric_drafting_attestation=None,
        rubric_reviewer=None,
        statistical_reviewer=None,
        classifier_rule_hash=classification.classifier_rule_hash,
        rubric_hash=None,
        post_hoc_register_path=None,
        **(manifest_kwargs or {}),
    )

    # 12. Write prereg lock
    prereg_dir = cycle / "prereg"
    prereg_dir.mkdir(parents=True, exist_ok=True)
    write_lock(manifest, prereg_dir / "prereg.lock.json")

    # 13. Run NUTS inference (reduced warmup/samples for synthetic)
    inference_result: InferenceResult | None = None
    nuts_failed = False
    try:
        inference_result = run_inference(
            manifest,
            censoring.measurable,
            strata,
            observed_counts,
            stratum_sizes_int,
            calibration,
            overlap,
            num_warmup=200,
            num_samples=500,
        )
    except DiagnosticsFailure as exc:
        nuts_failed = True
        notice_path = results_dir / "diagnostics_failure.txt"
        notice_path.write_text(f"NUTS diagnostics failed: {exc}\n")

    # Build ground-truth ranking from observed counts for synthetic votes
    entry_totals = {
        eid: sum(observed_counts.get((eid, s), 0) for s in strata)
        for eid in censoring.measurable
    }
    gt_ranking = sorted(entry_totals, key=lambda e: entry_totals[e], reverse=True)

    # 14-19: Steps that depend on inference success
    concordance_result = None
    twin_agreement = None
    selection_bias_result = None

    if inference_result is not None:
        # 14. Compute twin
        twin_result = compute_twin(
            censoring.measurable, strata, observed_counts,
            stratum_sizes_int, calibration, overlap,
        )

        # 15. Compare twin vs NUTS
        twin_agreement = compare_twin_nuts(inference_result, twin_result)

        # 16. Build synthetic vote data
        vote_rankings = _generate_synthetic_votes(
            censoring.measurable, gt_ranking, prng_seed,
        )
        vote_posterior = bootstrap_vote_ranks(
            vote_rankings, censoring.measurable, n_bootstrap=2000, seed=prng_seed,
        )

        # Tier boundaries from project.toml or derived from tier_size
        tier_size: int = proj.get("tier_size", 2)
        tier_boundaries_cfg = proj.get("tier_boundaries")
        if tier_boundaries_cfg is not None:
            tier_boundaries = tuple(int(b) for b in tier_boundaries_cfg)
        else:
            n_meas = len(censoring.measurable)
            tier_boundaries = tuple(
                i * (n_meas // tier_size)
                for i in range(1, tier_size)
            )

        total_entries = len(entries)

        # 17. Compute concordance
        concordance_result = compute_concordance(
            inference_result,
            vote_posterior,
            tier_boundaries,
            flag_threshold_tau,
            len(censoring.measurable),
            total_entries,
            meaningful_kappa_n,
            measurability_minimum,
        )

        # 18. Compute selection bias
        verdict_strings = {
            eid: v.value for eid, v in censoring.verdicts.items()
        }
        # Selection bias needs vote medians for ALL entries including unmeasurable;
        # for frame-blind entries not in the vote, assign worst rank.
        all_vote_medians: dict[str, float] = dict(vote_posterior.median_ranks)
        worst_rank = float(len(entries))
        for eid in entry_ids:
            if eid not in all_vote_medians:
                all_vote_medians[eid] = worst_rank
        selection_bias_result = compute_selection_bias(verdict_strings, all_vote_medians)

    # Fallback concordance/selection-bias for diagnostics-failure path
    if concordance_result is None:
        from engine.decide.concordance import STANDING_CAVEAT, ConcordanceResult

        concordance_result = ConcordanceResult(
            weighted_kappa_median=None,
            weighted_kappa_ci=None,
            measurable_count=len(censoring.measurable),
            total_count=len(entries),
            coverage_ratio=meas_map.coverage_ratio,
            below_prereg_minimum=meas_map.below_prereg_minimum,
            meaningful_kappa_n=meaningful_kappa_n,
            flags=(),
            standing_caveat=STANDING_CAVEAT,
        )
    if selection_bias_result is None:
        from engine.decide.selection_bias import SelectionBiasDisclosure

        selection_bias_result = SelectionBiasDisclosure(
            statistic_name="kruskal_wallis_h",
            statistic_value=float("nan"),
            p_value=float("nan"),
            n_entries_per_group={},
            severity="low",
        )

    # 19. Render report
    report_inputs = ReportInputs(
        cycle_id=cycle_id,
        engine_version=__version__,
        measurability_map=meas_map,
        concordance=concordance_result,
        selection_bias=selection_bias_result,
        robustness=None,
        twin_agreement=twin_agreement,
        non_publishable=True,
    )
    report_text = render_report(report_inputs)
    (results_dir / "report.md").write_text(report_text)

    # 20. Write results summary JSON
    summary: dict[str, object] = {
        "cycle_id": cycle_id,
        "engine_version": __version__,
        "corpus_mode": corpus_mode,
        "nuts_succeeded": not nuts_failed,
        "measurable_count": len(censoring.measurable),
        "frame_blind_count": len(censoring.frame_blind),
        "coverage_ratio": meas_map.coverage_ratio,
        "below_prereg_minimum": meas_map.below_prereg_minimum,
        "non_publishable": True,
    }
    if concordance_result.weighted_kappa_median is not None:
        summary["weighted_kappa_median"] = concordance_result.weighted_kappa_median
    if twin_agreement is not None:
        summary["twin_agreement_rate"] = twin_agreement.agreement_rate
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
