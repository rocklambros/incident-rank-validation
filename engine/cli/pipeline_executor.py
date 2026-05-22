"""Real-data pipeline orchestrator (R1).

Analogous to execute_synthetic_pipeline() but for real corpus data.
Wires Stage-1 classification, Stage-1→Stage-2 routing, calibration
loading, NUTS inference, and decision-layer assembly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

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


def _load_calibration(cal_path: Path) -> "Calibration":
    """Deserialize calibration posteriors from JSON."""
    from engine.calibrate.beta import BetaPosterior, Calibration

    data = json.loads(cal_path.read_text())

    def _parse_posteriors(d: dict) -> dict[tuple[str, str], BetaPosterior]:
        result: dict[tuple[str, str], BetaPosterior] = {}
        for key_str, params in d.items():
            parts = key_str.split("::")
            if len(parts) == 2:
                result[(parts[0], parts[1])] = BetaPosterior(
                    alpha=float(params["alpha"]),
                    beta=float(params["beta"]),
                )
        return result

    recall = _parse_posteriors(data.get("recall", {}))
    precision = _parse_posteriors(data.get("precision", {}))
    return Calibration(recall=recall, precision=precision)


def _load_manifest(manifest_path: Path) -> "PreregManifest":
    """Load PreregManifest from JSON file."""
    import dataclasses

    from engine.prereg.manifest import PreregManifest

    data = json.loads(manifest_path.read_text())

    field_names = {f.name for f in dataclasses.fields(PreregManifest)}
    filtered = {k: v for k, v in data.items() if k in field_names}

    if "robustness_specs" in filtered and isinstance(filtered["robustness_specs"], list):
        filtered["robustness_specs"] = tuple(filtered["robustness_specs"])

    return PreregManifest(**filtered)


def _build_counts_from_labeled(
    labeled: list[dict[str, object]],
) -> tuple[dict[tuple[str, str], int], dict[str, int], tuple[str, ...], tuple[str, ...]]:
    """Build observation counts from labeled_incidents.json.

    Returns (observed_counts, stratum_sizes, measurable_entries, strata).
    """
    from collections import Counter

    entry_stratum_counts: Counter[tuple[str, str]] = Counter()
    stratum_doc_counts: Counter[str] = Counter()
    entry_set: set[str] = set()
    stratum_set: set[str] = set()

    for item in labeled:
        eid = str(item.get("entry_id", ""))
        stratum = str(item.get("stratum", "default"))
        entry_stratum_counts[(eid, stratum)] += 1
        stratum_doc_counts[stratum] += 1
        entry_set.add(eid)
        stratum_set.add(stratum)

    measurable_entries = tuple(sorted(entry_set))
    strata = tuple(sorted(stratum_set))

    observed_counts = dict(entry_stratum_counts)
    stratum_sizes = {s: max(stratum_doc_counts[s], 1) for s in strata}

    return observed_counts, stratum_sizes, measurable_entries, strata


def execute_infer_phase(
    cycle: Path,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
) -> None:
    import os
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "true")

    cal_path = cycle / "calibrate" / "posteriors.json"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"Calibration posteriors not found: {cal_path}. "
            "Run the gold-set calibration pipeline (Plan 4) first. "
            "Real inference MUST NOT use uniform Beta(1,1) priors."
        )

    classify_dir = cycle / "classify"
    labeled_path = classify_dir / "labeled_incidents.json"
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Labeled incidents not found: {labeled_path}. Run classify first."
        )

    vote_dir = cycle / "vote"
    if vote_dir.exists() and any(vote_dir.iterdir()):
        raise RuntimeError(
            "Vote data found during infer phase. "
            "Vote enters only at decide (HANDOFF §6 control 2)."
        )

    # Load manifest
    prereg = cycle / "prereg"
    manifest = _load_manifest(prereg / "manifest.json")

    # Load calibration posteriors
    calibration = _load_calibration(cal_path)

    # Load labeled incidents
    labeled = json.loads(labeled_path.read_text())

    # Build observation arrays
    from engine.model.overlap import OverlapWeights

    observed_counts, stratum_sizes, measurable_entries, strata = _build_counts_from_labeled(
        labeled
    )

    overlap = OverlapWeights(weights={})

    # Run NUTS inference
    from engine.model.inference import DiagnosticsFailure, run_inference

    out_dir = cycle / "infer"

    try:
        result = run_inference(
            manifest=manifest,
            measurable_entries=measurable_entries,
            strata=strata,
            observed_counts=observed_counts,
            stratum_sizes=stratum_sizes,
            calibration=calibration,
            overlap=overlap,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        write_infer_artifacts(result, out_dir)
    except DiagnosticsFailure as e:
        write_nuts_failure(out_dir, str(e), None)
        raise


def write_infer_artifacts(
    result: "InferenceResult",
    out_dir: Path,
) -> None:
    from engine.model.inference import InferenceResult  # noqa: F401

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lambda_samples.npy", result.lambda_samples)
    summary = {
        "entry_ids": list(result.entry_ids),
        "r_hat": result.r_hat,
        "ess": result.ess,
        "divergences": result.divergences,
        "num_warmup": result.num_warmup,
        "num_samples": result.num_samples,
        "num_chains": 4,
    }
    (out_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def write_nuts_failure(
    out_dir: Path,
    error_message: str,
    partial_samples: npt.NDArray[np.float64] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics_failure.txt").write_text(
        f"NUTS diagnostics failed.\n\n{error_message}\n"
    )
    if partial_samples is not None:
        np.save(out_dir / "partial_samples.npy", partial_samples)
        (out_dir / "partial_results.json").write_text(
            json.dumps({"status": "partial", "shape": list(partial_samples.shape)}, indent=2) + "\n"
        )


def write_decide_artifacts(
    concordance: "ConcordanceResult",
    out_dir: Path,
    rollup_results: tuple = (),
    selection_bias: object | None = None,
    twin_agreement: object | None = None,
    robustness: object | None = None,
) -> None:
    from engine.decide.concordance import ConcordanceResult  # noqa: F401

    out_dir.mkdir(parents=True, exist_ok=True)

    conc_dict = {
        "weighted_kappa_median": concordance.weighted_kappa_median,
        "weighted_kappa_ci": list(concordance.weighted_kappa_ci) if concordance.weighted_kappa_ci else None,
        "measurable_count": concordance.measurable_count,
        "total_count": concordance.total_count,
        "coverage_ratio": concordance.coverage_ratio,
        "below_prereg_minimum": concordance.below_prereg_minimum,
        "flags": [
            {"entry_id": f.entry_id, "probability": f.probability, "direction": f.direction.value}
            for f in concordance.flags
        ],
    }
    (out_dir / "concordance.json").write_text(
        json.dumps(conc_dict, indent=2) + "\n"
    )

    if rollup_results:
        rollup_data = [
            {
                "parent_entry_id": r.parent_entry_id,
                "child_entry_id": r.child_entry_id,
                "verdict": r.verdict.value,
                "p_distinct_cluster": r.p_distinct_cluster,
            }
            for r in rollup_results
        ]
        (out_dir / "rollup.json").write_text(
            json.dumps(rollup_data, indent=2) + "\n"
        )

    if selection_bias is not None:
        (out_dir / "selection_bias.json").write_text(
            json.dumps({
                "statistic_name": selection_bias.statistic_name,
                "statistic_value": selection_bias.statistic_value,
                "p_value": selection_bias.p_value,
                "severity": selection_bias.severity,
            }, indent=2) + "\n"
        )


def write_reproduction_bundle(
    out_dir: Path,
    cycle_id: str,
    engine_version: str,
    snapshot_hash: str,
    manifest_hash: str,
    lockfile_hash: str,
    stage2_manifest_hash: str = "",
    calibration_hash: str = "",
    vote_data_hash: str = "",
) -> None:
    from engine.repro.bundle import ReproductionBundle

    bundle = ReproductionBundle(
        cycle_id=cycle_id,
        engine_version=engine_version,
        snapshot_hash=snapshot_hash,
        manifest_hash=manifest_hash,
        lockfile_hash=lockfile_hash,
        provenance={
            "stage2_manifest_hash": stage2_manifest_hash,
            "calibration_hash": calibration_hash,
            "vote_data_hash": vote_data_hash,
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle.write(out_dir / "repro_bundle.json")
