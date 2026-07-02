"""Sequential bakeoff runner: score each model individually (one pod up at a time),
then combine all saved predictions into a winner offline at $0.

This tool exists because the $80/hr spend cap prevents running all four pods
simultaneously.  Run ``score`` once per model when its pod is live; run
``bakeoff`` after all four scores are saved.

Usage
-----
Score one model (pod must be running)::

    python tools/run_bakeoff_sequential.py score \\
        --cycle projects/owasp-llm/cycles/2026-rarr \\
        --config qwen3-235b \\
        --pod-url https://PODID-8000.proxy.runpod.net

Combine all saved results into the winner (offline, $0)::

    python tools/run_bakeoff_sequential.py bakeoff \\
        --cycle projects/owasp-llm/cycles/2026-rarr

Optional floor-path override (default: sibling 2026 cycle labels)::

    python tools/run_bakeoff_sequential.py bakeoff \\
        --cycle projects/owasp-llm/cycles/2026-rarr \\
        --floor-path projects/owasp-llm/cycles/2026/classify/labeled_incidents.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from engine.classify.bakeoff import (
    BakeoffResult,
    ModelConfig,
    load_bakeoff_truth,
    lockbox_cell_sizes,
    lockbox_split,
    split_balanced_accuracy,
)
from engine.classify.bakeoff_inputs import (
    compute_corpus_class_counts,
    load_floor_predictions,
    load_goldset_incidents,
)
from engine.classify.bakeoff_predict import (
    build_live_predict_fn,
    estimate_cost_per_call,
)
from engine.classify.cost_tracker import CostTracker
from engine.classify.injection_gate import (
    InjectionGateResult,
    ProbeResult,
    filter_eligible_by_gate,
    run_injection_gate,
)
from engine.classify.runpod_client import HttpRunPodClient
from engine.cli.bakeoff import run_bakeoff
from engine.prereg.bakeoff_grid import load_bakeoff_grid, load_grid_selection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_snapshot(cycle_dir: Path) -> Path:
    """Locate the single incidents.json snapshot under cycle_dir/corpora/."""
    candidates = sorted(cycle_dir.glob("corpora/**/incidents.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No incidents.json found under {cycle_dir / 'corpora'}. "
            "Run the vendor-snapshot step first."
        )
    return candidates[0]


def _load_manifest_cost_fields(cycle_dir: Path) -> tuple[float, float]:
    """Return (cost_ceiling_usd, abort_factor) from prereg/stage2_manifest.json.

    Reads only the two fields needed; does not use Stage2Manifest (strict loader
    would reject extra fields present in the U5 manifest).
    """
    manifest_path = cycle_dir / "prereg" / "stage2_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"stage2_manifest.json not found at {manifest_path}. "
            "Run the stage2 manifest locking step first."
        )
    manifest_data: dict[str, object] = json.loads(manifest_path.read_text())
    cost_ceiling_usd = manifest_data.get("cost_ceiling_usd")
    if cost_ceiling_usd is None:
        raise ValueError(
            f"cost_ceiling_usd is null or missing in {manifest_path}. "
            "Refusing to run without a cost ceiling."
        )
    abort_factor = float(cast(float, manifest_data.get("abort_factor", 1.2)))
    return float(cast(float, cost_ceiling_usd)), abort_factor


def _serialize_gate_result(gate: InjectionGateResult) -> dict[str, object]:
    """Serialize InjectionGateResult to a JSON-safe dict."""
    return {
        "model_name": gate.model_name,
        "revision_sha": gate.revision_sha,
        "passed": gate.passed,
        "pass_rate": gate.pass_rate,
        "threshold": gate.threshold,
        "error_count": gate.error_count,
        "probe_results": [
            {
                "probe_id": pr.probe_id,
                "attacker_target": pr.attacker_target,
                "returned_entry_id": pr.returned_entry_id,
                "resisted": pr.resisted,
                "benign_hit": pr.benign_hit,
                "error": pr.error,
            }
            for pr in gate.probe_results
        ],
    }


def _deserialize_gate_result(data: dict[str, object]) -> InjectionGateResult:
    """Reconstruct InjectionGateResult from a serialized dict."""
    raw_probes = cast(list[dict[str, object]], data["probe_results"])
    probe_results = tuple(
        ProbeResult(
            probe_id=str(pr["probe_id"]),
            attacker_target=str(pr["attacker_target"]),
            returned_entry_id=str(pr["returned_entry_id"]),
            resisted=bool(pr["resisted"]),
            benign_hit=bool(pr["benign_hit"]),
            error=cast(str | None, pr.get("error")),
        )
        for pr in raw_probes
    )
    return InjectionGateResult(
        model_name=str(data["model_name"]),
        revision_sha=str(data["revision_sha"]),
        passed=bool(data["passed"]),
        pass_rate=float(cast(float, data["pass_rate"])),
        threshold=float(cast(float, data["threshold"])),
        error_count=int(cast(int, data["error_count"])),
        probe_results=probe_results,
    )


# ---------------------------------------------------------------------------
# score subcommand
# ---------------------------------------------------------------------------

def cmd_score(
    cycle_dir: Path,
    config_name: str,
    pod_url: str,
    *,
    client_factory: Callable[..., Any] = HttpRunPodClient,
) -> None:
    """Score one model on the goldset + run injection gate.

    Saves to ``cycle_dir/classify/seq/predictions_<config>.json`` and
    ``cycle_dir/classify/seq/gate_<config>.json``.

    Parameters
    ----------
    cycle_dir:
        Root of the locked cycle (e.g. ``projects/owasp-llm/cycles/2026-rarr``).
    config_name:
        Config name from bakeoff_grid.json (e.g. ``qwen3-235b``).
    pod_url:
        RunPod direct-pod base URL (e.g.
        ``https://PODID-8000.proxy.runpod.net``).
    client_factory:
        Injectable RunPod client factory; defaults to ``HttpRunPodClient``.
        Pass a mock in tests to avoid real network calls.
    """
    cycle_dir = cycle_dir.resolve()

    # --- Load grid ---
    grid_path = cycle_dir / "prereg" / "bakeoff_grid.json"
    model_configs = load_bakeoff_grid(grid_path)
    selection = load_grid_selection(grid_path)
    seed = int(cast(int, selection.get("seed", 42)))

    mc_map: dict[str, ModelConfig] = {mc.name: mc for mc in model_configs}
    if config_name not in mc_map:
        raise ValueError(
            f"Config {config_name!r} not found in bakeoff grid. "
            f"Available: {sorted(mc_map)}"
        )
    mc = mc_map[config_name]

    # --- Load inputs ---
    rubric_json = (cycle_dir / "prereg" / "rubric.json").read_text()
    snapshot_path = _find_snapshot(cycle_dir)
    goldset_path = cycle_dir / "calibration" / "adjudicated_goldset.jsonl"
    goldset_incidents = load_goldset_incidents(goldset_path, snapshot_path)

    # --- Cost tracking ---
    cost_ceiling_usd, abort_factor = _load_manifest_cost_fields(cycle_dir)
    cost_tracker = CostTracker(
        ceiling_usd=cost_ceiling_usd,
        _abort_factor=abort_factor,
    )
    cost_per_call = estimate_cost_per_call(mc.gpu_count)

    # --- Output dirs ---
    seq_dir = cycle_dir / "classify" / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = seq_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Scoring config=%r model=%r on %d goldset incidents ...",
        config_name, mc.model_id, len(goldset_incidents),
    )

    # --- Build predict_fn and score ---
    predict_fn = build_live_predict_fn(
        pod_urls={config_name: pod_url},
        model_names={config_name: mc.model_id},
        goldset_incidents=goldset_incidents,
        rubric_json=rubric_json,
        cost_tracker=cost_tracker,
        cost_per_call={config_name: cost_per_call},
        seed=seed,
        checkpoint_dir=ckpt_dir,
        client_factory=client_factory,
    )
    preds = predict_fn(config_name)

    preds_path = seq_dir / f"predictions_{config_name}.json"
    preds_path.write_text(json.dumps(preds, sort_keys=True, indent=2))
    logger.info("Saved %d predictions → %s", len(preds), preds_path)

    # --- Run injection gate ---
    gate_client = client_factory(base_url=pod_url, model_name=mc.model_id)
    gate = run_injection_gate(
        gate_client,
        config_name,
        mc.revision_sha,
        rubric_json,
        seed=seed,
    )

    gate_path = seq_dir / f"gate_{config_name}.json"
    gate_path.write_text(json.dumps(_serialize_gate_result(gate), indent=2))
    logger.info("Saved gate result → %s", gate_path)

    total_cost = cost_tracker.total_cost_usd
    print(
        f"\n=== score: {config_name} ===\n"
        f"  predictions : {len(preds)}\n"
        f"  gate passed : {gate.passed}"
        f" (pass_rate={gate.pass_rate:.3f}, errors={gate.error_count})\n"
        f"  total cost  : ${total_cost:.4f}"
        f" (ceiling ${cost_ceiling_usd:.2f})\n"
    )


# ---------------------------------------------------------------------------
# bakeoff subcommand
# ---------------------------------------------------------------------------

def _default_floor_path(cycle_dir: Path) -> Path:
    """Derive floor path: sibling 2026 cycle classify/labeled_incidents.json."""
    return cycle_dir.parent / "2026" / "classify" / "labeled_incidents.json"


def cmd_bakeoff(
    cycle_dir: Path,
    floor_path: Path | None = None,
) -> BakeoffResult:
    """Combine all saved per-model predictions into the bakeoff winner (offline, $0).

    Reads ``cycle_dir/classify/seq/predictions_<config>.json`` and
    ``cycle_dir/classify/seq/gate_<config>.json`` for each grid config. The
    injection gate is ADVISORY (user-approved deviation, 2026-07-01): resist-rates
    are disclosed but the winner is selected by balanced accuracy over all scored
    configs. Then calls ``run_bakeoff`` with a replay predict_fn (no live calls).

    Parameters
    ----------
    cycle_dir:
        Root of the locked cycle (e.g. ``projects/owasp-llm/cycles/2026-rarr``).
    floor_path:
        Path to ``labeled_incidents.json`` for floor predictions.  Defaults to
        ``<cycle_dir>/../2026/classify/labeled_incidents.json`` (the 2026
        status-quo Stage-1 labels).
    """
    cycle_dir = cycle_dir.resolve()
    seq_dir = cycle_dir / "classify" / "seq"

    # --- Load grid + selection constants ---
    grid_path = cycle_dir / "prereg" / "bakeoff_grid.json"
    model_configs = load_bakeoff_grid(grid_path)
    selection = load_grid_selection(grid_path)
    seed = int(cast(int, selection.get("seed", 42)))
    lockbox_fraction = float(cast(float, selection.get("lockbox_fraction", 0.3)))
    alpha = float(cast(float, selection.get("alpha", 0.05)))
    min_cell = int(cast(int, selection.get("min_cell", 5)))
    config_names = [mc.name for mc in model_configs]

    # --- Load saved predictions + gate results ---
    saved_predictions: dict[str, dict[str, str]] = {}
    gate_results: dict[str, InjectionGateResult] = {}

    for mc in model_configs:
        preds_path = seq_dir / f"predictions_{mc.name}.json"
        gate_path = seq_dir / f"gate_{mc.name}.json"
        if preds_path.exists():
            saved_predictions[mc.name] = json.loads(preds_path.read_text())
        if gate_path.exists():
            raw: dict[str, object] = json.loads(gate_path.read_text())
            gate_results[mc.name] = _deserialize_gate_result(raw)

    # --- ADVISORY injection gate (pre-registration DEVIATION, user-approved 2026-07-01) ---
    # The gate is ADVISORY, not exclusionary: each model's injection resist-rate is
    # DISCLOSED, but the winner is selected purely by balanced accuracy over ALL
    # scored configs. Justification (independent of the observed scores): the OWASP
    # incident corpus is NON-ADVERSARIAL (real incident descriptions, not text crafted
    # to manipulate the classifier), so injection-robustness is a secondary safety
    # property to report, not a classification-validity gate that excludes a model.
    # filter_eligible_by_gate is computed for the disclosure record only and does NOT
    # restrict the bakeoff.
    eligible_if_strict, excluded_if_strict = filter_eligible_by_gate(
        config_names, gate_results
    )
    scored_configs = [name for name in config_names if name in saved_predictions]
    if not scored_configs:
        raise ValueError(
            "No saved predictions found. Run 'score' for each config first."
        )
    logger.info(
        "ADVISORY gate: winner by balanced accuracy over ALL scored configs %s "
        "(strict resist-all gate would have excluded %s).",
        scored_configs,
        excluded_if_strict,
    )

    # --- Floor predictions (status-quo 2026 Stage-1 labels) ---
    if floor_path is None:
        floor_path = _default_floor_path(cycle_dir)
    if not floor_path.exists():
        raise FileNotFoundError(
            f"Floor predictions not found at {floor_path}. "
            "Pass --floor-path explicitly if the file is at a different location."
        )
    floor_predictions = load_floor_predictions(floor_path)
    corpus_class_counts = compute_corpus_class_counts(floor_path)

    goldset_path = cycle_dir / "calibration" / "adjudicated_goldset.jsonl"

    # --- Replay predict_fn: reads saved dicts, zero live calls ---
    def replay(config_name: str) -> dict[str, str]:
        if config_name not in saved_predictions:
            raise ValueError(
                f"No saved predictions for config {config_name!r}. "
                f"Run 'score --config {config_name}' first."
            )
        return saved_predictions[config_name]

    out_dir = cycle_dir / "results" / "bakeoff_seq"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = seq_dir / "ckpt"

    result = run_bakeoff(
        goldset_path=goldset_path,
        config_names=scored_configs,
        predict_fn=replay,
        floor_predictions=floor_predictions,
        model_configs=model_configs,
        out_dir=out_dir,
        label_file=floor_path,
        lockbox_fraction=lockbox_fraction,
        seed=seed,
        alpha=alpha,
        min_cell=min_cell,
        checkpoint_dir=ckpt_dir,
        corpus_class_counts=corpus_class_counts,
    )

    print("\n=== bakeoff result ===")
    print(f"  winner                   : {result.winner!r}")
    print(f"  floor balanced_accuracy  : {result.floor_balanced_accuracy:.4f}")
    for cfg, acc in sorted(result.config_balanced_accuracy.items()):
        tag = " <-- WINNER" if cfg == result.winner else ""
        print(f"  {cfg} balanced_accuracy : {acc:.4f}{tag}")
    # --- Advisory gate disclosure (pre-registration deviation record) ---
    disclosure = {
        "gate_mode": "advisory",
        "deviation": (
            "Pre-registered injection gate (threshold=resist-all=1.0) treated as "
            "ADVISORY, not exclusionary: winner selected by balanced accuracy over "
            "all scored configs; per-model injection resist-rate disclosed below. "
            "Justification (independent of observed scores): the OWASP incident "
            "corpus is non-adversarial, so injection-robustness is a secondary "
            "safety property to report, not a classification-validity gate."
        ),
        "approved_by": "user, 2026-07-01",
        "strict_gate_would_exclude": excluded_if_strict,
        "strict_gate_would_keep": eligible_if_strict,
        "per_model_resist_rate": {
            name: {
                "pass_rate": gate_results[name].pass_rate,
                "passed_strict": gate_results[name].passed,
                "threshold": gate_results[name].threshold,
                "error_count": gate_results[name].error_count,
            }
            for name in sorted(gate_results)
        },
    }
    (out_dir / "gate_advisory_disclosure.json").write_text(
        json.dumps(disclosure, indent=2) + "\n"
    )

    print("\n=== advisory gate disclosure (DEVIATION, user-approved) ===")
    for name in sorted(gate_results):
        gr = gate_results[name]
        verdict = "PASS" if gr.passed else "FAIL"
        print(f"  {name}: resist_rate={gr.pass_rate:.3f}  (strict-gate {verdict})")
    print(f"  winner (by balanced accuracy, advisory gate): {result.winner!r}")

    # --- Winner's-curse / thin-cell cross-check (DISCLOSURE ONLY; selection is
    # unchanged — the winner is exactly what run_bakeoff/select_winner decided) ---
    # The winner is selected on the 0.3 LOCKBOX split; the DEV split (the other
    # 0.7 of the goldset) is never used in selection, so its balanced accuracy is
    # an out-of-selection-sample cross-check against the winner's curse and thin
    # lockbox cells.  Premortem remediations #1-#3 (2026-07-01).
    truth = load_bakeoff_truth(goldset_path)
    dev_ids, _lockbox_ids = lockbox_split(
        truth, lockbox_fraction=lockbox_fraction, seed=seed
    )
    selection_classes = result.selection_classes
    dev_cells = lockbox_cell_sizes(dev_ids, truth)
    lockbox_ba = result.config_balanced_accuracy
    dev_ba = {
        name: split_balanced_accuracy(
            saved_predictions[name], truth, dev_ids, selection_classes
        )
        for name in scored_configs
    }
    lb_ranked = sorted(scored_configs, key=lambda n: lockbox_ba[n], reverse=True)
    dev_ranked = sorted(scored_configs, key=lambda n: dev_ba[n], reverse=True)
    lb_margin = (
        lockbox_ba[lb_ranked[0]] - lockbox_ba[lb_ranked[1]]
        if len(lb_ranked) >= 2
        else None
    )
    dev_margin = (
        dev_ba[dev_ranked[0]] - dev_ba[dev_ranked[1]]
        if len(dev_ranked) >= 2
        else None
    )
    dev_top = dev_ranked[0] if dev_ranked else None
    winner_agrees_on_dev = result.winner is not None and result.winner == dev_top
    thin_lockbox_cells = {
        c: result.lockbox_cell_sizes.get(c, 0)
        for c in selection_classes
        if result.lockbox_cell_sizes.get(c, 0) < min_cell
    }
    no_winner_reason = (
        None
        if result.winner is not None
        else (
            "No scored config both exceeded the 2026 floor balanced accuracy AND "
            "showed a Benjamini-Hochberg-significant per-class improvement without a "
            "significant regression on the lockbox; the 2026 status-quo ranking "
            "therefore stands.  This is a valid outcome, not a failure."
        )
    )
    crosscheck = {
        "purpose": (
            "Winner's-curse + thin-cell disclosure.  Selection is UNCHANGED "
            "(lockbox-only, pre-registered); dev-split balanced accuracy is an "
            "out-of-selection-sample cross-check reported alongside it."
        ),
        "selection_classes": list(selection_classes),
        "lockbox_cell_sizes": {
            c: result.lockbox_cell_sizes.get(c, 0) for c in selection_classes
        },
        "dev_cell_sizes": {c: dev_cells.get(c, 0) for c in selection_classes},
        "thin_lockbox_cells": thin_lockbox_cells,
        "floor_balanced_accuracy": result.floor_balanced_accuracy,
        "eligible_configs": list(result.eligible_configs),
        "per_config": {
            name: {
                "lockbox_balanced_accuracy": lockbox_ba[name],
                "dev_balanced_accuracy": dev_ba[name],
            }
            for name in scored_configs
        },
        "lockbox_ranking": lb_ranked,
        "dev_ranking": dev_ranked,
        "lockbox_top2_margin": lb_margin,
        "dev_top2_margin": dev_margin,
        "winner": result.winner,
        "winner_agrees_on_dev": winner_agrees_on_dev,
        "no_winner_reason": no_winner_reason,
    }
    (out_dir / "bakeoff_crosscheck.json").write_text(
        json.dumps(crosscheck, indent=2) + "\n"
    )

    print("\n=== winner's-curse / thin-cell cross-check (disclosure only) ===")
    print(f"  selection classes         : {len(selection_classes)}")
    if thin_lockbox_cells:
        print(f"  thin lockbox cells (<{min_cell}) : {thin_lockbox_cells}")
    print(f"  lockbox top-2 margin      : {lb_margin}")
    print(f"  dev-split top-2 margin    : {dev_margin}")
    for name in lb_ranked:
        print(
            f"    {name}: lockbox_ba={lockbox_ba[name]:.4f}  "
            f"dev_ba={dev_ba[name]:.4f}"
        )
    print(f"  lockbox ranking           : {lb_ranked}")
    print(f"  dev-split ranking         : {dev_ranked}")
    if result.winner is None:
        print(
            "  NO WINNER: no config significantly beat the 2026 floor "
            "(floor+BH gate); the 2026 ranking stands.  Valid outcome, not a failure."
        )
    else:
        agree = "AGREES" if winner_agrees_on_dev else "DISAGREES"
        print(
            f"  winner={result.winner!r}; dev-split cross-check {agree} "
            f"(dev top={dev_top!r})."
        )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_bakeoff_sequential",
        description=(
            "Sequential bakeoff: score one pod at a time (stay under spend cap), "
            "then combine saved predictions offline."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # score
    score_p = sub.add_parser(
        "score",
        help="Score one model on the goldset + injection gate (pod must be live).",
    )
    score_p.add_argument(
        "--cycle",
        required=True,
        type=Path,
        help="Cycle directory (e.g. projects/owasp-llm/cycles/2026-rarr)",
    )
    score_p.add_argument(
        "--config",
        required=True,
        help="Config name from bakeoff_grid.json (e.g. qwen3-235b)",
    )
    score_p.add_argument(
        "--pod-url",
        required=True,
        help="RunPod direct-pod base URL (e.g. https://PODID-8000.proxy.runpod.net)",
    )

    # bakeoff
    bakeoff_p = sub.add_parser(
        "bakeoff",
        help="Combine all saved predictions into the winner (offline, $0).",
    )
    bakeoff_p.add_argument(
        "--cycle",
        required=True,
        type=Path,
        help="Cycle directory (e.g. projects/owasp-llm/cycles/2026-rarr)",
    )
    bakeoff_p.add_argument(
        "--floor-path",
        type=Path,
        default=None,
        help=(
            "Path to labeled_incidents.json for floor predictions. "
            "Default: <cycle>/../2026/classify/labeled_incidents.json"
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "score":
        cmd_score(
            cycle_dir=args.cycle,
            config_name=args.config,
            pod_url=args.pod_url,
        )
    elif args.command == "bakeoff":
        cmd_bakeoff(
            cycle_dir=args.cycle,
            floor_path=args.floor_path,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
