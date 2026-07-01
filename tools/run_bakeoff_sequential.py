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

from engine.classify.bakeoff import BakeoffResult, ModelConfig
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
    ``cycle_dir/classify/seq/gate_<config>.json`` for each grid config, filters
    by injection gate, then calls ``run_bakeoff`` with a replay predict_fn that
    makes no live calls.

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

    # --- Gate filter (fail-closed) ---
    eligible, excluded = filter_eligible_by_gate(config_names, gate_results)
    if excluded:
        logger.warning("Excluded by injection gate: %s", excluded)
    if not eligible:
        raise ValueError(
            "No eligible configs after gate filtering. "
            "Run 'score' for each config before running 'bakeoff', "
            "or check gate_<config>.json files."
        )
    logger.info("Eligible configs: %s", eligible)

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
        config_names=eligible,
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
    if excluded:
        print(f"  excluded (gate failed)   : {excluded}")

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
