"""Bake-off orchestration (Plan 8e Phase 1: harness; U7: live predict_fn wiring).

``run_bakeoff()`` is fully testable with an injected predict_fn.
``bakeoff_cmd()`` is the plain-Python entry point with injectable
``predict_fn`` and ``client_factory`` seams (R6): pass a mock
``client_factory`` and ``execute=True`` to exercise the full
``env-pod-URLs → build_live_predict_fn → classify_one`` glue offline/$0.
The live production run is U8.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import click

from engine.classify.bakeoff import (
    BAKEOFF_ALPHA,
    LOCKBOX_FRACTION,
    MIN_CELL,
    BakeoffResult,
    ModelConfig,
    goldset_corpus_divergence,
    goldset_provenance,
    load_bakeoff_truth,
    lockbox_split,
    select_winner,
    write_bakeoff_provenance,
)
from engine.classify.bakeoff_inputs import (
    compute_corpus_class_counts,
    load_floor_predictions,
    load_goldset_incidents,
)
from engine.classify.bakeoff_predict import (
    PredictFn,
    build_live_predict_fn,
    estimate_cost_per_call,
)
from engine.classify.cost_tracker import CostTracker
from engine.classify.runpod_client import HttpRunPodClient
from engine.prereg.bakeoff_grid import load_bakeoff_grid, load_grid_selection


def run_bakeoff(
    goldset_path: Path,
    config_names: list[str],
    predict_fn: PredictFn,
    floor_predictions: Mapping[str, str],
    model_configs: list[ModelConfig],
    out_dir: Path,
    label_file: Path,
    lockbox_fraction: float = LOCKBOX_FRACTION,
    seed: int = 42,
    alpha: float = BAKEOFF_ALPHA,
    min_cell: int = MIN_CELL,
    checkpoint_dir: Path | None = None,
    corpus_class_counts: Mapping[str, int] | None = None,
) -> BakeoffResult:
    """Score every config against the goldset lockbox and select the winner."""
    truth = load_bakeoff_truth(goldset_path)
    _dev, lockbox = lockbox_split(truth, lockbox_fraction=lockbox_fraction, seed=seed)

    # F7: per-config checkpoint cache so a mid-sweep failure resumes instead of
    # discarding a multi-hour grid run.
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_predictions: dict[str, dict[str, str]] = {}
    for name in config_names:
        ckpt = checkpoint_dir / f"{name}.json" if checkpoint_dir is not None else None
        if ckpt is not None and ckpt.exists():
            config_predictions[name] = json.loads(ckpt.read_text())
            continue
        preds = dict(predict_fn(name))
        config_predictions[name] = preds
        if ckpt is not None:
            ckpt.write_text(json.dumps(preds, sort_keys=True))

    # Coverage guard: every lockbox incident must have a prediction, else the
    # metric denominator silently shrinks (a Phase-3 footgun).
    missing_floor = lockbox - set(floor_predictions)
    if missing_floor:
        raise ValueError(
            f"floor_predictions missing {len(missing_floor)} lockbox incidents"
        )
    for name, preds in config_predictions.items():
        missing = lockbox - set(preds)
        if missing:
            raise ValueError(
                f"config {name!r} missing {len(missing)} lockbox incidents"
            )

    result = select_winner(
        config_predictions, floor_predictions, truth, lockbox,
        alpha=alpha, min_cell=min_cell,
    )

    goldset_meta = goldset_provenance(goldset_path)
    if corpus_class_counts is not None:
        goldset_meta["corpus_tv_divergence"] = goldset_corpus_divergence(
            truth, corpus_class_counts
        )
    write_bakeoff_provenance(
        out_dir,
        result,
        model_configs,
        label_file,
        seed=seed,
        lockbox_fraction=lockbox_fraction,
        min_cell=min_cell,
        goldset_meta=goldset_meta,
    )
    return result


def bakeoff_cmd(
    cycle_dir: Path,
    *,
    execute: bool,
    predict_fn: PredictFn | None = None,
    client_factory: object = HttpRunPodClient,
) -> BakeoffResult:
    """Wire the bake-off: load locked grid + manifest ceiling, inputs, and run.

    The injectable ``predict_fn`` and ``client_factory`` parameters are
    offline test seams (R6).  Passing ``execute=True`` with ``predict_fn=None``
    and env pod URLs builds the live ``predict_fn`` via
    ``build_live_predict_fn`` — the client_factory is forwarded so tests can
    inject a mock without touching the network.

    Parameters
    ----------
    cycle_dir:
        Root of the locked cycle (e.g. ``cycles/2026-rarr``).  Must contain
        ``prereg/bakeoff_grid.json``, ``prereg/stage2_manifest.json``,
        ``calibration/adjudicated_goldset.jsonl``,
        ``classify/labeled_incidents.json``, and a snapshot under
        ``corpora/**/incidents.json``.
    execute:
        ``True`` → build the live predict_fn from env pod URLs (U8 run).
        ``False`` with ``predict_fn=None`` → raises immediately.
    predict_fn:
        Injected predict_fn (tests / dry-runs).  When not None, ``execute``
        and ``client_factory`` are ignored for prediction; cost_tracker is
        still constructed (manifest ceiling is always validated).
    client_factory:
        Injectable RunPod client factory forwarded to
        ``build_live_predict_fn``; defaults to ``HttpRunPodClient``.

    Returns
    -------
    BakeoffResult
        Selection result written to ``cycle_dir/classify/``.

    Raises
    ------
    FileNotFoundError
        If ``stage2_manifest.json`` is absent.
    ValueError
        If ``cost_ceiling_usd`` is null, or ``predict_fn`` is None and
        ``execute=False`` (or execute=True with no env pod URLs).
    """
    # --- Load locked grid ---
    grid_path = cycle_dir / "prereg" / "bakeoff_grid.json"
    model_configs = load_bakeoff_grid(grid_path)
    selection = load_grid_selection(grid_path)
    lockbox_fraction = float(cast(float, selection.get("lockbox_fraction", LOCKBOX_FRACTION)))
    seed = int(cast(int, selection.get("seed", 42)))
    alpha = float(cast(float, selection.get("alpha", BAKEOFF_ALPHA)))
    min_cell = int(cast(int, selection.get("min_cell", MIN_CELL)))

    # --- R3: read cost_ceiling_usd and abort_factor directly from JSON ---
    # Stage2Manifest.read() is STRICT (cls(**d)); the U5 manifest carries extra
    # fields (abort_factor, selected_from, injection_gate_*) that would raise
    # TypeError.  Read only what bakeoff_cmd needs; do NOT loosen the shared
    # strict loader.
    manifest_path = cycle_dir / "prereg" / "stage2_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"stage2_manifest.json not found at {manifest_path}. "
            "Run the stage2 manifest locking step before the bake-off."
        )
    manifest_data: dict[str, object] = json.loads(manifest_path.read_text())
    cost_ceiling_usd = manifest_data.get("cost_ceiling_usd")
    if cost_ceiling_usd is None:
        raise ValueError(
            f"cost_ceiling_usd is null or missing in {manifest_path}. "
            "Refusing to run the bake-off without a cost ceiling."
        )
    abort_factor = float(cast(float, manifest_data.get("abort_factor", 1.2)))
    cost_tracker = CostTracker(
        ceiling_usd=float(cast(float, cost_ceiling_usd)),
        _abort_factor=abort_factor,
    )

    # --- Derive standard paths within cycle_dir ---
    goldset_path = cycle_dir / "calibration" / "adjudicated_goldset.jsonl"
    labeled_incidents_path = cycle_dir / "classify" / "labeled_incidents.json"
    rubric_path = cycle_dir / "prereg" / "rubric.json"
    out_dir = cycle_dir / "classify"
    checkpoint_dir = cycle_dir / "classify" / "bakeoff_checkpoint"

    # Locate snapshot: corpora/**/incidents.json (exactly one expected)
    snapshot_candidates = sorted(cycle_dir.glob("corpora/**/incidents.json"))
    if not snapshot_candidates:
        raise FileNotFoundError(
            f"No incidents.json found under {cycle_dir / 'corpora'}. "
            "Run the vendor-snapshot step first."
        )
    snapshot_path = snapshot_candidates[0]

    # --- Load bake-off inputs ---
    goldset_incidents = load_goldset_incidents(goldset_path, snapshot_path)
    floor_predictions = load_floor_predictions(labeled_incidents_path)
    corpus_class_counts = compute_corpus_class_counts(labeled_incidents_path)
    rubric_json = rubric_path.read_text()

    config_names = [mc.name for mc in model_configs]

    # --- Build effective predict_fn ---
    if predict_fn is not None:
        # Injected seam: use as-is (offline tests / dry-runs)
        effective_predict_fn: PredictFn = predict_fn
    elif execute:
        # Live path: read pod URLs from env, build live predict_fn (U8)
        pod_urls: dict[str, str] = {}
        model_names_env: dict[str, str] = {}
        for i in range(1, 10):
            name = os.environ.get(f"RUNPOD_MODEL_{i}_NAME", "")
            url = os.environ.get(f"RUNPOD_MODEL_{i}_URL", "")
            if name and url:
                pod_urls[name] = url
                model_names_env[name] = name
        if not pod_urls:
            raise ValueError(
                "execute=True but no RUNPOD_MODEL_N_URL / RUNPOD_MODEL_N_NAME "
                "env vars found.  Set RUNPOD_MODEL_1_URL=..., "
                "RUNPOD_MODEL_1_NAME=..., etc. or inject a predict_fn."
            )
        cost_per_call = {
            mc.name: estimate_cost_per_call(mc.gpu_count)
            for mc in model_configs
            if mc.gpu_count is not None
        }
        effective_predict_fn = build_live_predict_fn(
            pod_urls=pod_urls,
            model_names=model_names_env,
            goldset_incidents=goldset_incidents,
            rubric_json=rubric_json,
            cost_tracker=cost_tracker,
            cost_per_call=cost_per_call,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            client_factory=client_factory,
        )
    else:
        raise ValueError(
            "predict_fn is None and execute=False.  "
            "Either inject a predict_fn via the predict_fn parameter "
            "(for offline tests / dry-runs), or set execute=True with "
            "RUNPOD_MODEL_N_URL / RUNPOD_MODEL_N_NAME env vars for the "
            "live U8 run."
        )

    # --- Run bake-off ---
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_bakeoff(
        goldset_path=goldset_path,
        config_names=config_names,
        predict_fn=effective_predict_fn,
        floor_predictions=floor_predictions,
        model_configs=model_configs,
        out_dir=out_dir,
        label_file=labeled_incidents_path,
        lockbox_fraction=lockbox_fraction,
        seed=seed,
        alpha=alpha,
        min_cell=min_cell,
        checkpoint_dir=checkpoint_dir,
        corpus_class_counts=corpus_class_counts,
    )


@click.command("bakeoff")
@click.argument(
    "cycle_dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
)
@click.option(
    "--execute/--no-execute",
    default=False,
    help=(
        "Build the live RunPod predict_fn and run the bake-off (U8 production "
        "run).  Requires RUNPOD_MODEL_N_URL / RUNPOD_MODEL_N_NAME env vars.  "
        "Without --execute the command raises immediately."
    ),
)
def bakeoff_cli_cmd(cycle_dir: Path, execute: bool) -> None:
    """Run the classifier bake-off against a locked cycle directory."""
    bakeoff_cmd(cycle_dir, execute=execute)
