"""Bake-off orchestration (Plan 8e Phase 1: harness, no live GPU).

run_bakeoff() is fully testable with an injected predict_fn.  The click command
defers live RunPod wiring to Phase 3 (the deliberate, cost-bearing run step).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

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

PredictFn = Callable[[str], dict[str, str]]


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


@click.command("bakeoff")
def bakeoff_cmd() -> None:
    """Run the classifier bake-off (live RunPod wiring lands in Phase 3)."""
    raise NotImplementedError(
        "live RunPod predict_fn is wired in Phase 3 (the deliberate GPU run "
        "step); the bake-off scoring/selection harness is run_bakeoff()."
    )
