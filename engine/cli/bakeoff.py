"""Bake-off orchestration (Plan 8e Phase 1: harness, no live GPU).

run_bakeoff() is fully testable with an injected predict_fn.  The click command
defers live RunPod wiring to Phase 3 (the deliberate, cost-bearing run step).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import click

from engine.classify.bakeoff import (
    BAKEOFF_ALPHA,
    LOCKBOX_FRACTION,
    BakeoffResult,
    ModelConfig,
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
) -> BakeoffResult:
    """Score every config against the goldset lockbox and select the winner."""
    truth = load_bakeoff_truth(goldset_path)
    _dev, lockbox = lockbox_split(truth, lockbox_fraction=lockbox_fraction, seed=seed)
    config_predictions = {name: predict_fn(name) for name in config_names}
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
        config_predictions, floor_predictions, truth, lockbox, alpha=alpha
    )
    write_bakeoff_provenance(
        out_dir,
        result,
        model_configs,
        label_file,
        seed=seed,
        lockbox_fraction=lockbox_fraction,
    )
    return result


@click.command("bakeoff")
def bakeoff_cmd() -> None:
    """Run the classifier bake-off (live RunPod wiring lands in Phase 3)."""
    raise NotImplementedError(
        "live RunPod predict_fn is wired in Phase 3 (the deliberate GPU run "
        "step); the bake-off scoring/selection harness is run_bakeoff()."
    )
