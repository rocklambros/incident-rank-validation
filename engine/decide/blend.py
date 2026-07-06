"""Probabilistic blend for the 2026 OWASP LLM Top-10.

Combines the recall-corrected incidence (lambda) posterior and the vote-rank posterior
into a distribution over each incumbent's final position (0.75 vote / 0.25 data,
sum-prevalence fold on the data axis, min-rank fold on the vote axis, frame-blind entries
placed by vote alone). Pure functions; RNG, N, and the manifest path are injected.
Paths in the manifest are repo-root-relative and resolved against the repo root found by
walking up from the manifest, so the module is CWD-independent.
See docs/superpowers/specs/2026-07-05-probabilistic-blend-relock-design.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from engine.snapshot.hashing import verify_snapshot_hash

INCUMBENTS: tuple[str, ...] = tuple(f"LLM{i:02d}" for i in range(1, 11))
W_VOTE: float = 0.75
SEED: int = 20260520
DEFAULT_N: int = 16000
EXPECTED_ROLLUP: dict[str, str] = {
    "ROLL-CMSB": "LLM01",
    "ROLL-LAPTF": "LLM03",
    "ROLL-CFAS": "LLM04",
    "ROLL-SICG": "LLM05",
}
EXPECTED_FRAME_BLIND: frozenset[str] = frozenset({"LLM04", "LLM08", "LLM10"})


@dataclass(frozen=True)
class BlendInputs:
    lambda_samples: npt.NDArray[np.float64]
    vote_rank_samples: npt.NDArray[np.float64]
    entry_ids: tuple[str, ...]
    rollup: dict[str, str]
    frame_blind: frozenset[str]


def _repo_root(start: Path) -> Path:
    p = start.resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found")


def load_inputs(manifest_path: Path) -> BlendInputs:
    """Load the blend inputs after verifying every input's sha256 and the label triangle."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    inputs = manifest["inputs"]
    try:
        root = _repo_root(manifest_path.parent)
    except FileNotFoundError:
        root = Path.cwd()

    def resolved(name: str) -> Path:
        p = Path(inputs[name]["path"])
        return p if p.is_absolute() else root / p

    for spec in inputs.values():
        p = Path(spec["path"])
        verify_snapshot_hash(p if p.is_absolute() else root / p, spec["sha256"])

    inf = json.loads(resolved("inference_summary").read_text())
    rb = json.loads(resolved("rankings_baselines").read_text())
    vote_ids = json.loads(resolved("vote_entry_ids").read_text())
    tax = json.loads(resolved("taxonomy").read_text())

    entry_ids = tuple(inf["entry_ids"])
    if tuple(rb["entry_ids"]) != entry_ids:
        raise ValueError("inference<->baselines entry_ids differ")
    if tuple(vote_ids) != entry_ids:
        raise ValueError("vote fixture entry_ids differ from inference entry_ids")

    lam = np.load(resolved("lambda_samples"))
    vote = np.load(resolved("vote_rank_samples"))
    if list(lam.shape) != inputs["lambda_samples"]["shape"]:
        raise ValueError(f"lambda_samples shape {lam.shape} != manifest")
    if list(vote.shape) != inputs["vote_rank_samples"]["shape"]:
        raise ValueError(f"vote_rank_samples shape {vote.shape} != manifest")

    rollup = {
        e["entry_id"]: e["rolled_into"]
        for e in tax["entries"]
        if e.get("is_rollup_candidate")
    }
    if rollup != EXPECTED_ROLLUP:
        raise ValueError(f"crosswalk drift: {rollup} != {EXPECTED_ROLLUP}")
    frame_blind = frozenset(rb["not_measurable"])
    if frame_blind != EXPECTED_FRAME_BLIND:
        raise ValueError(f"frame-blind drift: {frame_blind} != {EXPECTED_FRAME_BLIND}")

    return BlendInputs(lam, vote, entry_ids, rollup, frame_blind)
