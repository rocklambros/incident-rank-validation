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


@dataclass(frozen=True)
class BlendResult:
    order: tuple[str, ...]
    mean_position: dict[str, float]
    p_top3: dict[str, float]
    p_top5: dict[str, float]
    interval: dict[str, tuple[int, int]]
    tiers: dict[str, tuple[str, ...]]


def _zscore(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    std = a.std(axis=1, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    z: npt.NDArray[np.float64] = (a - a.mean(axis=1, keepdims=True)) / std
    return z


def _fold(
    inputs: BlendInputs, lam_p: npt.NDArray[np.float64], vote_p: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    idx = {e: i for i, e in enumerate(inputs.entry_ids)}
    lam = {e: lam_p[:, idx[e]].copy() for e in INCUMBENTS}
    vote = {e: vote_p[:, idx[e]].copy() for e in INCUMBENTS}
    for child, parent in inputs.rollup.items():
        lam[parent] = lam[parent] + lam_p[:, idx[child]]          # sum-prevalence
        vote[parent] = np.minimum(vote[parent], vote_p[:, idx[child]])  # min-rank
    lam_arr = np.stack([lam[e] for e in INCUMBENTS], axis=1)
    vote_arr = np.stack([vote[e] for e in INCUMBENTS], axis=1)
    return lam_arr, vote_arr


def _positions(
    inputs: BlendInputs, n: int, seed: int, measurable_z: bool = False
) -> npt.NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    li = rng.integers(0, inputs.lambda_samples.shape[0], size=n)
    vi = rng.integers(0, inputs.vote_rank_samples.shape[0], size=n)
    lam_arr, vote_arr = _fold(inputs, inputs.lambda_samples[li], inputs.vote_rank_samples[vi])
    fb_mask = np.array([e in inputs.frame_blind for e in INCUMBENTS])
    if measurable_z:
        keep = ~fb_mask
        ds = np.zeros_like(lam_arr)
        ds[:, keep] = _zscore(lam_arr[:, keep])
        data_score = ds
    else:
        data_score = _zscore(lam_arr)
    vote_score = _zscore(-vote_arr.astype(np.float64))
    w_vote = np.where(fb_mask, 1.0, W_VOTE)
    w_data = np.where(fb_mask, 0.0, 1.0 - W_VOTE)
    blended = w_vote * vote_score + w_data * data_score
    order_idx = np.argsort(-blended, axis=1, kind="stable")  # (score desc, entry_id asc)
    pos = np.empty_like(order_idx)
    pos[np.arange(n)[:, None], order_idx] = np.arange(1, 11)[None, :]
    return pos


def _summarize(pos: npt.NDArray[np.int64]) -> BlendResult:
    col = {e: i for i, e in enumerate(INCUMBENTS)}
    mean = {e: float(pos[:, col[e]].mean()) for e in INCUMBENTS}
    order = tuple(sorted(INCUMBENTS, key=lambda e: (mean[e], e)))
    p3 = {e: float((pos[:, col[e]] <= 3).mean()) for e in INCUMBENTS}
    p5 = {e: float((pos[:, col[e]] <= 5).mean()) for e in INCUMBENTS}
    interval = {
        e: (int(np.percentile(pos[:, col[e]], 5)), int(np.percentile(pos[:, col[e]], 95)))
        for e in INCUMBENTS
    }
    tiers = {"pair": order[:2], "band": order[2:5], "tail": order[5:]}
    return BlendResult(order, mean, p3, p5, interval, tiers)


def blend(inputs: BlendInputs, n: int = DEFAULT_N, seed: int = SEED) -> BlendResult:
    return _summarize(_positions(inputs, n, seed))


def _blend_measurable_z(inputs: BlendInputs, n: int = DEFAULT_N, seed: int = SEED) -> BlendResult:
    """Disclosed alternative: z-score over the measurable entries only (sensitivity check)."""
    return _summarize(_positions(inputs, n, seed, measurable_z=True))
