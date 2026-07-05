"""Committed reconstruction of the probabilistic blend — PROVENANCE, not runtime.

Reconstructs the accepted method after the original scratchpad was lost. Under the linear
data transform it reproduces the recorded order; the log branch shows the tail-only swap
other transforms produce. engine/decide/blend.py is the production reimplementation; this
file is the CI cross-implementation anchor (Task 5) and is hash-frozen in the manifest.
Do not import at runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_INCUMB = tuple(f"LLM{i:02d}" for i in range(1, 11))
_ROLL = {
    "ROLL-CMSB": "LLM01",
    "ROLL-LAPTF": "LLM03",
    "ROLL-CFAS": "LLM04",
    "ROLL-SICG": "LLM05",
}
_W = 0.75
APPROVED = (
    "LLM01",
    "LLM02",
    "LLM06",
    "LLM03",
    "LLM04",
    "LLM10",
    "LLM09",
    "LLM07",
    "LLM08",
    "LLM05",
)


def _repo_root(start: Path) -> Path:
    p = start.resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found")


def _z(a: NDArray) -> NDArray:  # type: ignore[type-arg]
    s = a.std(1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return (a - a.mean(1, keepdims=True)) / s  # type: ignore[no-any-return]


def reconstruct_order(transform: str = "lin", seed: int = 20260520, n: int = 16000) -> list[str]:
    root = _repo_root(Path(__file__))
    base = root / "projects" / "owasp-llm"
    inf = json.loads((base / "cycles/2026/infer/inference_summary.json").read_text())
    rb = json.loads((base / "baselines/2026/rankings_baselines.json").read_text())
    idx = {e: i for i, e in enumerate(inf["entry_ids"])}
    lam = np.load(base / "cycles/2026/infer/lambda_samples.npy")
    vote = np.load(base / "baselines/2026/vote_rank_samples.npy")
    fb = set(rb["not_measurable"])
    rng = np.random.default_rng(seed)
    lp = lam[rng.integers(0, lam.shape[0], n)]
    vp = vote[rng.integers(0, vote.shape[0], n)]
    la = {e: lp[:, idx[e]].copy() for e in _INCUMB}
    vo = {e: vp[:, idx[e]].copy() for e in _INCUMB}
    for c, par in _ROLL.items():
        la[par] = la[par] + lp[:, idx[c]]
        vo[par] = np.minimum(vo[par], vp[:, idx[c]])
    la_a = np.stack([la[e] for e in _INCUMB], 1)
    vo_a = np.stack([vo[e] for e in _INCUMB], 1)
    data = _z(np.log(la_a)) if transform == "log" else _z(la_a)
    vsc = _z(-vo_a.astype(float))
    m = np.array([1.0 if e in fb else 0.0 for e in _INCUMB])
    bl = np.where(m == 1, 1.0, _W) * vsc + np.where(m == 1, 0.0, 1 - _W) * data
    pos = np.empty_like(np.argsort(-bl, 1))
    pos[np.arange(n)[:, None], np.argsort(-bl, 1, kind="stable")] = np.arange(1, 11)[None, :]
    mean = pos.mean(0)
    return [_INCUMB[i] for i in sorted(range(10), key=lambda i: (mean[i], _INCUMB[i]))]


def approved_order() -> tuple[str, ...]:
    return APPROVED
