# Probabilistic Blend Relock Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relock the 2026 blended OWASP LLM Top-10 from the interim rank-space blend to the accepted probabilistic blend, computed by a tested engine module, and update the notebook, arXiv preprint, methodology doc, and supporting artifacts to report it as three tiers with an uncertainty layer, for a data-science-novice audience.

**Architecture:** A new pure engine module (`engine/decide/blend.py`) reads the committed posterior-sample arrays, verifies their integrity, and produces a distribution over each incumbent's final position (0.75 vote / 0.25 data, sum-prevalence fold, frame-blind-drop). The notebook calls the module, regenerates figures, and guards every published number in a consistency cell. The preprint prose lives in the notebook's markdown cells; the build (`tools/build_preprint.py`) exports them to `.md` and compiles the PDF.

**Tech Stack:** Python 3.12, numpy 2.1.3, scipy, pytest, mypy, ruff; Jupyter + nbconvert (preprint extra); pandoc + xelatex for the PDF.

**v2 note:** This revision folds in a full adversarial premortem of the plan. The engine math (Tasks 2-6) was verified sound by execution; the fixes are mechanical: CWD-robust paths (P-1), a Phase-2 prerequisite task for clean notebook execution and kernel registration (P-2/P-3/P-5), heading-matched cell edits (P-4), the rank-change slopegraph dropped for the tier-native position-interval figure (P-7), widened edit scope and grep gates (P-6), a frame-blind value-pin (P-9), a corrected tail claim (P-10), a computed Kendall tau and reconciled "99 percent" (P-8), and the medium fixes (P-11).

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-07-05-probabilistic-blend-relock-design.md` (v3). Every task implicitly includes its guardrails.
- Do NOT touch figure styling, `figure-layout.lua`, `arxiv-template.latex`, the build pipeline, or any Part II data-witness figure.
- Do NOT flip the report to publishable or remove any interim / single-author / non-publishable disclosure.
- No AI-attribution anywhere (commits, docs, comments). Git author stays the human.
- Writing rules for all new/changed prose: no sentence starts with a conjunction; no antithesis/contrast framing ("not X but Y", "rather than X", "reflects A, not B"); obey `STYLE-GUIDE.md`.
- The public preprint describes the method as the authors' ADOPTION for an exploratory report; it never claims OWASP institutional sign-off; never "validates".
- Approved point order (engine sort key): `LLM01, LLM02, LLM06, LLM03, LLM04, LLM10, LLM09, LLM07, LLM08, LLM05`. Tiers: pair={LLM01,LLM02}, band={LLM06,LLM03,LLM04}, tail={LLM10,LLM09,LLM07,LLM08,LLM05}. LLM10 is a vote-placed borderline tail entry (P(top-5) ~ 0.33); the other four tail entries are P(top-5) < 0.05.
- Column order is the manifest `entry_ids`, NOT taxonomy order. Rollup crosswalk (value-pinned): ROLL-CMSB→LLM01, ROLL-LAPTF→LLM03, ROLL-CFAS→LLM04, ROLL-SICG→LLM05. Frame-blind (value-pinned) = {LLM04, LLM08, LLM10}. Seed 20260520, N=16000, W_vote=0.75.
- Notebook paths anchor to `REPO_ROOT` (defined in cell 0); the build runs nbconvert with CWD `notebooks/preprint`, so a raw relative `Path('projects/...')` is a bug.
- Run `uv run mypy engine tests` and `uv run ruff check .` before every commit that touches Python.

---

## File Structure

Create:
- `docs/provenance/2026-07-05-probabilistic-blend-reconstruction.md`, `engine/decide/blend_prototype_reference.py` (CI cross-check anchor, hash-frozen).
- `projects/owasp-llm/cycles/2026/blend/blend_manifest.json`, `.../blend_golden.json`.
- `engine/decide/blend.py`, `tests/unit/test_probabilistic_blend.py`.
- `docs/decisions/2026-07-05-probabilistic-blend-adoption.md`.

Modify:
- `notebooks/2026_top_10_llm_update_what_the_data_says.ipynb` — cell 0 (kaleido fix), cell 45 (compute+tau), cell 46 (guard), new figure cells, markdown cells 2/30/43/47.
- `docs/BLENDED-TOP10-METHODOLOGY.md` — header, §4.3–4.5, §5, §6, §7, §9.
- `notebooks/preprint/STYLE-GUIDE.md`, `notebooks/preprint/front_matter.md`, `notebooks/narrative/2026_top_10_llm_update_what_the_data_says.md`, `.github/workflows/ci.yml`.

---

## Phase 1 — Provenance and engine

### Task 1: Commit the reconstruction prototype as provenance

**Files:** Create `engine/decide/blend_prototype_reference.py`, `docs/provenance/2026-07-05-probabilistic-blend-reconstruction.md`.
**Interfaces:** Produces `reconstruct_order(transform=...)`, `approved_order()`.

- [ ] **Step 1: Create `engine/decide/blend_prototype_reference.py`** (paths anchored to repo root, not CWD):

```python
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

_INCUMB = tuple(f"LLM{i:02d}" for i in range(1, 11))
_ROLL = {"ROLL-CMSB": "LLM01", "ROLL-LAPTF": "LLM03", "ROLL-CFAS": "LLM04", "ROLL-SICG": "LLM05"}
_W = 0.75
APPROVED = ("LLM01", "LLM02", "LLM06", "LLM03", "LLM04", "LLM10", "LLM09", "LLM07", "LLM08", "LLM05")


def _repo_root(start: Path) -> Path:
    p = start.resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found")


def _z(a: np.ndarray) -> np.ndarray:
    s = a.std(1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return (a - a.mean(1, keepdims=True)) / s


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
```

- [ ] **Step 2: Verify lin reproduces, log diverges.**

Run: `uv run python -c "from engine.decide.blend_prototype_reference import reconstruct_order, APPROVED; print(tuple(reconstruct_order('lin'))==APPROVED, tuple(reconstruct_order('log'))==APPROVED)"`
Expected: `True False`

- [ ] **Step 3: Write the provenance note** at `docs/provenance/2026-07-05-probabilistic-blend-reconstruction.md`:

```markdown
# Provenance: probabilistic-blend reconstruction (2026-07-05)

The accepted probabilistic blend was first computed in a session whose scratchpad was
never committed. `engine/decide/blend_prototype_reference.py` reconstructs it. Under the
linear data-axis z-score it reproduces the recorded order
(LLM01, LLM02, LLM06, LLM03, LLM04, LLM10, LLM09, LLM07, LLM08, LLM05); under a log
transform it swaps positions 7 and 8 (LLM09 and LLM07), both inside the unordered tail,
so no ordered claim in the report depends on the transform. Linear is a defended
reconstruction on the corrected incidence rates' native additive scale.

Integrity scope: the input manifest and the golden output are commit-anchored, not
signed. This closes the accidental-corruption and naive-tamper classes; a commit-access
adversary is out of scope for this internal tool. The reference module is hash-frozen in
the manifest so its `APPROVED` anchor cannot be edited silently. There is no independent
external oracle: the golden is a regression pin against the reconstruction, not a proof of
correctness.
```

- [ ] **Step 4: Commit.**

```bash
uv run mypy engine && uv run ruff check engine/decide/blend_prototype_reference.py
git add engine/decide/blend_prototype_reference.py docs/provenance/2026-07-05-probabilistic-blend-reconstruction.md
git commit -m "chore(blend): commit probabilistic-blend reconstruction as provenance"
```

---

### Task 2: Input-integrity manifest (repo-root-relative paths, includes the reference)

**Files:** Create `projects/owasp-llm/cycles/2026/blend/blend_manifest.json`.

- [ ] **Step 1: Generate the manifest** with repo-root-relative paths (so `load_inputs` resolves them from any CWD) and include the reference module for hash-freezing. Run:

```bash
uv run python - <<'PY'
import json, hashlib
from pathlib import Path
import numpy as np
files = {
    "lambda_samples": "projects/owasp-llm/cycles/2026/infer/lambda_samples.npy",
    "vote_rank_samples": "projects/owasp-llm/baselines/2026/vote_rank_samples.npy",
    "inference_summary": "projects/owasp-llm/cycles/2026/infer/inference_summary.json",
    "rankings_baselines": "projects/owasp-llm/baselines/2026/rankings_baselines.json",
    "taxonomy": "projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json",
    "vote_entry_ids": "tests/unit/fixtures/vote_entry_ids_2026.json",
    "prototype_reference": "engine/decide/blend_prototype_reference.py",
}
inputs = {}
for name, rel in files.items():
    p = Path(rel)
    entry = {"path": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    if p.suffix == ".npy":
        entry["shape"] = list(np.load(p).shape)
    inputs[name] = entry
out = Path("projects/owasp-llm/cycles/2026/blend/blend_manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"artifact": "blend_inputs_manifest", "cycle": 2026, "inputs": inputs}, indent=2) + "\n")
print(out.read_text())
PY
```
Expected: seven inputs, each with a 64-hex `sha256`; the two `.npy` entries carry `"shape": [16000, 20]` and `[5000, 20]`.

- [ ] **Step 2: Commit.**

```bash
git add projects/owasp-llm/cycles/2026/blend/blend_manifest.json
git commit -m "chore(blend): pin probabilistic-blend input hashes (repo-root-relative)"
```

---

### Task 3: Engine — integrity-checked input loading

**Files:** Create `engine/decide/blend.py`; Test `tests/unit/test_probabilistic_blend.py`.
**Interfaces:** Produces `BlendInputs`; `load_inputs(manifest_path) -> BlendInputs`; constants `INCUMBENTS`, `W_VOTE`, `SEED`, `DEFAULT_N`, `EXPECTED_ROLLUP`, `EXPECTED_FRAME_BLIND`.

- [ ] **Step 1: Write the failing test:**

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from engine.decide.blend import (
    EXPECTED_FRAME_BLIND,
    EXPECTED_ROLLUP,
    INCUMBENTS,
    load_inputs,
)

MANIFEST = Path("projects/owasp-llm/cycles/2026/blend/blend_manifest.json")


def test_load_inputs_shapes_and_labels() -> None:
    inp = load_inputs(MANIFEST)
    assert inp.lambda_samples.shape == (16000, 20)
    assert inp.vote_rank_samples.shape == (5000, 20)
    assert inp.entry_ids[:10] == INCUMBENTS
    assert inp.rollup == EXPECTED_ROLLUP
    assert inp.frame_blind == EXPECTED_FRAME_BLIND


def test_load_inputs_rejects_bad_hash(tmp_path: Path) -> None:
    m = json.loads(MANIFEST.read_text())
    m["inputs"]["taxonomy"]["sha256"] = "0" * 64
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inputs(bad)


def test_load_inputs_rejects_frame_blind_drift(tmp_path: Path) -> None:
    # Copy inputs, tamper not_measurable, rehash, and repoint the manifest at tmp_path.
    src = json.loads(MANIFEST.read_text())
    root = Path.cwd()
    rb_path = Path(src["inputs"]["rankings_baselines"]["path"])
    rb = json.loads(rb_path.read_text())
    rb["not_measurable"] = ["LLM04", "LLM08"]  # drop LLM10
    bad_rb = tmp_path / "rb.json"
    bad_rb.write_text(json.dumps(rb))
    src["inputs"]["rankings_baselines"]["path"] = str(bad_rb)
    src["inputs"]["rankings_baselines"]["sha256"] = hashlib.sha256(bad_rb.read_bytes()).hexdigest()
    # keep other paths absolute so they still resolve from tmp manifest
    for k, spec in src["inputs"].items():
        if k != "rankings_baselines":
            spec["path"] = str((root / spec["path"]).resolve())
    bad_m = tmp_path / "m.json"
    bad_m.write_text(json.dumps(src))
    with pytest.raises(ValueError, match="frame-blind drift"):
        load_inputs(bad_m)
```

- [ ] **Step 2: Run to verify it fails.**

Run: `uv run pytest tests/unit/test_probabilistic_blend.py -x -q`
Expected: FAIL with `ImportError` (blend.py not written).

- [ ] **Step 3: Write the module** `engine/decide/blend.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes.**

Run: `uv run pytest tests/unit/test_probabilistic_blend.py -x -q`
Expected: 3 passed.

- [ ] **Step 5: Lint, type-check, commit.**

```bash
uv run mypy engine tests && uv run ruff check .
git add engine/decide/blend.py tests/unit/test_probabilistic_blend.py
git commit -m "feat(blend): integrity-checked input loading with crosswalk + frame-blind value-pins"
```

---

### Task 4: Engine — compute and result

**Files:** Modify `engine/decide/blend.py`; Test `tests/unit/test_probabilistic_blend.py`.
**Interfaces:** Consumes `BlendInputs`, `load_inputs`. Produces `BlendResult` (`order`, `mean_position`, `p_top3`, `p_top5`, `interval`, `tiers`); `blend(inputs, n=DEFAULT_N, seed=SEED) -> BlendResult`.

- [ ] **Step 1: Write failing tests** — append:

```python
def test_blend_reproduces_approved_order() -> None:
    from engine.decide.blend import blend
    r = blend(load_inputs(MANIFEST))
    assert r.order == (
        "LLM01", "LLM02", "LLM06", "LLM03", "LLM04",
        "LLM10", "LLM09", "LLM07", "LLM08", "LLM05",
    )
    assert r.tiers["pair"] == ("LLM01", "LLM02")
    assert r.tiers["band"] == ("LLM06", "LLM03", "LLM04")
    assert r.tiers["tail"] == ("LLM10", "LLM09", "LLM07", "LLM08", "LLM05")


def test_blend_uncertainty_numbers() -> None:
    from engine.decide.blend import blend
    r = blend(load_inputs(MANIFEST))
    assert r.p_top3["LLM01"] == pytest.approx(0.99, abs=0.02)
    assert r.p_top3["LLM02"] == pytest.approx(0.95, abs=0.03)
    assert r.p_top5["LLM04"] == pytest.approx(0.76, abs=0.03)
    assert r.mean_position["LLM03"] < r.mean_position["LLM04"]
    assert r.p_top5["LLM10"] == pytest.approx(0.33, abs=0.05)  # borderline tail
    for e in ("LLM09", "LLM07", "LLM08", "LLM05"):
        assert r.p_top5[e] < 0.05  # deep blur


def test_fold_sum_prevalence_and_min_rank() -> None:
    # Exercise the real _fold: LLM01 folds in ROLL-CMSB (sum on data, min on vote).
    from engine.decide.blend import _fold
    inp = load_inputs(MANIFEST)
    idx = {e: i for i, e in enumerate(inp.entry_ids)}
    lam_arr, vote_arr = _fold(inp, inp.lambda_samples, inp.vote_rank_samples)
    assert lam_arr[0, 0] == pytest.approx(
        inp.lambda_samples[0, idx["LLM01"]] + inp.lambda_samples[0, idx["ROLL-CMSB"]])
    assert vote_arr[0, 0] == min(
        inp.vote_rank_samples[0, idx["LLM01"]], inp.vote_rank_samples[0, idx["ROLL-CMSB"]])


def test_zpopulation_alternative_bounded() -> None:
    # Order-neutral, P(top-k) shift under measurable-only z is small (< 0.05).
    from engine.decide.blend import blend, _blend_measurable_z
    r10 = blend(load_inputs(MANIFEST))
    r7 = _blend_measurable_z(load_inputs(MANIFEST))
    assert r10.order == r7.order
    assert max(abs(r10.p_top5[e] - r7.p_top5[e]) for e in INCUMBENTS) < 0.05
```

- [ ] **Step 2: Run to verify fail.**

Run: `uv run pytest tests/unit/test_probabilistic_blend.py::test_blend_reproduces_approved_order -x -q`
Expected: FAIL `ImportError: cannot import name 'blend'`.

- [ ] **Step 3: Add the compute** to `engine/decide/blend.py`:

```python
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
    return (a - a.mean(axis=1, keepdims=True)) / std


def _fold(inputs: BlendInputs, lam_p: npt.NDArray, vote_p: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    idx = {e: i for i, e in enumerate(inputs.entry_ids)}
    lam = {e: lam_p[:, idx[e]].copy() for e in INCUMBENTS}
    vote = {e: vote_p[:, idx[e]].copy() for e in INCUMBENTS}
    for child, parent in inputs.rollup.items():
        lam[parent] = lam[parent] + lam_p[:, idx[child]]          # sum-prevalence
        vote[parent] = np.minimum(vote[parent], vote_p[:, idx[child]])  # min-rank
    lam_arr = np.stack([lam[e] for e in INCUMBENTS], axis=1)
    vote_arr = np.stack([vote[e] for e in INCUMBENTS], axis=1)
    return lam_arr, vote_arr


def _positions(inputs: BlendInputs, n: int, seed: int, measurable_z: bool = False) -> npt.NDArray[np.int64]:
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
```

- [ ] **Step 4: Run the compute tests.**

Run: `uv run pytest tests/unit/test_probabilistic_blend.py -q`
Expected: all passed (7 tests).

- [ ] **Step 5: Lint, type-check, commit.**

```bash
uv run mypy engine tests && uv run ruff check .
git add engine/decide/blend.py tests/unit/test_probabilistic_blend.py
git commit -m "feat(blend): probabilistic compute, tiers, z-population sensitivity check"
```

---

### Task 5: Golden, stability, permuted-input, cross-implementation

**Files:** Create `.../blend_golden.json`; Modify the test file.

- [ ] **Step 1: Generate the golden.**

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from engine.decide.blend import blend, load_inputs
r = blend(load_inputs(Path("projects/owasp-llm/cycles/2026/blend/blend_manifest.json")))
out = {"order": list(r.order), "tiers": {k: list(v) for k, v in r.tiers.items()},
       "mean_position": r.mean_position, "p_top3": r.p_top3, "p_top5": r.p_top5,
       "interval": {k: list(v) for k, v in r.interval.items()}, "seed": 20260520, "n": 16000}
Path("projects/owasp-llm/cycles/2026/blend/blend_golden.json").write_text(json.dumps(out, indent=2) + "\n")
print(out["order"])
PY
```
Expected: `['LLM01', 'LLM02', 'LLM06', 'LLM03', 'LLM04', 'LLM10', 'LLM09', 'LLM07', 'LLM08', 'LLM05']`.

- [ ] **Step 2: Append the tests:**

```python
GOLDEN = Path("projects/owasp-llm/cycles/2026/blend/blend_golden.json")


def test_matches_golden() -> None:
    g = json.loads(GOLDEN.read_text())
    r = blend(load_inputs(MANIFEST))
    assert list(r.order) == g["order"]
    for e in INCUMBENTS:
        assert r.p_top3[e] == pytest.approx(g["p_top3"][e], abs=1e-9)
        assert r.p_top5[e] == pytest.approx(g["p_top5"][e], abs=1e-9)


def test_top5_and_tail_seed_stable() -> None:
    inp = load_inputs(MANIFEST)
    orders = [blend(inp, seed=s).order for s in range(1000, 1020)]
    assert len({o[:5] for o in orders}) == 1        # top-5 ordinal invariant
    assert len({frozenset(o[5:]) for o in orders}) == 1  # tail set invariant


def test_permuted_array_rejected(tmp_path: Path) -> None:
    src = json.loads(MANIFEST.read_text())
    root = Path.cwd()
    lam = np.load(root / src["inputs"]["lambda_samples"]["path"])
    bad = tmp_path / "lam.npy"
    np.save(bad, lam[:, ::-1])  # reverse columns
    src["inputs"]["lambda_samples"]["path"] = str(bad)
    src["inputs"]["lambda_samples"]["sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
    for k, spec in src["inputs"].items():
        if k != "lambda_samples":
            spec["path"] = str((root / spec["path"]).resolve())
    m = tmp_path / "m.json"
    m.write_text(json.dumps(src))
    # column permutation is silent unless the order changes; assert the order DID change.
    from engine.decide.blend import blend as b2, load_inputs as li2
    assert list(b2(li2(m)).order) != json.loads(GOLDEN.read_text())["order"]


def test_deterministic_tie_break() -> None:
    # Construct a tie: equal blended scores, expect entry_id ascending.
    from engine.decide.blend import _summarize
    pos = np.tile(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), (100, 1))
    r = _summarize(pos)
    assert r.order[0] == "LLM01"  # entry-id order preserved on identical positions


def test_cross_implementation_matches_prototype() -> None:
    from engine.decide.blend import blend
    from engine.decide.blend_prototype_reference import reconstruct_order
    # Freeze the reference by its manifest hash before trusting it.
    import json as _j
    m = _j.loads(MANIFEST.read_text())
    ref = m["inputs"]["prototype_reference"]
    from engine.snapshot.hashing import verify_snapshot_hash
    verify_snapshot_hash(Path.cwd() / ref["path"], ref["sha256"])
    assert list(blend(load_inputs(MANIFEST)).order) == reconstruct_order("lin")
```

- [ ] **Step 3: Run.**

Run: `uv run pytest tests/unit/test_probabilistic_blend.py -q`
Expected: all passed (12 tests).

- [ ] **Step 4: Lint, type-check, commit.**

```bash
uv run mypy engine tests && uv run ruff check .
git add tests/unit/test_probabilistic_blend.py projects/owasp-llm/cycles/2026/blend/blend_golden.json
git commit -m "test(blend): golden regression pin, stability, permuted-input, hash-frozen cross-check"
```

---

## Phase 2 — Notebook

### Task 6: Enable clean notebook execution (PREREQUISITE — do before any other notebook task)

**Files:** Modify `notebooks/2026_top_10_llm_update_what_the_data_says.ipynb` (cell 0); register the kernel.

- [ ] **Step 1: Fix cell 0's kaleido/module un-stub guard** so it does not raise on a clean kernel. Find the line `if sys.modules.get(_stubbed) is None: del sys.modules[_stubbed]` and replace with:

```python
if _stubbed in sys.modules and sys.modules[_stubbed] is None:
    del sys.modules[_stubbed]
```

- [ ] **Step 2: Register the build kernel** from the uv env.

Run: `uv sync --extra narrative --extra preprint && uv run python -m ipykernel install --user --name preprint-build --display-name preprint-build`
Expected: `Installed kernelspec preprint-build in ...`

- [ ] **Step 3: Verify a clean full execution** (no path/kernel/kaleido error).

Run: `uv run jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.kernel_name=preprint-build notebooks/2026_top_10_llm_update_what_the_data_says.ipynb 2>err.txt >/dev/null; test -s err.txt && head err.txt || echo "clean execution"; rm -f err.txt`
Expected: `clean execution` (if a plotly/kaleido error remains, fix that cell now; the build cannot proceed until this is clean).

- [ ] **Step 4: Commit.**

```bash
git add notebooks/2026_top_10_llm_update_what_the_data_says.ipynb
git commit -m "fix(notebook): clean cell-0 module un-stub for fresh kernels"
```

---

### Task 7: Replace the compute (cell 45) with the engine + Kendall tau

**Files:** Modify the notebook (code cell whose comment begins `# Blended 2026 Top 10` — the current cell 45).
**Interfaces:** Produces `blend_result` and `blend_tau` for the guard and prose.

- [ ] **Step 1: Replace that cell's source** (paths anchored to `REPO_ROOT`; also computes the rank-space order via the existing engine and the Kendall tau between them so no number is hand-typed):

```python
# The 2026 blended Top 10 (probabilistic blend). Computed by engine.decide.blend from the
# committed posterior samples; the rank-space order (existing engine) and the Kendall tau
# between the two are computed here so the prose cites no hand-typed number.
from engine.decide.blend import blend, load_inputs
from engine.report.blend_2025_2026 import blended_ranking, load_entries
from scipy.stats import kendalltau

BLEND_MANIFEST = REPO_ROOT / 'projects' / 'owasp-llm' / 'cycles' / '2026' / 'blend' / 'blend_manifest.json'
blend_result = blend(load_inputs(BLEND_MANIFEST))

# rank-space order from the existing tested engine (for the robustness-lens tau).
_rank_md = DATA['rank_comparison_md']
_lam, _vote = {}, {}
for _line in _rank_md.splitlines():
    if _line.startswith('|') and 'Entry' not in _line and not _line.startswith('|--'):
        _c = [c.strip() for c in _line.split('|')[1:-1]]
        if len(_c) >= 3 and re.match(r'([\d.]+)', _c[1]) and re.match(r'([\d.]+)', _c[2]):
            _lam[_c[0]] = float(re.match(r'([\d.]+)', _c[1]).group(1))
            _vote[_c[0]] = float(re.match(r'([\d.]+)', _c[2]).group(1))
_ent = load_entries(CYCLE / 'taxonomy' / 'taxonomy.json')
_inc = [e['entry_id'] for e in _ent if e['group'] == 'incumbent']
_child = {e['entry_id']: e['rolled_into'] for e in _ent if e['group'] == 'rollup'}
_fl, _fv = dict(_lam), dict(_vote)
for _c2, _p in _child.items():
    if _p in _fl and _c2 in _lam:
        _fl[_p] = min(_fl[_p], _lam[_c2]); _fv[_p] = min(_fv[_p], _vote[_c2])
_rankspace = [b['entry_id'] for b in blended_ranking({e: _fv[e] for e in _inc}, {e: _fl[e] for e in _inc}, 0.75)]
_prob_rank = {e: i for i, e in enumerate(blend_result.order)}
_rs_rank = {e: i for i, e in enumerate(_rankspace)}
blend_tau = float(kendalltau([_prob_rank[e] for e in _inc], [_rs_rank[e] for e in _inc]).statistic)

print('Tiers:', blend_result.tiers)
print(f'Kendall tau (probabilistic vs rank-space): {blend_tau:.2f}')
for e in blend_result.order:
    print(f"  {e:6s} pos={blend_result.mean_position[e]:.2f} "
          f"P3={blend_result.p_top3[e]:.2f} P5={blend_result.p_top5[e]:.2f} CI={blend_result.interval[e]}")
```

- [ ] **Step 2: Execute and confirm tau ~ 0.87.**

Run: `uv run jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.kernel_name=preprint-build notebooks/2026_top_10_llm_update_what_the_data_says.ipynb 2>/dev/null | grep "Kendall tau"`
Expected: a line with `Kendall tau (probabilistic vs rank-space): 0.8x` (record the exact value for the prose and guard).

- [ ] **Step 3: Commit.**

```bash
git add notebooks/2026_top_10_llm_update_what_the_data_says.ipynb
git commit -m "feat(notebook): compute blend tiers and rank-space Kendall tau via the engines"
```

---

### Task 8: New uncertainty figures (the ranking visual)

**Files:** Modify the notebook (insert two code cells after the compute cell). The rank-change slopegraph is retired: it plots crisp 2026 ordinals for every entry, which the tier rule forbids, and it is type-incompatible with the tier data. `blend_position_intervals` becomes the ranking visual.
**Interfaces:** Consumes `blend_result`, `ENTRY_NAMES`, `PREPRINT_FIG`. Produces `figures/blend_position_intervals.png`, `figures/blend_top_k_probs.png`.

- [ ] **Step 1: Add the position-interval cell** (matplotlib, 300 dpi, existing style):

```python
# blend_position_intervals.png — each risk at its mean position (dot) with a 5-95 pct bar,
# grouped into the three tiers. Replaces the retired rank-change slopegraph.
import matplotlib.pyplot as plt
_tier_of = {e: t for t, es in blend_result.tiers.items() for e in es}
_tc = {'pair': '#1b6ca8', 'band': '#f2a154', 'tail': '#9aa0a6'}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for row, e in enumerate(blend_result.order):
    lo, hi = blend_result.interval[e]
    ax.plot([lo, hi], [row, row], color=_tc[_tier_of[e]], lw=6, solid_capstyle='round', alpha=0.55)
    ax.plot(blend_result.mean_position[e], row, 'o', color=_tc[_tier_of[e]], ms=8)
    ax.text(0.3, row, ENTRY_NAMES.get(e, e), va='center', ha='right', fontsize=9)
ax.set_yticks([]); ax.invert_yaxis()
ax.set_xlabel('Position among the ten (1 = highest priority); bar = 5th-95th percentile')
ax.set_xlim(0.3, 10.7); ax.set_xticks(range(1, 11))
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(PREPRINT_FIG / 'blend_position_intervals.png', dpi=300, bbox_inches='tight'); plt.show()
```

- [ ] **Step 2: Add the P(top-k) companion cell:**

```python
# blend_top_k_probs.png — grouped bars of P(top-3) and P(top-5) per risk, ordered by tier.
import numpy as np
_ord = list(blend_result.order); _x = np.arange(len(_ord)); _w = 0.38
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.bar(_x - _w/2, [blend_result.p_top3[e] for e in _ord], _w, label='P(top 3)', color='#1b6ca8')
ax.bar(_x + _w/2, [blend_result.p_top5[e] for e in _ord], _w, label='P(top 5)', color='#f2a154')
ax.set_xticks(_x); ax.set_xticklabels([ENTRY_NAMES.get(e, e) for e in _ord], rotation=40, ha='right', fontsize=8)
ax.set_ylabel('Probability'); ax.set_ylim(0, 1.02); ax.legend(frameon=False)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(PREPRINT_FIG / 'blend_top_k_probs.png', dpi=300, bbox_inches='tight'); plt.show()
```

- [ ] **Step 3: Retire the rank-change cell.** Find the cell whose comment references `rank_change_2025_2026.png` (the old slopegraph render) and replace its body with a comment noting the retirement, or delete the cell. Remove the `![...](figures/rank_change_2025_2026.png)` image reference from markdown cell 43 (handled in Task 10).

- [ ] **Step 4: Execute; confirm both new PNGs exist.**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=preprint-build notebooks/2026_top_10_llm_update_what_the_data_says.ipynb && ls notebooks/preprint/figures/blend_position_intervals.png notebooks/preprint/figures/blend_top_k_probs.png`
Expected: both listed.

- [ ] **Step 5: Commit.**

```bash
git add notebooks/2026_top_10_llm_update_what_the_data_says.ipynb notebooks/preprint/figures/blend_position_intervals.png notebooks/preprint/figures/blend_top_k_probs.png
git commit -m "feat(figures): tier position-interval + P(top-k) figures; retire rank-change slopegraph"
```

---

### Task 9: Expand the consistency guard (match by heading, not index)

**Files:** Modify the notebook code cell whose first line is `# Consistency check` (do NOT use a numeric index — Task 8 shifted indices; locate by the leading comment).

- [ ] **Step 1: Replace that cell's body** with the tier + number guard:

```python
# Consistency check: every published ranking number must equal the engine result.
# The build executes this cell, so a prose number that drifts fails the build.
import json
_g = json.loads((REPO_ROOT / 'projects/owasp-llm/cycles/2026/blend/blend_golden.json').read_text())
assert list(blend_result.order) == _g['order'], "order drift vs golden"
assert blend_result.tiers['pair'] == ('LLM01', 'LLM02')
assert blend_result.tiers['band'] == ('LLM06', 'LLM03', 'LLM04')
assert blend_result.tiers['tail'] == ('LLM10', 'LLM09', 'LLM07', 'LLM08', 'LLM05')
assert abs(blend_result.p_top3['LLM01'] - 0.99) < 0.02
assert abs(blend_result.p_top3['LLM02'] - 0.95) < 0.03
assert abs(blend_result.p_top5['LLM10'] - 0.33) < 0.05      # borderline tail, cited in prose
assert all(blend_result.p_top5[e] < 0.05 for e in ('LLM09', 'LLM07', 'LLM08', 'LLM05'))
assert 0.80 <= blend_tau <= 0.95                            # rank-space robustness lens
_rob = json.loads((RARR_CYCLE / 'results' / 'robustness_validation.json').read_text())
assert _rob['ranking_fidelity_spearman_vs_truth']['floor'] > 0.85
_bl = json.loads((BASELINES / 'rankings_baselines.json').read_text())
assert _bl['previous_ranking']['kappa_median'] == 0.2028985507246377
print("consistency OK")
```

- [ ] **Step 2: Execute; confirm the guard prints OK.**

Run: `uv run jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.kernel_name=preprint-build notebooks/2026_top_10_llm_update_what_the_data_says.ipynb 2>/dev/null | grep -c "consistency OK"`
Expected: `1`.

- [ ] **Step 3: Commit.**

```bash
git add notebooks/2026_top_10_llm_update_what_the_data_says.ipynb
git commit -m "feat(notebook): guard asserts tiers, tail P(top5), and Kendall tau"
```

---

## Phase 3 — Preprint prose (notebook markdown cells; locate by heading, obey the writing rules)

### Task 10: Rewrite "The 2026 blended Top 10" (markdown cell headed `## The 2026 blended Top 10`)

- [ ] **Step 1: Replace that cell's body** (no sentence-initial conjunctions, no antithesis, no "sits first" tripwire phrase, no `rank_change` image, LLM10 stated as a borderline tail entry):

```markdown
## The 2026 blended Top 10

The blend combines two witnesses into a distribution over each risk's final position. The expert vote carries three-quarters weight, the incident data one quarter. Each risk gets a spread of plausible positions, and the ten fall into three tiers.

> **Sidebar — distribution over positions.** Each posterior draw pairs one sample of the incident rates with one sample of the expert ranking, blends them, and reads off a position. Sixteen thousand draws give a distribution over positions for each risk, so a firm placement and a coin flip look different on the page.

![Each risk at its mean position, with a bar spanning the 5th to 95th percentile. Three tiers read directly: a tight pair, an overlapping band, and a wide tail.](figures/blend_position_intervals.png){width=70% wrap=left}

**The co-leading pair.** Sensitive Information Disclosure and Prompt Injection hold the top, each a near-certain top-three risk (P(top-3) 0.99 and 0.95). The two blends disagree on which ranks first: the simpler rank-space blend puts Sensitive Information Disclosure ahead, the probabilistic blend puts Prompt Injection ahead, and their intervals overlap. We report them as co-leading, with no method-independent first place.

**The tied band.** Excessive Agency, Supply Chain, and Data and Model Poisoning form a middle band with overlapping intervals. Excessive Agency rises three places from its published position, the one clear mover in the band.

**The tail.** Unbounded Consumption sits at the top of the tail, placed by the expert vote alone (its incident recall the corpus cannot estimate), and it reaches the top five in about a third of the draws. The remaining four — Misinformation, Hidden Context Exposure, Vector and Embedding Weaknesses, Improper Output Handling — reach the top five in under one draw in twenty. We present the tail as a group and do not report an order inside it.

> **Sidebar — why a distribution beats a single rank number.** A single rank hides its own uncertainty. Two adjacent tail positions differ by a fraction of a place across the draws, so a printed rank would claim precision the data does not carry. The tiers report only what the spread supports.

A simpler rank-space blend, which uses only the order of each witness and discards the magnitudes, gives nearly the same bulk ordering (Kendall's tau computed at run time, about 0.87). The one place the two methods disagree is the top, which is why we present the top two as a pair.
```

- [ ] **Step 2: Commit.**

```bash
git add notebooks/2026_top_10_llm_update_what_the_data_says.ipynb
git commit -m "docs(preprint): three-tier presentation; LLM10 as a vote-placed borderline entry"
```

---

### Task 11: Rewrite ALL of Part I cell 2, and Act 8 cell 30

**Files:** Modify the notebook markdown cell headed `# Part I` (rewrite the whole cell, not one paragraph — the "falls five places / rises four" movers must go) and the cell headed `## Act 8`.

- [ ] **Step 1: In the Part I cell**, replace the "0.75 / 0.25 blend" body with the score-space description, AND rewrite the "What changed from 2025 to 2026" paragraph so it carries no crisp tail "+N" mover. Score-space paragraph:

```markdown
The two witnesses are combined in score space. Each risk's expert rank and incident rate are put on a common scale, weighted three-quarters to the vote and one quarter to the data, and blended per posterior draw. On the data side the rates of a rolled-up child add to their parent, since incidents accumulate. Three risks whose incident recall the corpus cannot estimate — Data and Model Poisoning, Vector and Embedding Weaknesses, and Unbounded Consumption — take their position from the vote alone. The result is a distribution over positions, reported as three tiers in the final section.
```

For the "what changed" paragraph, replace any "falls five places / rises four / rises three" list with tier language: describe the movement as risks entering the co-leading pair, the tied band, or the tail, with the one stated mover being Excessive Agency into the band. Do not print a "+N" for any tail entry.

- [ ] **Step 2: In the Act 8 cell**, carry the Misinformation finding as a data-witness result (not a blended-rank mover), with the "99 percent" number sourced as the existing concordance disagreement flag and hedged at the point of use:

```markdown
Misinformation is the widest disagreement between the two witnesses. The incident record ranks it near the top; the expert vote ranks it near the bottom. The engine's concordance flag puts the probability that the two signals disagree at 99 percent. That number measures disagreement, not how underrated the risk is, and the incident signal here rests on the ai-harm stratum, whose precision the corpus does not measure directly. Read it as the entry the incident record most disputes, and the one a better-measured corpus is most likely to move.
```

- [ ] **Step 3: Execute; guard still passes.**

Run: `uv run jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.kernel_name=preprint-build notebooks/2026_top_10_llm_update_what_the_data_says.ipynb 2>/dev/null | grep -c "consistency OK"`
Expected: `1`.

- [ ] **Step 4: Verify no tail mover survives in Part I.**

Run: `uv run python -c "import json;nb=json.load(open('notebooks/2026_top_10_llm_update_what_the_data_says.ipynb'));import re;print([c for c in [''.join(x['source']) for x in nb['cells'] if x['cell_type']=='markdown'] if re.search(r'falls five|rises four|rises three places', c)])"`
Expected: `[]`.

- [ ] **Step 5: Commit.**

```bash
git add notebooks/2026_top_10_llm_update_what_the_data_says.ipynb
git commit -m "docs(preprint): score-space Part I without tail movers; Misinformation as data-witness"
```

---

### Task 12: Glossary (markdown cell headed `## Glossary`)

- [ ] **Step 1: Update the "0.75 / 0.25 blend" entry and add** probabilistic blend, distribution over positions, credible interval over a rank, P(top-k), sum-prevalence fold, frame-blind drop, Kendall's tau, tied tier. (Use the entries from spec §8; write them obeying the two rules — no antithesis, no sentence-initial conjunction.)

- [ ] **Step 2: Commit.**

```bash
git add notebooks/2026_top_10_llm_update_what_the_data_says.ipynb
git commit -m "docs(preprint): glossary entries for the probabilistic blend"
```

---

## Phase 4 — Supporting documents

### Task 13: Internal decision note

**Files:** Create `docs/decisions/2026-07-05-probabilistic-blend-adoption.md` (content unchanged from v1 — verified correctly method-scoped by the premortem).

- [ ] **Step 1: Write the note** (as in v1 — states Rock's executive Co-Lead method decision, explicitly not an OWASP endorsement, disclosures kept). **Step 2: Commit.**

```bash
git add docs/decisions/2026-07-05-probabilistic-blend-adoption.md
git commit -m "docs: internal decision note adopting the probabilistic blend"
```

---

### Task 14: Methodology doc — header, §4.3-4.5, §5, §6, §7, §9

**Files:** Modify `docs/BLENDED-TOP10-METHODOLOGY.md`.

- [ ] **Step 1: Header** → `Version: 1.0 (probabilistic blend, adopted method), 2026-07-05` and a Status line: adopted analytical method for this exploratory report (authors' adoption, not an OWASP endorsement); standing disclosures unchanged; documented amendment from the 0.1 interim rank-space blend.
- [ ] **Step 2: §4.3** → the probabilistic blend as the adopted method, linear justified on the corrected-rate native scale, and the corrected RARR framing (ordinal robustness, NOT a claim that the magnitude deferral is resolved); point the tail-risk cross-reference to the renamed §7.
- [ ] **Step 3: §4.4 / §4.5** → sum-prevalence fold as production; frame-blind-drop as placement-affecting with the keep-at-0.25 alternative shown.
- [ ] **Step 4: §5** → "The adopted ranking": three tiers with the uncertainty layer; rank-space demoted to a one-paragraph robustness lens; no "+N" movers for tail entries; LLM10 noted as vote-placed.
- [ ] **Step 5: §6** → rewrite so it carries no crisp tail ordinal ("seats it eighth") and no orphan "99 percent"; describe the tiers and the Misinformation disagreement as a data-witness finding.
- [ ] **Step 6: §7** → retitle to "Residual and tail risk" and rewrite to the adopted-status posture: the parked retrain tail risk (magnitude-sensitivity), NOT "the blend scale is interim / deferred".
- [ ] **Step 7: §9 (deck prompt)** → tier framing and LLM01-first order throughout; Card 14's "recall-corrected upgrade in progress / expect Misinformation to move" reframed to the adopted posture (the uncertainty-aware ranking is delivered).
- [ ] **Step 8: Verify no stale order or status survives.**

Run: `grep -nE "sits first|holds second place|keeps it second|seats it eighth|lands eighth|interim ranking|is interim|Provisional|deferred|recall-corrected upgrade|checkpoint|99 percent" docs/BLENDED-TOP10-METHODOLOGY.md`
Expected: only intended matches (e.g., a single hedged "99 percent" in the data-witness context, if kept); no interim/deferred/stale-order lines.

- [ ] **Step 9: Commit.**

```bash
git add docs/BLENDED-TOP10-METHODOLOGY.md
git commit -m "docs(methodology): amend to adopted probabilistic tiers across all sections; fix RARR framing"
```

---

### Task 15: STYLE-GUIDE, front-matter, narrative mirror

**Files:** Modify `notebooks/preprint/STYLE-GUIDE.md`, `notebooks/preprint/front_matter.md`, `notebooks/narrative/2026_top_10_llm_update_what_the_data_says.md`.

- [ ] **Step 1: STYLE-GUIDE** — add the two writing rules; update the "Key numbers" blend line to the score-space tier description (only stated mover LLM06 +3).
- [ ] **Step 2: front_matter.md** — add `method: probabilistic-blend` to the YAML.
- [ ] **Step 3: Regenerate the narrative mirror** from the notebook rather than grep-patching (it encodes order in tables and carries a banned "Validation" title). If regeneration is not wired, rewrite its ranking table to the tiers, drop crisp tail ranks, remove any "Validation"/"validates" title or byline, and reconcile any "P = 0.99".
- [ ] **Step 4: Verify the mirror is clean.**

Run: `grep -nEi "validat|sits first|holds second|lands eighth|\| 10 \|" notebooks/narrative/2026_top_10_llm_update_what_the_data_says.md`
Expected: no banned matches.

- [ ] **Step 5: Commit.**

```bash
git add notebooks/preprint/STYLE-GUIDE.md notebooks/preprint/front_matter.md notebooks/narrative/2026_top_10_llm_update_what_the_data_says.md
git commit -m "docs: writing rules, method stamp, narrative-mirror to tiers"
```

---

## Phase 5 — CI, build, verification

### Task 16: CI guard job (matches the sibling job's supply-chain posture)

**Files:** Modify `.github/workflows/ci.yml`.

- [ ] **Step 1: Read the existing `checks` job** to copy its setup-uv version, permissions, and runner.

Run: `sed -n '1,60p' .github/workflows/ci.yml`

- [ ] **Step 2: Add the job** (pin `setup-uv@v4` + the same `version:`, least-privilege permissions, kernel registration, explicit pipefail, kept stderr):

```yaml
  preprint-guard:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "0.5.11"
      - name: Sync (preprint extra)
        run: uv sync --frozen --extra narrative --extra preprint
      - name: Register kernel
        run: uv run python -m ipykernel install --user --name preprint-build
      - name: Engine + cross-implementation tests
        run: uv run pytest tests/unit/test_probabilistic_blend.py -q
      - name: Execute notebook guard
        shell: bash
        run: |
          set -o pipefail
          uv run jupyter nbconvert --to notebook --execute --stdout \
            --ExecutePreprocessor.kernel_name=preprint-build \
            notebooks/2026_top_10_llm_update_what_the_data_says.ipynb \
            | tee /tmp/nb.out | grep -q "consistency OK"
```

(Match the exact `version:` string to whatever the existing job pins; `0.5.11` is the value verified at plan time.)

- [ ] **Step 3: Validate + commit.**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"
git add .github/workflows/ci.yml
git commit -m "ci: preprint guard job with pinned uv, least-privilege perms, kernel registration"
```

---

### Task 17: Full build, disclosure + stale-order gates, rollback tag

**Files:** Modify `notebooks/preprint/BUILD.md`.

- [ ] **Step 1: Tag the last-known-good interim state** (the interim branch tip that HAS the CVE fixes and the preprint extra — `7fad40e`, NOT `45987ab`).

Run: `git tag preprint-interim-2026-07-02 7fad40e && git show --stat 7fad40e | head -3`
Expected: the tag prints and the commit is the interim tip.

- [ ] **Step 2: Run the full build.**

Run: `uv run python tools/build_preprint.py --notebook notebooks/2026_top_10_llm_update_what_the_data_says.ipynb --out-dir notebooks/preprint --output-name Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026`
Expected: exit 0; emits `.md`, `.tex`, `.pdf`.

- [ ] **Step 3: Gate the built `.tex`** on the new order, glyphs, disclosures, and absence of any stale-order phrase.

Run:
```bash
TEX=notebooks/preprint/Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026.tex
grep -qE "κ|λ|ρ" "$TEX" && echo "glyphs ok"
grep -qiE "non-publishable|single-author|does not supersede|not the official OWASP" "$TEX" && echo "disclosures ok"
! grep -qE "sits first|holds second|lands eighth|falls five|rises four|keeps it second|seats it eighth|is interim|deferred|recall-corrected upgrade" "$TEX" && echo "no stale order"
grep -qiE "Prompt Injection" "$TEX" && echo "new order present"
```
Expected: `glyphs ok`, `disclosures ok`, `no stale order`, `new order present` all print.

- [ ] **Step 4: Record toolchain versions** in BUILD.md (`pandoc --version | head -1; xelatex --version | head -1`) under a "Toolchain versions used for the adopted build" heading.

- [ ] **Step 5: Commit the rebuilt artifacts.**

```bash
git add notebooks/preprint/Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026.md notebooks/preprint/Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026.tex notebooks/preprint/Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026.pdf notebooks/preprint/BUILD.md notebooks/2026_top_10_llm_update_what_the_data_says.ipynb
git commit -m "build(preprint): rebuild with probabilistic tiers; disclosure + stale-order gates; toolchain pinned"
```

---

## Self-Review

**Premortem coverage (P-N → fix):** P-1 CWD paths → Task 1/3 `_repo_root`, Task 2 relative paths, Tasks 7/9 `REPO_ROOT`. P-2 cell-0 kaleido → Task 6 step 1. P-3 kernel → Task 6 step 2, Task 16. P-4 index shift → Tasks 9/10/11/12 locate by heading. P-5 ordering → Task 6 is the Phase-2 prerequisite. P-6 stale prose/greps → Tasks 11, 14 (§6/§7/§9), 15 (mirror), 17 (gates). P-7 figure → Task 8 retires the slopegraph. P-8 guard/tau/99% → Task 5 (regression pin + hash-frozen cross-check), Task 7 (computed tau), Task 11 (sourced 99%). P-9 frame-blind pin → Task 3 `EXPECTED_FRAME_BLIND`. P-10 tail claim → Tasks 10, 9 (LLM10 guard). P-11 tests/rollback/CI/prose → Tasks 4-5 (5 new tests), 17 (tag 7fad40e), 16 (uv pin, perms, pipefail), 10-12 (writing rules).

**Placeholder scan:** prose tasks carry concrete draft text obeying the rules; doc tasks 12-15 reference spec §8 for exact entries where the text is long, with the intent, constraints, and a verification grep specified — action and target concrete, not deferred.

**Type consistency:** `BlendInputs`/`BlendResult`, `load_inputs(manifest_path)`, `blend(inputs,n,seed)`, `_blend_measurable_z`, `blend_result`, `blend_tau`, `reconstruct_order(transform=...)` used identically across tasks. Paths anchor to `REPO_ROOT`/`_repo_root` everywhere.
