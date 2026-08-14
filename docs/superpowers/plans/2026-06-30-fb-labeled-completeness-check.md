# F-B: labeled_incidents completeness check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a DISTINCT completeness guard — removed when Unit A's OOS policy (a) replaced the coverage-guard raise with `classifier_labels.get(incident_id, OUT_OF_SCOPE)` — so a genuinely truncated/partial `labeled_incidents.json` raises loudly instead of silently scoring its absent incidents as out-of-scope recall-misses, while genuine OOS continues to score as a miss (policy (a) preserved).

**Architecture:** OOS is encoded as *absence* from `labeled_incidents.json`, which overloads absence (genuine-OOS vs never-processed). We disambiguate against an authoritative record that is independent of the (truncatable) classifier output: the **pinned corpus snapshot** (`manifest.snapshot_hash` → `corpora/<corpus>/<snapshot_hash>/incidents.json`). The classify producer writes a count-form completion marker `classify_coverage.json` = `{snapshot_hash, n_corpus, n_in_scope, n_oos}` with the write-time invariant `n_in_scope + n_oos == n_corpus`. At the two recall-flip sites (`execute_infer_phase`, `cal_tally`) a verifier reconciles the marker to the pinned snapshot and to `labeled_incidents.json`; a **proven** coverage gap raises `LabeledIncidentsIncompleteError`. No pinned snapshot resolvable (synthetic / minimal-fixture cycles) ⇒ no-op.

**Tech Stack:** Python 3.12, stdlib `json`/`dataclasses`/`pathlib`, pytest, click (CLI), numpyro/JAX (infer phase, untouched here).

## Global Constraints

- **No AI/Claude/Anthropic attribution** in any commit message, PR title/body, code comment, or any GitHub-visible content. Commits read as if authored entirely by the user.
- **CI gate (from `.github/workflows/ci.yml`), run the EXACT commands:** `uv run ruff check .` (whole repo) + `uv run mypy engine tests` (engine AND tests) **before every commit**; **FULL `uv run pytest -q`** (no `-k` subset) **before any push**.
- **mypy is strict** (`pyproject.toml:71`): every test function needs `-> None`; helpers need typed args; `PreregManifest(**dict)` needs a typed dict or the `_make_manifest` factory. Under `from __future__ import annotations`, do NOT use quoted annotations (ruff UP037); use `X | Y` unions (ruff UP038, e.g. `isinstance(x, int | float)`).
- **F4 pin `tests/unit/test_recall_single_label_semantics.py` MUST stay green.** This plan does NOT touch `engine/calibrate/tally.py` recall semantics; if any step would, stop and reconsider.
- **`labeled_incidents.json` stays in-scope-only** (OOS = absence). This plan does NOT add OOS rows and does NOT change `_build_counts_from_labeled`. The count-form marker is the coverage record.
- **Two-stage review per task** (SPEC + QUALITY), then push to **PR #22** and verify **CI green**.
- Branch: `plan7/engine-upgrade-recall-pl`. Ledger: `docs/superpowers/plans/LESSONS-rarr.md`; progress: `.superpowers/sdd/progress.md` (local-only, not tracked in this repository).

---

## File Structure

- **Create** `engine/calibrate/coverage.py` — the completeness module: `ClassifyCoverage` dataclass, `LabeledIncidentsIncompleteError`, `write_classify_coverage(...)` (producer marker writer), `verify_labeled_completeness(...)` (flip-site verifier), and private snapshot-resolution/marker-read helpers. One responsibility: reconcile `labeled_incidents.json` to the pinned corpus snapshot.
- **Create** `tests/unit/test_coverage.py` — unit tests for the module (writer invariant + every verifier raise + the pass paths + the synthetic no-op).
- **Modify** `engine/cli/pipeline.py` (real classify path, ~line 419-426) — call `write_classify_coverage(...)` after `write_classify_artifacts(...)`.
- **Modify** `engine/cli/reclassify.py` — call `write_classify_coverage(...)` at its `labeled_incidents.json` write point (only if it writes one for a snapshot-bearing cycle; read it first).
- **Modify** `engine/cli/pipeline_executor.py` (`execute_infer_phase`, ~line 269-302) — call `verify_labeled_completeness(...)` after loading `labeled`, and again (with goldset ids) inside the gold block.
- **Modify** `engine/cli/calibration.py` (`cal_tally`, ~line 401-415) — parse `snapshot_hash` from the manifest; call `verify_labeled_completeness(...)` before `calibrate_with_gold(...)` when `_classifier_labels is not None`; echo a coverage line.
- **Modify** `tests/unit/test_pipeline_executor.py` and/or **Create** `tests/unit/test_completeness_wiring.py` — snapshot-bearing minimal-cycle fixture exercising the flip-site wiring (complete ⇒ pass; truncated ⇒ raise; missing marker ⇒ raise; legit-OOS ⇒ pass; synthetic ⇒ no-op).
- **Modify** `docs/superpowers/plans/LESSONS-rarr.md` and `.superpowers/sdd/progress.md` — record F-B outcome + the producer contract for Phase-3 winner-classify.

---

### Task 1: Completeness module (`engine/calibrate/coverage.py`)

**Files:**
- Create: `engine/calibrate/coverage.py`
- Test: `tests/unit/test_coverage.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Reads `cycle/corpora/<corpus>/<snapshot_hash>/incidents.json` (top-level `{"incidents": [{"id": ...}, ...]}`) and `cycle/classify/classify_coverage.json`.
- Produces:
  - `class LabeledIncidentsIncompleteError(RuntimeError)`
  - `@dataclass(frozen=True, slots=True) class ClassifyCoverage: snapshot_hash: str; n_corpus: int; n_in_scope: int; n_oos: int`
  - `COVERAGE_FILENAME = "classify_coverage.json"`
  - `write_classify_coverage(out_dir: Path, *, snapshot_hash: str, corpus_incident_ids: set[str], in_scope_incident_ids: set[str]) -> ClassifyCoverage`
  - `verify_labeled_completeness(cycle: Path, snapshot_hash: str, labeled_incident_ids: set[str], *, goldset_recall_ids: set[str] | None = None) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_coverage.py`:

```python
"""Unit tests for the F-B labeled_incidents completeness guard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.calibrate.coverage import (
    COVERAGE_FILENAME,
    ClassifyCoverage,
    LabeledIncidentsIncompleteError,
    verify_labeled_completeness,
    write_classify_coverage,
)


def _make_snapshot(cycle: Path, snapshot_hash: str, ids: list[str]) -> None:
    snap_dir = cycle / "corpora" / "genai_agentic" / snapshot_hash
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "incidents.json").write_text(
        json.dumps({"incident_count": len(ids), "incidents": [{"id": i} for i in ids]})
    )


def _write_marker(cycle: Path, **fields: object) -> None:
    (cycle / "classify").mkdir(parents=True, exist_ok=True)
    (cycle / "classify" / COVERAGE_FILENAME).write_text(json.dumps(fields))


# --- writer ---------------------------------------------------------------

def test_write_marker_records_split_and_invariant(tmp_path: Path) -> None:
    cov = write_classify_coverage(
        tmp_path / "classify",
        snapshot_hash="h",
        corpus_incident_ids={"a", "b", "c", "d"},
        in_scope_incident_ids={"a", "b"},
    )
    assert cov == ClassifyCoverage(snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=2)
    data = json.loads((tmp_path / "classify" / COVERAGE_FILENAME).read_text())
    assert data == {"snapshot_hash": "h", "n_corpus": 4, "n_in_scope": 2, "n_oos": 2}


def test_write_marker_rejects_in_scope_id_absent_from_corpus(tmp_path: Path) -> None:
    with pytest.raises(LabeledIncidentsIncompleteError, match="absent from the corpus"):
        write_classify_coverage(
            tmp_path / "classify",
            snapshot_hash="h",
            corpus_incident_ids={"a", "b"},
            in_scope_incident_ids={"a", "z"},
        )


# --- verifier: no-op paths ------------------------------------------------

def test_verify_noop_when_synthetic_snapshot(tmp_path: Path) -> None:
    # No corpora dir resolvable -> no authoritative universe -> no-op.
    verify_labeled_completeness(tmp_path, "synthetic-no-snapshot", {"a", "b"})


def test_verify_noop_when_snapshot_dir_absent(tmp_path: Path) -> None:
    verify_labeled_completeness(tmp_path, "deadbeef", {"a", "b"})


# --- verifier: pass path (genuine OOS does NOT raise) ---------------------

def test_verify_passes_with_complete_marker_and_genuine_oos(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])  # 4 corpus incidents
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=2)
    # labeled = {a, b}; c and d are genuinely OOS (absent). goldset references c
    # (a recall incident the classifier put OOS) -> must NOT raise.
    verify_labeled_completeness(tmp_path, "h", {"a", "b"}, goldset_recall_ids={"a", "c"})


# --- verifier: proven-gap raises -----------------------------------------

def test_verify_raises_on_foreign_labeled_id(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=2, n_in_scope=2, n_oos=0)
    with pytest.raises(LabeledIncidentsIncompleteError, match="absent from the pinned"):
        verify_labeled_completeness(tmp_path, "h", {"a", "z"})


def test_verify_raises_on_missing_marker(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b", "c"])
    with pytest.raises(LabeledIncidentsIncompleteError, match="coverage marker"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_on_marker_snapshot_hash_mismatch(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b"])
    _write_marker(tmp_path, snapshot_hash="OTHER", n_corpus=2, n_in_scope=2, n_oos=0)
    with pytest.raises(LabeledIncidentsIncompleteError, match="different snapshot"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_on_partial_corpus_coverage(tmp_path: Path) -> None:
    # Snapshot has 4 incidents but the producer recorded n_corpus=2 -> partial run.
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=2, n_in_scope=2, n_oos=0)
    with pytest.raises(LabeledIncidentsIncompleteError, match="partial corpus"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_on_inconsistent_marker(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=1)
    with pytest.raises(LabeledIncidentsIncompleteError, match="internally inconsistent"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_when_labeled_count_below_marker(tmp_path: Path) -> None:
    # Marker says 3 in-scope but labeled_incidents.json now has 2 -> truncated file.
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=3, n_oos=1)
    with pytest.raises(LabeledIncidentsIncompleteError, match="truncated"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_on_goldset_incident_absent_from_snapshot(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=2)
    with pytest.raises(LabeledIncidentsIncompleteError, match="goldset"):
        verify_labeled_completeness(
            tmp_path, "h", {"a", "b"}, goldset_recall_ids={"a", "MANUAL-X-001"}
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_coverage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.calibrate.coverage'`.

- [ ] **Step 3: Write the module**

Create `engine/calibrate/coverage.py`:

```python
"""Classify-completeness guard against the pinned corpus snapshot (RARR F-B).

Unit A's OOS policy (a) scores a goldset incident with no in-scope classifier
label as a recall MISS, encoding OOS as ABSENCE from labeled_incidents.json.
That makes absence overloaded: a genuinely truncated / partial classify run
(crash) is ALSO absence, and would silently corrupt recall (and the incidence
counts built from labeled_incidents.json) with no error.

This module restores a DISTINCT completeness guard keyed on the pinned corpus
snapshot (manifest.snapshot_hash -> corpora/<corpus>/<snapshot_hash>/
incidents.json) -- an authoritative record INDEPENDENT of the (truncatable)
classifier output, so it cannot be co-truncated.  The classify producer writes
a count-form completion marker (classify_coverage.json) certifying how many
snapshot incidents it issued a verdict for; verify_labeled_completeness()
reconciles that marker to the pinned snapshot and to labeled_incidents.json.

A PROVEN coverage gap RAISES (loud-fail posture, matching _verify_goldset_hash
and the present-but-broken-goldset guard).  A genuinely-OOS incident under a
complete marker does NOT raise -- policy (a) is preserved (it remains a miss).
When no pinned snapshot is resolvable (synthetic cycles use the sentinel
"synthetic-no-snapshot"; minimal fixtures have no corpora dir) this is a no-op:
there is no authoritative universe to reconcile against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

COVERAGE_FILENAME = "classify_coverage.json"
_SYNTHETIC_SNAPSHOT = "synthetic-no-snapshot"


class LabeledIncidentsIncompleteError(RuntimeError):
    """labeled_incidents.json fails to reconcile to the pinned corpus snapshot."""


@dataclass(frozen=True, slots=True)
class ClassifyCoverage:
    snapshot_hash: str
    n_corpus: int
    n_in_scope: int
    n_oos: int


def write_classify_coverage(
    out_dir: Path,
    *,
    snapshot_hash: str,
    corpus_incident_ids: set[str],
    in_scope_incident_ids: set[str],
) -> ClassifyCoverage:
    """Record, at classify completion, the producer's coverage of the snapshot.

    Call this ONLY after the classify/predict phase has run to completion over
    the full corpus universe (so its absence is itself the primary truncation
    signal).  ``corpus_incident_ids`` is the full snapshot universe the producer
    was given; ``in_scope_incident_ids`` are the incidents it assigned an
    in-scope entry (== the rows it wrote to labeled_incidents.json).  n_oos is
    the remainder (OOS by absence).  Raises if an in-scope label names an
    incident absent from the snapshot (the producer ran against wrong inputs).
    """
    foreign = in_scope_incident_ids - corpus_incident_ids
    if foreign:
        raise LabeledIncidentsIncompleteError(
            f"classify produced {len(foreign)} in-scope label(s) for incident(s) "
            f"absent from the corpus snapshot (e.g. {sorted(foreign)[:3]}); the "
            "labels do not belong to the pinned snapshot."
        )
    n_corpus = len(corpus_incident_ids)
    n_in_scope = len(in_scope_incident_ids)
    cov = ClassifyCoverage(
        snapshot_hash=snapshot_hash,
        n_corpus=n_corpus,
        n_in_scope=n_in_scope,
        n_oos=n_corpus - n_in_scope,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / COVERAGE_FILENAME).write_text(
        json.dumps(
            {
                "snapshot_hash": cov.snapshot_hash,
                "n_corpus": cov.n_corpus,
                "n_in_scope": cov.n_in_scope,
                "n_oos": cov.n_oos,
            },
            indent=2,
        )
        + "\n"
    )
    return cov


def _resolve_snapshot_incidents(cycle: Path, snapshot_hash: str) -> Path | None:
    """Locate corpora/<corpus>/<snapshot_hash>/incidents.json (corpus-agnostic).

    Returns None when no pinned snapshot is resolvable (synthetic sentinel,
    empty hash, or the snapshot dir is absent) -> the verifier no-ops.
    """
    if not snapshot_hash or snapshot_hash == _SYNTHETIC_SNAPSHOT:
        return None
    matches = sorted((cycle / "corpora").glob(f"*/{snapshot_hash}/incidents.json"))
    return matches[0] if matches else None


def _load_snapshot_ids(incidents_json: Path) -> set[str]:
    data = json.loads(incidents_json.read_text(encoding="utf-8"))
    incidents = data["incidents"] if isinstance(data, dict) else data
    return {str(rec["id"]) for rec in incidents}


def _read_coverage(path: Path) -> ClassifyCoverage:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ClassifyCoverage(
        snapshot_hash=str(data["snapshot_hash"]),
        n_corpus=int(data["n_corpus"]),
        n_in_scope=int(data["n_in_scope"]),
        n_oos=int(data["n_oos"]),
    )


def verify_labeled_completeness(
    cycle: Path,
    snapshot_hash: str,
    labeled_incident_ids: set[str],
    *,
    goldset_recall_ids: set[str] | None = None,
) -> None:
    """RAISE LabeledIncidentsIncompleteError on a PROVEN coverage gap.

    ``goldset_recall_ids`` (optional) are the goldset incident ids that will be
    SCORED against this classifier (i.e. recall labels with a non-None
    classifier_entry_id); each must be present in the pinned snapshot, else a
    goldset/snapshot provenance break is being silently scored as OOS-misses.
    """
    incidents_json = _resolve_snapshot_incidents(cycle, snapshot_hash)
    if incidents_json is None:
        return
    corpus_ids = _load_snapshot_ids(incidents_json)

    # (1) foreign / stale labeled file: a label for an incident absent from the
    #     pinned snapshot means labeled_incidents.json came from another run.
    foreign = labeled_incident_ids - corpus_ids
    if foreign:
        raise LabeledIncidentsIncompleteError(
            f"labeled_incidents.json references {len(foreign)} incident(s) absent "
            f"from the pinned corpus snapshot {snapshot_hash} "
            f"(e.g. {sorted(foreign)[:3]}); the labeled file does not match the snapshot."
        )

    # (2) the completion marker is REQUIRED for a snapshot-bearing cycle: its
    #     absence is the primary truncation signal (the producer writes it only
    #     after running to completion).
    cov_path = cycle / "classify" / COVERAGE_FILENAME
    if not cov_path.exists():
        raise LabeledIncidentsIncompleteError(
            f"classify coverage marker {cov_path} is missing: the classify phase "
            "did not record completion over the corpus snapshot, so a truncated / "
            "partial run cannot be distinguished from genuine OOS. Re-run classify "
            "to emit the coverage marker."
        )
    cov = _read_coverage(cov_path)

    # (3) marker must bind the SAME pinned snapshot.
    if cov.snapshot_hash != snapshot_hash:
        raise LabeledIncidentsIncompleteError(
            f"coverage marker snapshot_hash {cov.snapshot_hash!r} != manifest "
            f"snapshot_hash {snapshot_hash!r}: classify ran against a different snapshot."
        )

    # (4) anti-circularity: marker n_corpus must equal the INDEPENDENT pinned
    #     universe size (a partial run cannot fake having been given fewer).
    if cov.n_corpus != len(corpus_ids):
        raise LabeledIncidentsIncompleteError(
            f"coverage marker n_corpus={cov.n_corpus} != pinned snapshot size "
            f"{len(corpus_ids)}: classify covered a partial corpus."
        )

    # (5) marker invariant + reconciliation to the labeled file.
    if cov.n_in_scope + cov.n_oos != cov.n_corpus:
        raise LabeledIncidentsIncompleteError(
            f"coverage marker is internally inconsistent: n_in_scope("
            f"{cov.n_in_scope}) + n_oos({cov.n_oos}) != n_corpus({cov.n_corpus})."
        )
    if cov.n_in_scope != len(labeled_incident_ids):
        raise LabeledIncidentsIncompleteError(
            f"coverage marker n_in_scope={cov.n_in_scope} != labeled_incidents.json "
            f"count {len(labeled_incident_ids)}: labeled_incidents.json was truncated "
            "after the coverage marker was written."
        )

    # (6) goldset/snapshot provenance: every scored recall incident must be in
    #     the snapshot the classifier actually ran (else "absent from labeled"
    #     is a provenance break, not genuine OOS).
    if goldset_recall_ids is not None:
        missing = goldset_recall_ids - corpus_ids
        if missing:
            raise LabeledIncidentsIncompleteError(
                f"{len(missing)} goldset recall incident(s) are absent from the "
                f"pinned corpus snapshot {snapshot_hash} (e.g. {sorted(missing)[:3]}); "
                "the goldset does not match the snapshot the classifier ran."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_coverage.py -q`
Expected: PASS (12 tests).

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check .` then `uv run mypy engine tests`
Expected: both clean (no errors).

- [ ] **Step 6: Commit**

```bash
git add engine/calibrate/coverage.py tests/unit/test_coverage.py
git commit -m "feat(calibrate): add labeled_incidents completeness guard vs pinned snapshot (F-B)"
```

---

### Task 2: Producers emit the coverage marker

**Files:**
- Modify: `engine/cli/pipeline.py` (real classify path, after `write_classify_artifacts(...)`, ~line 419-426)
- Modify: `engine/cli/reclassify.py` (at its `labeled_incidents.json` write point — read the file first to locate it)
- Test: `tests/unit/test_pipeline_executor.py` (extend `TestClassifyPhase`)

**Interfaces:**
- Consumes: `write_classify_coverage(out_dir, *, snapshot_hash, corpus_incident_ids, in_scope_incident_ids) -> ClassifyCoverage` (Task 1).
- Produces: `cycle/classify/classify_coverage.json` alongside `labeled_incidents.json`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_pipeline_executor.py` inside `class TestClassifyPhase`:

```python
    def test_write_classify_coverage_marker(self, tmp_path: Path) -> None:
        from engine.calibrate.coverage import COVERAGE_FILENAME, write_classify_coverage

        out_dir = tmp_path / "classify"
        cov = write_classify_coverage(
            out_dir,
            snapshot_hash="snap123",
            corpus_incident_ids={"INC-001", "INC-002", "INC-003"},
            in_scope_incident_ids={"INC-001", "INC-002"},
        )
        assert cov.n_corpus == 3
        assert cov.n_in_scope == 2
        assert cov.n_oos == 1
        data = json.loads((out_dir / COVERAGE_FILENAME).read_text())
        assert data["snapshot_hash"] == "snap123"
        assert data["n_in_scope"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_pipeline_executor.py::TestClassifyPhase::test_write_classify_coverage_marker -v`
Expected: PASS already if Task 1 landed — this test only exercises the Task 1 helper, pinning the producer-facing contract. If it PASSES, that is the expected state (it documents the marker shape the producers below rely on). Proceed.

- [ ] **Step 3: Wire the marker into the real classify path (`engine/cli/pipeline.py`)**

In the classify-real `try` block, immediately after the existing `write_classify_artifacts(...)` call (currently ~line 422-426) and before `click.echo(f"Classify phase complete...")`, add:

```python
        from engine.calibrate.coverage import read_snapshot_universe_ids, write_classify_coverage

        write_classify_coverage(
            out_dir,
            snapshot_hash=manifest_data.get("snapshot_hash", ""),
            corpus_incident_ids=read_snapshot_universe_ids(snapshot_dir / "incidents.json"),
            in_scope_incident_ids={c.incident_id for c in result.classifications},
        )
```

`manifest_data`, `snapshot_dir`, and `result` are all already in scope (see `pipeline.py:279`, `:313`, `:413-417`).

> **CORRECTED (final-review Critical #1):** `corpus_incident_ids` MUST come from `read_snapshot_universe_ids(snapshot_dir / "incidents.json")` — the SAME raw, UNFILTERED reader the verifier uses — **not** `{inc.id for inc in incidents_list}`. `incidents_list = adapter.iter_incidents()` is DATE-FILTERED (`GenAIAgenticAdapter` drops `date > snapshot_date`), so it undercounts the raw snapshot (7713 vs 7714 on the real 2026 corpus) and the verifier's `n_corpus == len(corpus_ids)` guard would false-positive on a complete run. The producer and verifier share one reader so their universes cannot drift. The same correction applies to the **Phase-3 winner-classify producer contract**.

- [ ] **Step 4: Wire the marker into the multimodel reclassify path (`engine/cli/reclassify.py`)**

Read `engine/cli/reclassify.py` fully. Locate where it writes `labeled_incidents.json`. Immediately after that write, add a `write_classify_coverage(...)` call using: the cycle's `classify` out-dir, `snapshot_hash` from the manifest it loads (or `corpus_dir` provenance), the **full** corpus universe id-set it iterated (the snapshot's incident ids — read from the resolved `corpora/<corpus>/<hash>/incidents.json`, NOT from the reclassify checkpoint, which is a routed subset), and the in-scope ids it wrote to `labeled_incidents.json`. If reclassify does NOT resolve the full snapshot universe, load it via the same glob the verifier uses (`corpus_dir.glob("*/<snapshot_hash>/incidents.json")`) and take `{rec["id"] for rec in data["incidents"]}`.

If, on reading, `reclassify.py` does not itself write `labeled_incidents.json` for a snapshot-bearing cycle (e.g. it only writes a `_multimodel` variant that a later step promotes), wire the marker at whichever step finalizes `cycle/classify/labeled_incidents.json`, and note the actual writer in the commit message. Do not guess — anchor the call at the real write site.

- [ ] **Step 5: Run the full suite + lint + type-check**

Run: `uv run pytest -q`
Expected: PASS (existing classify tests still green — `write_classify_artifacts` is unchanged; the new marker write is additive).
Run: `uv run ruff check .` then `uv run mypy engine tests`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add engine/cli/pipeline.py engine/cli/reclassify.py tests/unit/test_pipeline_executor.py
git commit -m "feat(classify): emit classify_coverage.json completion marker at producers (F-B)"
```

---

### Task 3: Wire the verifier into the two recall-flip sites

**Files:**
- Modify: `engine/cli/pipeline_executor.py` (`execute_infer_phase`, ~line 268-302)
- Modify: `engine/cli/calibration.py` (`cal_tally`, ~line 386-419)
- Create: `tests/unit/test_completeness_wiring.py`

**Interfaces:**
- Consumes: `verify_labeled_completeness(cycle, snapshot_hash, labeled_incident_ids, *, goldset_recall_ids=None)` (Task 1); the existing `load_gold_calibration` / `load_classifier_labels` (`engine/calibrate/gold_loader.py`).
- Produces: a raise on a proven coverage gap at both flip sites; a one-line coverage echo in `cal_tally`.

- [ ] **Step 1: Write the failing wiring tests**

Create `tests/unit/test_completeness_wiring.py`:

```python
"""Flip-site wiring for the F-B completeness guard (execute_infer_phase, cal_tally)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

from pathlib import Path

import pytest

from engine.calibrate.coverage import (
    COVERAGE_FILENAME,
    LabeledIncidentsIncompleteError,
    write_classify_coverage,
)


def _snapshot(cycle: Path, snap: str, ids: list[str]) -> None:
    d = cycle / "corpora" / "genai_agentic" / snap
    d.mkdir(parents=True, exist_ok=True)
    (d / "incidents.json").write_text(
        json.dumps({"incident_count": len(ids), "incidents": [{"id": i} for i in ids]})
    )


def _labeled(cycle: Path, rows: list[tuple[str, str]]) -> None:
    d = cycle / "classify"
    d.mkdir(parents=True, exist_ok=True)
    (d / "labeled_incidents.json").write_text(
        json.dumps([
            {"incident_id": iid, "entry_id": eid, "confidence": 0.9,
             "stage": 1, "rationale": "x", "stratum": "security"}
            for iid, eid in rows
        ])
    )


def test_verify_passes_on_complete_cycle(tmp_path: Path) -> None:
    # The verifier itself is the unit under test here (the flip sites call it).
    from engine.calibrate.coverage import verify_labeled_completeness

    cycle = tmp_path / "cycle"
    _snapshot(cycle, "snap", ["INC-1", "INC-2", "INC-3"])
    _labeled(cycle, [("INC-1", "LLM01"), ("INC-2", "LLM02")])
    write_classify_coverage(
        cycle / "classify",
        snapshot_hash="snap",
        corpus_incident_ids={"INC-1", "INC-2", "INC-3"},
        in_scope_incident_ids={"INC-1", "INC-2"},
    )
    labeled = json.loads((cycle / "classify" / "labeled_incidents.json").read_text())
    labeled_ids = {str(r["incident_id"]) for r in labeled}
    # INC-3 is a genuine-OOS goldset recall incident -> must NOT raise.
    verify_labeled_completeness(cycle, "snap", labeled_ids, goldset_recall_ids={"INC-1", "INC-3"})


def test_infer_raises_on_truncated_labeled(tmp_path: Path) -> None:
    from engine.cli.pipeline_executor import execute_infer_phase

    cycle = tmp_path / "cycle"
    _snapshot(cycle, "snap", ["INC-1", "INC-2", "INC-3", "INC-4"])
    _labeled(cycle, [("INC-1", "LLM01"), ("INC-2", "LLM02")])  # only 2 rows
    # Marker claims 3 in-scope but the labeled file has 2 -> truncated.
    (cycle / "classify" / COVERAGE_FILENAME).write_text(
        json.dumps({"snapshot_hash": "snap", "n_corpus": 4, "n_in_scope": 3, "n_oos": 1})
    )
    (cycle / "calibration").mkdir(parents=True)
    (cycle / "calibration" / "posteriors.json").write_text(json.dumps({"recall": {}, "precision": {}}))
    (cycle / "prereg").mkdir(parents=True)
    (cycle / "prereg" / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "snapshot_hash": "snap",
    }))
    with pytest.raises(LabeledIncidentsIncompleteError, match="truncated"):
        execute_infer_phase(cycle)
```

NOTE on the infer test: `execute_infer_phase` parses the manifest via `_load_manifest`, which constructs `PreregManifest(**filtered)`. If `PreregManifest` requires fields beyond `schema_version`/`snapshot_hash`, build the manifest JSON with the minimal required field set (mirror `tests/unit/test_prereg.py::_make_manifest` defaults, serialized to JSON) so construction succeeds and the completeness check (which runs right after `labeled = json.loads(...)`) is reached before any heavier failure. If aligning the manifest is fiddly, assert the wiring at the verifier seam instead: call the private `_run_completeness_for_infer(cycle, manifest)` helper you extract, OR keep the infer assertion as a direct `verify_labeled_completeness(...)` call on the fixture (the `test_verify_passes_on_complete_cycle` pattern) and rely on Step-3 code review to confirm the call is present at the flip site. Prefer the real `execute_infer_phase` path if the manifest builds cleanly.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_completeness_wiring.py -q`
Expected: `test_verify_passes_on_complete_cycle` PASSES (verifier exists); the infer-raise test FAILS (verifier not yet wired into `execute_infer_phase`).

- [ ] **Step 3: Wire into `execute_infer_phase` (`engine/cli/pipeline_executor.py`)**

After `labeled = json.loads(labeled_path.read_text())` (currently ~line 269) and before `_build_counts_from_labeled(...)`, add the labeled-vs-snapshot guard (runs for ALL cycles; protects incidence counts too):

```python
    from engine.calibrate.coverage import verify_labeled_completeness

    _labeled_ids = {str(item["incident_id"]) for item in labeled}
    verify_labeled_completeness(cycle, manifest.snapshot_hash, _labeled_ids)
```

Then inside the `if _has_gold_files:` block, immediately after `_verify_goldset_hash(manifest, _gold)` and before `build_overlap_from_confusion(...)`, add the goldset/snapshot provenance guard:

```python
            verify_labeled_completeness(
                cycle, manifest.snapshot_hash, _labeled_ids,
                goldset_recall_ids={
                    lbl.incident_id for lbl in _gold.recall_labels
                    if lbl.classifier_entry_id is not None
                },
            )
```

`LabeledIncidentsIncompleteError` is a `RuntimeError`, which is NOT in the `except (ValueError, OSError, json.JSONDecodeError)` tuple wrapping the gold load — so it propagates as itself rather than being re-wrapped. Confirm the import of `verify_labeled_completeness` sits with the other `from engine.calibrate...` imports in that block (ruff I001 — use `uv run ruff check --fix` to order).

- [ ] **Step 4: Wire into `cal_tally` (`engine/cli/calibration.py`)**

Inside `if gold_calibration is not None:`, after `_classifier_labels = (...)` and after `gold = load_gold_calibration(...)` but before `calibrate_with_gold(...)` (currently ~line 401-415), add:

```python
        if _classifier_labels is not None:
            from engine.calibrate.coverage import verify_labeled_completeness

            _manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            verify_labeled_completeness(
                cycle,
                str(_manifest_data.get("snapshot_hash", "")),
                set(_classifier_labels),
                goldset_recall_ids={
                    lbl.incident_id for lbl in gold.recall_labels
                    if lbl.classifier_entry_id is not None
                },
            )
            click.echo(
                f"Completeness check passed: {len(_classifier_labels)} classifier "
                "labels reconcile to the pinned corpus snapshot."
            )
```

(The check runs only when `_classifier_labels is not None` — i.e. when recall is actually flipped to the classifier; the backward-compat `llm_consensus` path is unaffected.)

- [ ] **Step 5: Run the wiring tests**

Run: `uv run pytest tests/unit/test_completeness_wiring.py -q`
Expected: PASS (both/all tests).

- [ ] **Step 6: Run the FULL suite + lint + type-check**

Run: `uv run pytest -q`
Expected: PASS, including the F4 pin `tests/unit/test_recall_single_label_semantics.py` and all existing `cal_tally` / `execute_infer_phase` tests (they use non-snapshot fixtures ⇒ verifier no-ops). If a pre-existing test now fails because its fixture IS snapshot-bearing but lacks a marker, ADD the marker to that fixture (a legitimate fixture update, not a weakening) — do NOT relax the guard.
Run: `uv run ruff check .` then `uv run mypy engine tests`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add engine/cli/pipeline_executor.py engine/cli/calibration.py tests/unit/test_completeness_wiring.py
git commit -m "feat(calibrate): enforce labeled completeness at infer + tally recall-flip sites (F-B)"
```

---

### Task 4: Ledger update + push + CI

**Files:**
- Modify: `docs/superpowers/plans/LESSONS-rarr.md`
- Modify: `.superpowers/sdd/progress.md`

- [ ] **Step 1: Record the F-B outcome in `LESSONS-rarr.md`**

Append under the Phase-2 section: F-B done; reference set = pinned corpus snapshot; action = hard raise on a proven gap; count-form marker `classify_coverage.json` written at producers (`pipeline.py` real classify; `reclassify.py` finalizer); verifier wired at `execute_infer_phase` (labeled-vs-snapshot + goldset/snapshot) and `cal_tally`. Note the **Phase-3 producer contract**: the bake-off winner-classify path MUST call `write_classify_coverage(...)` over the full corpus universe, else the flip sites raise "coverage marker missing." Record that the frozen 2026 cycle has no marker (and is never re-inferred) so CI is unaffected. Note any real number movement (expected: none — the frozen 2026 goldset is 100% in-scope; this is build-time infrastructure).

- [ ] **Step 2: Record in `.superpowers/sdd/progress.md`**

Add a `## F-B — labeled_incidents completeness check` block: BASE commit, commit range, review verdicts, full-suite-green confirmation, CI status.

- [ ] **Step 3: Commit the ledger**

```bash
git add docs/superpowers/plans/LESSONS-rarr.md .superpowers/sdd/progress.md
git commit -m "docs(lessons): record F-B labeled completeness guard + Phase-3 producer contract"
```

- [ ] **Step 4: Final full verification before push**

Run: `uv run pytest -q` (FULL, no `-k`) — Expected: all green incl. F4 pin.
Run: `uv run ruff check .` and `uv run mypy engine tests` — Expected: clean.

- [ ] **Step 5: Push to PR #22 and verify CI green**

```bash
git push origin plan7/engine-upgrade-recall-pl
```

Then watch CI: `gh pr checks 22 --watch` (or `gh run list --branch plan7/engine-upgrade-recall-pl --limit 3`). Expected: ubuntu + macos + cross-platform-diff all green. If red, triage from the CI log (it runs `uv sync --frozen --extra narrative` → `ruff check .` → `mypy engine tests` → `pytest -v` → semgrep → cyclonedx → cosign → synthetic run).

---

## Self-Review

**Spec coverage (the two decisions + the prompt's F-B framing):**
- "Keep OOS-as-miss" — preserved: `_load_recall_from_adjudicated` / `calibrate_with_gold` untouched; the verifier only RAISES on a proven coverage gap, never on a genuine-OOS incident under a complete marker (Task 1 `test_verify_passes_with_complete_marker_and_genuine_oos`). ✔
- "Add a DISTINCT completeness check that catches genuine truncation" — `verify_labeled_completeness`, wired at both flip sites. ✔
- Decision 1 (reference = pinned corpus snapshot, producer commits coverage) — `_resolve_snapshot_incidents` + `classify_coverage.json` marker anchored to `manifest.snapshot_hash`. ✔
- Decision 2 (hard raise on proven gap) — `LabeledIncidentsIncompleteError` raised; marker IS the persisted provenance record; `cal_tally` echoes a pass line. ✔
- Both named flip sites covered: `execute_infer_phase` (overlap W) + `cal_tally` (recall). ✔
- Producer coverage (`_build_counts_from_labeled` / `write_classify_artifacts` lineage): marker emitted at the real classify writer + reclassify finalizer. ✔

**Placeholder scan:** No "TBD/handle edge cases" — every step has concrete code or a concrete read-then-wire instruction (Task 2 Step 4 / Task 3 Step 1 note give explicit fallbacks rather than hand-waving). ✔

**Type consistency:** `ClassifyCoverage(snapshot_hash, n_corpus, n_in_scope, n_oos)`, `write_classify_coverage(out_dir, *, snapshot_hash, corpus_incident_ids, in_scope_incident_ids)`, `verify_labeled_completeness(cycle, snapshot_hash, labeled_incident_ids, *, goldset_recall_ids=None)`, `COVERAGE_FILENAME`, `LabeledIncidentsIncompleteError` — names/signatures identical across Tasks 1–3. The snapshot id field is `id`; the labeled id field is `incident_id`; both reconcile to the same id space (verified: 0 foreign on the real 2026 cycle). ✔

**F4 pin:** No change to `engine/calibrate/tally.py`; Task 3 Step 6 explicitly re-verifies the pin. ✔
